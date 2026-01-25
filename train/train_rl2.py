#!/usr/bin/env python3
"""
Recurrent PPO with (frozen) vision encoder + GRU + policy/value heads,
and optional message encoder/head (disabled / dummy in single-agent warmup).

This matches the architecture you referenced:

  obs -> FrozenVisionEncoder -> (adapter) -> GRU -> policy/value heads
  msg_in -> MessageEncoder ----------------^

  GRU hidden -> MessageHead -> msg_out  (only used in multi-agent)

This script is:
- online (on-policy), synchronous PPO
- parallel env collection via gymnasium AsyncVectorEnv
- pure PyTorch (no SB3), so you can control frozen encoders and recurrence

USAGE (single-agent warmup; messages disabled):
  python train_recurrent_ppo_comm.py \
    --env doom \
    --scenario_cfg /path/to/my_way_home.cfg \
    --n_envs 32 --total_steps 30000000 \
    --use_messages 0

Later (multi-agent), you can enable messages and feed msg_in from other agents.
This file includes the policy architecture and PPO trainer; multi-agent wiring of
message passing is a small extension (not included unless you want it).

Requirements:
  pip install torch gymnasium numpy opencv-python vizdoom

NOTE:
- ViZDoom wrapper included is minimal and uses a small discrete action set.
- If you already have your own env wrapper, you can swap `make_env()` easily.
"""

import os
import time
import math
import argparse
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# -----------------------------
# (Optional) ViZDoom environment wrapper
# -----------------------------
try:
    from vizdoom import DoomGame, Mode, ScreenResolution, ScreenFormat, Button
    import cv2
    VIZDOOM_OK = True
except Exception:
    VIZDOOM_OK = False


def build_doom_action_map(buttons: List["Button"]) -> List[List[float]]:
    btn_to_idx = {b: i for i, b in enumerate(buttons)}

    def act(*pressed: "Button") -> List[float]:
        a = [0.0] * len(buttons)
        for p in pressed:
            if p in btn_to_idx:
                a[btn_to_idx[p]] = 1.0
        return a

    actions = [
        act(),  # NOOP
        act(Button.MOVE_FORWARD),
        act(Button.MOVE_BACKWARD),
        act(Button.MOVE_LEFT),
        act(Button.MOVE_RIGHT),
        act(Button.TURN_LEFT),
        act(Button.TURN_RIGHT),
        act(Button.TURN_LEFT, Button.MOVE_FORWARD),
        act(Button.TURN_RIGHT, Button.MOVE_FORWARD),
        act(Button.USE),
    ]
    if Button.ATTACK in buttons:
        actions += [act(Button.ATTACK), act(Button.ATTACK, Button.MOVE_FORWARD)]
    return actions


class VizDoomGym(gym.Env):
    """
    Minimal ViZDoom -> Gymnasium wrapper.
    Observation: uint8 grayscale (1,H,W) or RGB (3,H,W)
    Action: Discrete combos
    Reward: ViZDoom game reward
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        cfg_path: str,
        grayscale: bool = True,
        obs_h: int = 84,
        obs_w: int = 84,
        frame_skip: int = 4,
        max_episode_steps: int = 2100,
        seed: int = 0,
        render: bool = False,
    ):
        super().__init__()
        if not VIZDOOM_OK:
            raise RuntimeError("vizdoom/cv2 not available. Install vizdoom + opencv-python.")

        self.cfg_path = cfg_path
        self.grayscale = grayscale
        self.obs_h = obs_h
        self.obs_w = obs_w
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self._elapsed = 0
        self._seed = seed
        self._render = render

        self.game = DoomGame()
        self.game.load_config(cfg_path)
        self.game.set_seed(seed)
        self.game.set_mode(Mode.PLAYER)
        self.game.set_screen_resolution(ScreenResolution.RES_160X120)
        self.game.set_screen_format(ScreenFormat.GRAY8 if grayscale else ScreenFormat.RGB24)
        self.game.set_window_visible(render)
        self.game.init()

        self.buttons = self.game.get_available_buttons()
        self.actions = build_doom_action_map(self.buttons)

        c = 1 if grayscale else 3
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(c, obs_h, obs_w), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(len(self.actions))

    def _obs(self) -> np.ndarray:
        state = self.game.get_state()
        if state is None or state.screen_buffer is None:
            return np.zeros(self.observation_space.shape, dtype=np.uint8)
        buf = state.screen_buffer

        if self.grayscale:
            if buf.ndim == 2:
                img = buf
            else:
                img = buf[0] if buf.shape[0] in (1, 3) else buf[..., 0]
            img = cv2.resize(img, (self.obs_w, self.obs_h), interpolation=cv2.INTER_AREA)
            return img[None, :, :].astype(np.uint8)
        else:
            # Handle (C,H,W) or (H,W,C)
            if buf.ndim == 3 and buf.shape[0] == 3:
                img = np.transpose(buf, (1, 2, 0))
            elif buf.ndim == 3 and buf.shape[-1] == 3:
                img = buf
            else:
                raise RuntimeError(f"Unexpected RGB buffer shape: {buf.shape}")
            img = cv2.resize(img, (self.obs_w, self.obs_h), interpolation=cv2.INTER_AREA)
            return np.transpose(img, (2, 0, 1)).astype(np.uint8)

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._elapsed = 0
        if seed is not None:
            self.game.set_seed(int(seed))
        self.game.new_episode()
        return self._obs(), {}

    def step(self, action: int):
        self._elapsed += 1
        r = float(self.game.make_action(self.actions[int(action)], self.frame_skip))
        terminated = self.game.is_episode_finished()
        truncated = self._elapsed >= self.max_episode_steps
        obs = self._obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.uint8)
        return obs, r, terminated, truncated, {}

    def close(self):
        try:
            self.game.close()
        except Exception:
            pass


# -----------------------------
# Model: Frozen vision -> adapter -> GRU -> policy/value
# Optional message encoder/head
# -----------------------------

class FrozenVisionEncoder(nn.Module):
    """
    Replace this with your real frozen encoder (CLIP/DINO/etc).
    For now: a small CNN that is frozen by default.

    Output: feature vector [B, feat_dim]
    """
    def __init__(self, in_ch: int, feat_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        # Infer flattened size with a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, 84, 84)
            n = self.net(dummy).shape[1]
        self.proj = nn.Linear(n, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return self.proj(x)


class ActorCriticComm(nn.Module):
    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        n_actions: int,
        *,
        vision_feat_dim: int = 512,
        adapter_dim: int = 256,
        gru_hidden: int = 256,
        use_messages: bool = False,
        msg_dim: int = 16,         # size of incoming/outgoing messages
        msg_enc_dim: int = 64,     # message encoder output
        freeze_vision: bool = True,
    ):
        super().__init__()
        c, h, w = obs_shape
        self.use_messages = use_messages
        self.msg_dim = msg_dim

        # Frozen vision encoder
        self.vision = FrozenVisionEncoder(in_ch=c, feat_dim=vision_feat_dim)
        if freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False
            self.vision.eval()  # keep in eval mode by default (no dropout/bn anyway)

        # Trainable adapter to map frozen features -> control-friendly embedding
        self.adapter = nn.Sequential(
            nn.LayerNorm(vision_feat_dim),
            nn.Linear(vision_feat_dim, adapterl:=adapter_dim),
            nn.ReLU(),
        )

        # Message encoder (optional)
        if use_messages:
            self.msg_encoder = nn.Sequential(
                nn.LayerNorm(msg_dim),
                nn.Linear(msg_dim, msg_enc_dim),
                nn.ReLU(),
            )
            gru_in = adapter_dim + msg_enc_dim
        else:
            self.msg_encoder = None
            gru_in = adapter_dim

        self.gru = nn.GRU(input_size=gru_in, hidden_size=gru_hidden, num_layers=1)

        # Policy & value heads
        self.policy = nn.Linear(gru_hidden, n_actions)
        self.value = nn.Linear(gru_hidden, 1)

        # Message head (optional): from GRU hidden to msg_out
        if use_messages:
            self.msg_head = nn.Sequential(
                nn.Linear(gru_hidden, msg_dim),
                nn.Tanh(),  # keeps messages bounded; you can swap this
            )
        else:
            self.msg_head = None

    @torch.no_grad()
    def init_hidden(self, batch: int, device: torch.device) -> torch.Tensor:
        # GRU expects [num_layers, batch, hidden]
        return torch.zeros(1, batch, self.gru.hidden_size, device=device)

    def forward(
        self,
        obs_u8: torch.Tensor,            # [B,C,H,W] uint8 or float
        h: torch.Tensor,                 # [1,B,H]
        msg_in: Optional[torch.Tensor],  # [B,msg_dim] or None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Returns:
          logits: [B,n_actions]
          value:  [B]
          msg_out: [B,msg_dim] or None
          h_next: [1,B,H]
        """
        # Convert obs to float in [0,1]
        if obs_u8.dtype != torch.float32:
            obs = obs_u8.float() / 255.0
        else:
            obs = obs_u8

        # Vision (frozen) + adapter
        feat = self.vision(obs)
        z = self.adapter(feat)  # [B,adapter_dim]

        if self.use_messages:
            assert msg_in is not None, "use_messages=True but msg_in is None"
            m = self.msg_encoder(msg_in)
            x = torch.cat([z, m], dim=-1)
        else:
            x = z

        # GRU expects seq-first: [T,B,dim]
        x_seq = x.unsqueeze(0)
        y_seq, h_next = self.gru(x_seq, h)
        y = y_seq.squeeze(0)  # [B,gru_hidden]

        logits = self.policy(y)
        value = self.value(y).squeeze(-1)

        msg_out = self.msg_head(y) if self.use_messages else None
        return logits, value, msg_out, h_next


# -----------------------------
# PPO utilities (GAE, batching)
# -----------------------------

@dataclass
class PPOConfig:
    n_envs: int = 32
    n_steps: int = 128
    total_steps: int = 30_000_000

    gamma: float = 0.99
    gae_lambda: float = 0.95

    clip_coef: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    lr: float = 2.5e-4
    update_epochs: int = 4
    minibatch_size: int = 4096  # will be adjusted to divide rollout size

    target_kl: Optional[float] = 0.02

    save_every_steps: int = 2_000_000
    log_every_steps: int = 200_000


def compute_gae(
    rewards: torch.Tensor,      # [T, N]
    values: torch.Tensor,       # [T, N]
    dones: torch.Tensor,        # [T, N] (1 if done at step t)
    last_value: torch.Tensor,   # [N]
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns advantages [T,N] and returns [T,N]
    """
    T, N = rewards.shape
    adv = torch.zeros(T, N, device=rewards.device)
    gae = torch.zeros(N, device=rewards.device)

    for t in reversed(range(T)):
        next_nonterminal = 1.0 - dones[t]
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        gae = delta + gamma * lam * next_nonterminal * gae
        adv[t] = gae

    ret = adv + values
    return adv, ret


# -----------------------------
# Trainer
# -----------------------------

class PPOTrainer:
    def __init__(
        self,
        envs: gym.vector.VectorEnv,
        model: ActorCriticComm,
        cfg: PPOConfig,
        device: torch.device,
        run_dir: str,
        use_messages: bool,
        msg_dim: int,
    ):
        self.envs = envs
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.run_dir = run_dir
        self.use_messages = use_messages
        self.msg_dim = msg_dim

        os.makedirs(run_dir, exist_ok=True)
        self.optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

        # Hidden state per env (single-agent warmup: one policy per env)
        self.h = self.model.init_hidden(cfg.n_envs, device)

        # Messages per env (single-agent warmup: zeros)
        self.msg_in = torch.zeros(cfg.n_envs, msg_dim, device=device)

        self.global_step = 0
        self.start_time = time.time()
        self.next_log = cfg.log_every_steps
        self.next_save = cfg.save_every_steps

    def save(self, name: str):
        path = os.path.join(self.run_dir, name)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "cfg": vars(self.cfg),
            },
            path,
        )
        print(f"[save] {path}")

    def collect_rollout(self, obs0: np.ndarray) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        device = self.device
        self.model.train()  # only affects adapter/heads; frozen vision stays frozen

        obs = torch.as_tensor(obs0, device=device)

        # Storage (T,N, ...)
        obs_buf = torch.zeros(cfg.n_steps, cfg.n_envs, *obs.shape[1:], device=device, dtype=obs.dtype)
        act_buf = torch.zeros(cfg.n_steps, cfg.n_envs, device=device, dtype=torch.long)
        logp_buf = torch.zeros(cfg.n_steps, cfg.n_envs, device=device)
        rew_buf = torch.zeros(cfg.n_steps, cfg.n_envs, device=device)
        done_buf = torch.zeros(cfg.n_steps, cfg.n_envs, device=device)
        val_buf = torch.zeros(cfg.n_steps, cfg.n_envs, device=device)

        # For recurrent policies, we also store h at each step
        h_buf = torch.zeros(cfg.n_steps, *self.h.shape, device=device)

        # Message inputs (if enabled). For warmup, it's zeros; for multi-agent you’d fill it.
        msg_in = self.msg_in

        for t in range(cfg.n_steps):
            obs_buf[t] = obs
            h_buf[t] = self.h

            with torch.no_grad():
                logits, value, msg_out, h_next = self.model(obs, self.h, msg_in if self.use_messages else None)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

            val_buf[t] = value
            act_buf[t] = action
            logp_buf[t] = logp

            # Step envs (CPU)
            act_np = action.cpu().numpy()
            next_obs, rew, term, trunc, _ = self.envs.step(act_np)
            done = np.logical_or(term, trunc)

            rew_buf[t] = torch.as_tensor(rew, device=device, dtype=torch.float32)
            done_buf[t] = torch.as_tensor(done.astype(np.float32), device=device)

            # Update hidden: reset where done
            self.h = h_next
            if done.any():
                done_t = torch.as_tensor(done, device=device)
                self.h[:, done_t, :] = 0.0

            # Messages:
            # - single-agent warmup: keep msg_in as zeros
            # - if you enable messages later, you'd compute msg_in from other agents' msg_out
            if self.use_messages:
                # In a true multi-agent setup, msg_out would be sent to others.
                # Here we do NOT feed it back to self by default. Keep msg_in as zeros unless you wire it.
                pass

            obs = torch.as_tensor(next_obs, device=device)

            self.global_step += cfg.n_envs

        # Bootstrap value
        with torch.no_grad():
            logits, last_val, _, _ = self.model(obs, self.h, msg_in if self.use_messages else None)

        return {
            "obs": obs_buf,
            "actions": act_buf,
            "logp": logp_buf,
            "rewards": rew_buf,
            "dones": done_buf,
            "values": val_buf,
            "last_value": last_val,
            "h": h_buf,
        }, obs.cpu().numpy()

    def update(self, batch: Dict[str, torch.Tensor]):
        cfg = self.cfg
        device = self.device
        self.model.train()

        rewards = batch["rewards"]
        values = batch["values"]
        dones = batch["dones"]
        last_value = batch["last_value"]

        adv, ret = compute_gae(rewards, values, dones, last_value, cfg.gamma, cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Flatten T,N -> B
        T, N = rewards.shape
        B = T * N

        obs = batch["obs"].reshape(B, *batch["obs"].shape[2:])
        actions = batch["actions"].reshape(B)
        old_logp = batch["logp"].reshape(B)
        returns = ret.reshape(B)
        adv_flat = adv.reshape(B)

        # For recurrence: we keep it simple by treating each step independently for backprop-through-time
        # across n_steps by using stored h at each step. This is a common "truncated BPTT" PPO pattern.
        h = batch["h"].reshape(T, 1, N, self.model.gru.hidden_size).permute(0, 2, 1, 3).reshape(B, 1, self.model.gru.hidden_size)
        h = h.permute(1, 0, 2).contiguous()  # [1,B,H]

        # Messages for warmup: zeros. (Multi-agent: you’ll provide real msg_in aligned to obs.)
        if self.use_messages:
            msg_in = torch.zeros(B, self.msg_dim, device=device)
        else:
            msg_in = None

        # Adjust minibatch size to divide rollout size
        mb = cfg.minibatch_size
        if mb > B:
            mb = B
        # Make mb a divisor of B if possible
        for cand in [mb, 2048, 1024, 512, 256, 128]:
            if B % cand == 0:
                mb = cand
                break

        inds = torch.arange(B, device=device)

        approx_kl = 0.0
        for epoch in range(cfg.update_epochs):
            perm = inds[torch.randperm(B)]
            for start in range(0, B, mb):
                mb_inds = perm[start:start + mb]

                logits, v, _, _ = self.model(obs[mb_inds], h[:, mb_inds, :], msg_in[mb_inds] if msg_in is not None else None)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(actions[mb_inds])
                entropy = dist.entropy().mean()

                ratio = (logp - old_logp[mb_inds]).exp()
                pg_loss1 = -adv_flat[mb_inds] * ratio
                pg_loss2 = -adv_flat[mb_inds] * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * (returns[mb_inds] - v).pow(2).mean()

                loss = pg_loss - cfg.ent_coef * entropy + cfg.vf_coef * v_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_logp[mb_inds] - logp).mean().item()

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        return approx_kl

    def train(self):
        obs, _ = self.envs.reset()
        while self.global_step < self.cfg.total_steps:
            batch, obs = self.collect_rollout(obs)
            kl = self.update(batch)

            if self.global_step >= self.next_log:
                elapsed = time.time() - self.start_time
                fps = int(self.global_step / max(elapsed, 1e-9))
                # Simple logging: mean rollout return over the last rollout
                mean_rew = batch["rewards"].sum(dim=0).mean().item()
                print(f"[step {self.global_step:,}] fps~{fps:,}  rollout_return_mean={mean_rew:.3f}  approx_kl={kl:.4f}")
                self.next_log += self.cfg.log_every_steps

            if self.global_step >= self.next_save:
                self.save(f"checkpoint_{self.global_step}.pt")
                self.next_save += self.cfg.save_every_steps

        self.save("final.pt")


# -----------------------------
# Env factory
# -----------------------------

def make_env_doom(cfg_path: str, seed: int, grayscale: bool, render: bool) -> Callable[[], gym.Env]:
    def _thunk():
        env = VizDoomGym(
            cfg_path=cfg_path,
            grayscale=grayscale,
            obs_h=84,
            obs_w=84,
            frame_skip=4,
            max_episode_steps=2100,
            seed=seed,
            render=render,
        )
        return env
    return _thunk


# -----------------------------
# Main
# -----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="doom", choices=["doom"])
    p.add_argument("--scenario_cfg", type=str, required=True, help="Path to ViZDoom .cfg scenario")
    p.add_argument("--run_dir", type=str, default="./runs_recurrent_ppo")
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--n_envs", type=int, default=32)
    p.add_argument("--n_steps", type=int, default=128)
    p.add_argument("--total_steps", type=int, default=30_000_000)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--use_messages", type=int, default=0, help="0: disable (single-agent warmup). 1: enable modules.")
    p.add_argument("--msg_dim", type=int, default=16)

    p.add_argument("--freeze_vision", type=int, default=1)
    p.add_argument("--vision_feat_dim", type=int, default=512)
    p.add_argument("--adapter_dim", type=int, default=256)
    p.add_argument("--gru_hidden", type=int, default=256)

    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--target_kl", type=float, default=0.02)

    p.add_argument("--grayscale", type=int, default=1)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.run_dir, exist_ok=True)

    # Vectorized envs (parallel CPU processes)
    if args.env == "doom":
        if not VIZDOOM_OK:
            raise RuntimeError("vizdoom/cv2 not available. Install vizdoom + opencv-python.")
        env_fns = [
            make_env_doom(args.scenario_cfg, seed=args.seed + i * 1000, grayscale=bool(args.grayscale), render=False)
            for i in range(args.n_envs)
        ]
        envs = gym.vector.AsyncVectorEnv(env_fns)
    else:
        raise ValueError("Unknown env")

    obs_shape = envs.single_observation_space.shape  # (C,H,W)
    n_actions = envs.single_action_space.n

    model = ActorCriticComm(
        obs_shape=obs_shape,
        n_actions=n_actions,
        vision_feat_dim=args.vision_feat_dim,
        adapter_dim=args.adapter_dim,
        gru_hidden=args.gru_hidden,
        use_messages=bool(args.use_messages),
        msg_dim=args.msg_dim,
        freeze_vision=bool(args.freeze_vision),
    )

    cfg = PPOConfig(
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        total_steps=args.total_steps,
        lr=args.lr,
        clip_coef=args.clip,
        ent_coef=args.ent,
        vf_coef=args.vf,
        update_epochs=args.epochs,
        minibatch_size=args.minibatch,
        target_kl=args.target_kl,
    )

    trainer = PPOTrainer(
        envs=envs,
        model=model,
        cfg=cfg,
        device=device,
        run_dir=args.run_dir,
        use_messages=bool(args.use_messages),
        msg_dim=args.msg_dim,
    )

    print(f"Obs shape: {obs_shape}, actions: {n_actions}")
    print(f"use_messages={bool(args.use_messages)} freeze_vision={bool(args.freeze_vision)} device={device}")
    trainer.train()
    envs.close()


if __name__ == "__main__":
    main()
