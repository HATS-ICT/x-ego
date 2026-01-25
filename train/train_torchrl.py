#!/usr/bin/env python3
"""
TorchRL PPO with:
  Frozen vision encoder -> adapter -> GRU -> policy head + value head

Supports:
- Single-agent: obs shape [C,H,W], action shape []
- Multi-agent: obs shape [A,C,H,W], action shape [A]
  (parameter sharing is implemented by flattening [B,A,...] -> [B*A,...])

You must provide an env factory `make_env()` that returns a TorchRL EnvBase,
or wrap a gymnasium env via GymWrapper (see notes).

Core idea:
- Set --n_agents 1 for warmup
- Set --n_agents K for multi-agent, IF your env returns agent-dim tensors.

Dependencies:
  pip install torch tensordict torchrl gymnasium

If you use VizDoom:
  pip install vizdoom opencv-python
and write a gymnasium wrapper (or EnvBase) that outputs the right shapes.

This file focuses on the training+model side; the env side is pluggable.
"""

import os
import time
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn

from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from torchrl.envs import ParallelEnv
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import Categorical
from torchrl.envs.utils import ExplorationType, set_exploration_type


# -------------------------
# Config
# -------------------------

@dataclass
class PPOCfg:
    # env/collector
    n_envs: int = 32
    n_agents: int = 1
    frames_per_batch: int = 32 * 128  # must be n_envs * rollout_len (or close)
    rollout_len: int = 128

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.1
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 2.5e-4
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 4096
    target_kl: float | None = 0.02

    # training length
    total_env_steps: int = 30_000_000

    # logging/checkpoints
    log_every_steps: int = 200_000
    save_every_steps: int = 2_000_000


# -------------------------
# Model components
# -------------------------

class FrozenVisionEncoder(nn.Module):
    """
    Replace this with your real frozen encoder (CLIP/DINO/etc).
    Output: [B, feat_dim]
    """
    def __init__(self, in_ch: int, feat_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, 84, 84)
            n = self.net(dummy).shape[-1]
        self.proj = nn.Linear(n, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x))


class SharedRecurrentActorCritic(nn.Module):
    """
    Works for both single- and multi-agent if you feed:
      obs: [B,C,H,W]  (single)  OR  [B,A,C,H,W] (multi)

    Internally flattens agent dimension (parameter sharing):
      [B,A,...] -> [B*A,...]
    """
    def __init__(
        self,
        obs_channels: int,
        n_actions: int,
        *,
        vision_feat_dim: int = 512,
        adapter_dim: int = 256,
        rnn_hidden: int = 256,
        freeze_vision: bool = True,
        use_messages: bool = False,
        msg_dim: int = 16,
        msg_enc_dim: int = 64,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.use_messages = use_messages
        self.msg_dim = msg_dim
        self.rnn_hidden = rnn_hidden

        self.vision = FrozenVisionEncoder(obs_channels, feat_dim=vision_feat_dim)
        if freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False
            self.vision.eval()

        self.adapter = nn.Sequential(
            nn.LayerNorm(vision_feat_dim),
            nn.Linear(vision_feat_dim, adapter_dim),
            nn.ReLU(),
        )

        if use_messages:
            self.msg_encoder = nn.Sequential(
                nn.LayerNorm(msg_dim),
                nn.Linear(msg_dim, msg_enc_dim),
                nn.ReLU(),
            )
            rnn_in = adapter_dim + msg_enc_dim
        else:
            self.msg_encoder = None
            rnn_in = adapter_dim

        # GRU over time. TorchRL collector will give us time-major rollouts.
        self.gru = nn.GRU(input_size=rnn_in, hidden_size=rnn_hidden, num_layers=1)

        self.policy_head = nn.Linear(rnn_hidden, n_actions)
        self.value_head = nn.Linear(rnn_hidden, 1)

        # message head is included for later, but you typically disable it in single-agent warmup
        if use_messages:
            self.msg_head = nn.Sequential(nn.Linear(rnn_hidden, msg_dim), nn.Tanh())
        else:
            self.msg_head = None

    def forward(self, td: TensorDict) -> TensorDict:
        """
        Expects:
          td["obs"]  : uint8 image
             single: [T,B,C,H,W] or [B,C,H,W]
             multi : [T,B,A,C,H,W] or [B,A,C,H,W]
          td["rnn_h"]: [B,A,H] or [B,H] (optional, if present)
          td["msg_in"]: [B,A,msg_dim] or [B,msg_dim] (optional)
        Produces:
          td["logits"], td["value"], td["action"] (via ProbabilisticActor), td["rnn_h_next"]
        """
        obs = td["obs"]
        # Normalize image
        obs = obs.float() / 255.0 if obs.dtype != torch.float32 else obs

        # Handle optional time dimension (collector typically provides [T, ...])
        has_time = obs.dim() in (5, 6)  # [T,B,C,H,W] or [T,B,A,C,H,W]
        if has_time:
            T = obs.shape[0]
            # merge time into batch for vision+adapter, then un-merge for GRU
            obs_flat = obs.reshape(-1, *obs.shape[-3:])
        else:
            T = None
            obs_flat = obs

        # Multi-agent? (agent dim present after B)
        # If has_time: shapes are [T,B,...]
        # We'll flatten [B,A] into one batch for shared parameters, then restore.
        if has_time:
            # after flattening time, shape is [(T*B*A), C,H,W] or [(T*B), C,H,W]
            # but if multi-agent, we need to include A in that flattening.
            pass

        # Determine if multi-agent by checking original obs dims
        # single no-time: [B,C,H,W] dim=4
        # multi  no-time: [B,A,C,H,W] dim=5
        # single time   : [T,B,C,H,W] dim=5
        # multi  time   : [T,B,A,C,H,W] dim=6
        multi = (obs.dim() == 5 and not has_time) or (obs.dim() == 6 and has_time)

        if has_time:
            if multi:
                # obs: [T,B,A,C,H,W] -> flatten to [T*B*A, C,H,W]
                T, B, A = obs.shape[0], obs.shape[1], obs.shape[2]
                obs_flat = obs.reshape(T * B * A, *obs.shape[-3:])
            else:
                # obs: [T,B,C,H,W] -> [T*B, C,H,W]
                T, B = obs.shape[0], obs.shape[1]
                obs_flat = obs.reshape(T * B, *obs.shape[-3:])
        else:
            if multi:
                # obs: [B,A,C,H,W] -> [B*A, C,H,W]
                B, A = obs.shape[0], obs.shape[1]
                obs_flat = obs.reshape(B * A, *obs.shape[-3:])
            else:
                # obs: [B,C,H,W]
                B = obs.shape[0]
                A = 1
                obs_flat = obs

        with torch.no_grad():
            feat = self.vision(obs_flat)
        z = self.adapter(feat)

        if self.use_messages:
            msg_in = td.get("msg_in", None)
            if msg_in is None:
                raise RuntimeError("use_messages=True but msg_in not found in tensordict.")
            if has_time:
                if multi:
                    msg_flat = msg_in.reshape(T * B * A, -1)
                else:
                    msg_flat = msg_in.reshape(T * B, -1)
            else:
                msg_flat = msg_in.reshape(B * A, -1) if multi else msg_in
            m = self.msg_encoder(msg_flat)
            x = torch.cat([z, m], dim=-1)
        else:
            x = z

        # Now feed GRU. GRU expects [T, batch, dim]. If no time, treat as T=1.
        if not has_time:
            x_seq = x.unsqueeze(0)  # [1, B*A, dim]
            # hidden state: [1, batch, H]
            h0 = td.get("rnn_h", None)
            if h0 is None:
                h0 = torch.zeros(1, x_seq.shape[1], self.rnn_hidden, device=x_seq.device)
            else:
                # rnn_h stored as [B,A,H] or [B,H]; flatten to [1, B*A, H]
                if multi:
                    h0 = h0.reshape(1, B * A, self.rnn_hidden)
                else:
                    h0 = h0.reshape(1, B, self.rnn_hidden)

            y_seq, h1 = self.gru(x_seq, h0)
            y = y_seq.squeeze(0)  # [B*A,H]
        else:
            # has_time: x is [T*B*A, dim] or [T*B, dim] -> reshape to [T, B*A, dim]
            batch = B * A if multi else B
            x_seq = x.reshape(T, batch, -1)

            h0 = td.get("rnn_h", None)
            if h0 is None:
                h0 = torch.zeros(1, batch, self.rnn_hidden, device=x_seq.device)
            else:
                # assume h0 provided as [B,A,H] (or [B,H]), flatten to [1, batch, H]
                if multi:
                    h0 = h0.reshape(1, B * A, self.rnn_hidden)
                else:
                    h0 = h0.reshape(1, B, self.rnn_hidden)

            y_seq, h1 = self.gru(x_seq, h0)
            y = y_seq.reshape(T * batch, self.rnn_hidden)

        logits = self.policy_head(y)
        value = self.value_head(y).squeeze(-1)

        # reshape outputs back to match input batching (so TorchRL can compute losses)
        if has_time:
            if multi:
                logits = logits.reshape(T, B, A, self.n_actions)
                value = value.reshape(T, B, A)
            else:
                logits = logits.reshape(T, B, self.n_actions)
                value = value.reshape(T, B)
        else:
            if multi:
                logits = logits.reshape(B, A, self.n_actions)
                value = value.reshape(B, A)
            else:
                logits = logits.reshape(B, self.n_actions)
                value = value.reshape(B)

        td.set("logits", logits)
        td.set("state_value", value)

        # save next hidden state in a shape consistent with multi/single
        if multi:
            h1_out = h1.reshape(1, B, A, self.rnn_hidden).squeeze(0) if not has_time else h1.reshape(1, B, A, self.rnn_hidden).squeeze(0)
        else:
            h1_out = h1.reshape(1, B, self.rnn_hidden).squeeze(0)

        td.set("rnn_h_next", h1_out)

        # optional outgoing message head (disabled in warmup by setting use_messages=False)
        if self.use_messages and self.msg_head is not None:
            # message computed from last y (no-time) or from all y (time)
            msg = self.msg_head(y)
            if has_time:
                msg = msg.reshape(T, B, A, self.msg_dim) if multi else msg.reshape(T, B, self.msg_dim)
            else:
                msg = msg.reshape(B, A, self.msg_dim) if multi else msg.reshape(B, self.msg_dim)
            td.set("msg_out", msg)

        return td


# -------------------------
# Environment plumbing
# -------------------------

def make_env_stub(n_agents: int):
    """
    Placeholder environment factory.

    Replace this with your actual ViZDoom TorchRL env.
    The key contract for easy switching is:
      td["obs"] shape:
        single-agent: [C,H,W]
        multi-agent : [A,C,H,W]
      td["reward"] shape:
        single-agent: []
        multi-agent : [A] or [] (team reward) (both can work, but be consistent)
      td["done"] shape:
        single-agent: []
        multi-agent : [] (episode done for the env)

    If you already have a gymnasium env, wrap it with torchrl.envs.GymWrapper.
    """
    from torchrl.envs import EnvBase
    from torchrl.data import Composite, UnboundedContinuousTensorSpec, DiscreteTensorSpec, BinaryDiscreteTensorSpec

    class StubEnv(EnvBase):
        def __init__(self):
            super().__init__()
            C, H, W = 1, 84, 84
            self._n_agents = n_agents
            obs_shape = (C, H, W) if n_agents == 1 else (n_agents, C, H, W)

            self.observation_spec = Composite(
                obs=UnboundedContinuousTensorSpec(shape=obs_shape),
                done=BinaryDiscreteTensorSpec(shape=()),
                terminated=BinaryDiscreteTensorSpec(shape=()),
            )
            self.action_spec = DiscreteTensorSpec(10, shape=() if n_agents == 1 else (n_agents,))
            self.reward_spec = UnboundedContinuousTensorSpec(shape=() if n_agents == 1 else (n_agents,))

        def _reset(self, td=None):
            obs = torch.zeros(self.observation_spec["obs"].shape)
            return TensorDict(
                {"obs": obs, "done": torch.zeros(()), "terminated": torch.zeros(())},
                batch_size=[],
            )

        def _step(self, td):
            # fake transition
            obs = torch.zeros(self.observation_spec["obs"].shape)
            reward = torch.zeros(self.reward_spec.shape)
            done = torch.zeros(())
            terminated = torch.zeros(())
            return TensorDict(
                {"obs": obs, "reward": reward, "done": done, "terminated": terminated},
                batch_size=[],
            )

    return StubEnv

# -------------------------
# Training
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default="./runs_torchrl_ppo")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--n_envs", type=int, default=32)
    ap.add_argument("--n_agents", type=int, default=1)
    ap.add_argument("--total_steps", type=int, default=30_000_000)
    ap.add_argument("--rollout_len", type=int, default=128)
    ap.add_argument("--frames_per_batch", type=int, default=0, help="0 = n_envs*rollout_len")

    ap.add_argument("--freeze_vision", type=int, default=1)
    ap.add_argument("--use_messages", type=int, default=0)
    ap.add_argument("--msg_dim", type=int, default=16)

    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--target_kl", type=float, default=0.02)

    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.run_dir, exist_ok=True)

    cfg = PPOCfg(
        n_envs=args.n_envs,
        n_agents=args.n_agents,
        rollout_len=args.rollout_len,
        total_env_steps=args.total_steps,
        lr=args.lr,
        clip_eps=args.clip,
        entropy_coef=args.ent,
        vf_coef=args.vf,
        ppo_epochs=args.epochs,
        minibatch_size=args.minibatch,
        target_kl=args.target_kl if args.target_kl > 0 else None,
    )
    if args.frames_per_batch and args.frames_per_batch > 0:
        cfg.frames_per_batch = args.frames_per_batch
    else:
        cfg.frames_per_batch = cfg.n_envs * cfg.rollout_len

    # ---- Env: replace make_env_stub with your ViZDoom env factory
    env_cls = make_env_stub(cfg.n_agents)
    make_env = lambda: env_cls()
    env = ParallelEnv(cfg.n_envs, make_env)  # CPU-parallel stepping

    # Inspect specs to build model
    obs_spec = env.observation_spec["obs"]
    act_spec = env.action_spec
    if cfg.n_agents == 1:
        obs_channels = obs_spec.shape[0]
        n_actions = act_spec.space.n
    else:
        obs_channels = obs_spec.shape[1]  # [A,C,H,W]
        n_actions = act_spec.space.n

    model_core = SharedRecurrentActorCritic(
        obs_channels=obs_channels,
        n_actions=n_actions,
        freeze_vision=bool(args.freeze_vision),
        use_messages=bool(args.use_messages),
        msg_dim=args.msg_dim,
    ).to(device)

    # TorchRL expects modules that read/write keys in a TensorDict.
    # We'll produce "logits" and "state_value".
    core_td_module = TensorDictModule(
        model_core,
        in_keys=["obs", "rnn_h", "msg_in"] if bool(args.use_messages) else ["obs", "rnn_h"],
        out_keys=["logits", "state_value", "msg_out", "rnn_h_next"] if bool(args.use_messages) else ["logits", "state_value", "rnn_h_next"],
    )

    # Actor: logits -> categorical distribution -> action
    actor = ProbabilisticActor(
        module=core_td_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=Categorical,
        return_log_prob=True,            # needed for PPO
        log_prob_key="action_log_prob",
        spec=env.action_spec,
    )

    # Critic: reads "state_value" produced by core
    critic = ValueOperator(
        module=core_td_module,
        in_keys=[],  # state_value already computed inside core module
        out_keys=["state_value"],
    )

    # Advantage estimator
    gae = GAE(
        gamma=cfg.gamma,
        lmbda=cfg.gae_lambda,
        value_network=critic,
        average_gae=True,
    )

    # PPO loss
    loss = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg.clip_eps,
        entropy_coef=cfg.entropy_coef,
        critic_coef=cfg.vf_coef,
        normalize_advantage=True,
    )

    optim_ = torch.optim.Adam([p for p in model_core.parameters() if p.requires_grad], lr=cfg.lr)

    # Rollout collector
    collector = SyncDataCollector(
        env,
        policy=actor,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_env_steps,
        device=device,          # policy device
        storing_device="cpu",   # keep rollouts on CPU, move minibatches to GPU
        split_trajs=False,
    )

    # On-policy buffer (TorchRL uses a replay-buffer-like object for batching)
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.frames_per_batch),
        batch_size=cfg.minibatch_size,
    )

    # Logging/checkpoint
    start = time.time()
    next_log = cfg.log_every_steps
    next_save = cfg.save_every_steps
    global_steps = 0

    def save_ckpt(tag: str):
        path = os.path.join(args.run_dir, f"{tag}.pt")
        torch.save(
            {
                "model": model_core.state_dict(),
                "optim": optim_.state_dict(),
                "steps": global_steps,
                "cfg": cfg.__dict__,
            },
            path,
        )
        print(f"[save] {path}")

    # ---- Training loop
    with set_exploration_type(ExplorationType.RANDOM):
        for batch_td in collector:
            # batch_td is a TensorDict with time/batch dims depending on collector.
            # We compute advantages, then optimize PPO for a few epochs.
            global_steps += cfg.frames_per_batch

            # For warmup: provide dummy rnn_h / msg_in if missing
            # If your env already emits these, remove this section.
            if "rnn_h" not in batch_td.keys(True, True):
                # Create zeros hidden state per step (TorchRL will carry it in a real recurrent rollout if wired via env/policy)
                # This keeps the example runnable; for best RNN training, store/propagate rnn_h properly.
                pass
            if bool(args.use_messages) and "msg_in" not in batch_td.keys(True, True):
                # Dummy zeros messages
                # Shape should align with obs batch shape (single or multi)
                pass

            # Advantage estimation (adds "advantage" and "value_target" keys)
            gae(batch_td)

            # Load to buffer, then sample minibatches
            rb.extend(batch_td.reshape(-1))  # flatten time/env dims

            approx_kl = 0.0
            for _ in range(cfg.ppo_epochs):
                for mb in rb:
                    mb = mb.to(device)
                    loss_td = loss(mb)
                    total_loss = loss_td["loss_objective"] + loss_td["loss_critic"] + loss_td["loss_entropy"]

                    optim_.zero_grad(set_to_none=True)
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(model_core.parameters(), cfg.max_grad_norm)
                    optim_.step()

                    if "approx_kl" in loss_td.keys():
                        approx_kl = float(loss_td["approx_kl"].detach().cpu())

                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    break

            rb.empty()

            if global_steps >= next_log:
                elapsed = time.time() - start
                fps = int(global_steps / max(elapsed, 1e-9))
                # crude return proxy: mean reward in batch
                mean_rew = float(batch_td.get("reward").mean().cpu())
                print(f"[step {global_steps:,}] fps~{fps:,} mean_reward={mean_rew:.4f} approx_kl={approx_kl:.4f}")
                next_log += cfg.log_every_steps

            if global_steps >= next_save:
                save_ckpt(f"ckpt_{global_steps}")
                next_save += cfg.save_every_steps

    save_ckpt("final")
    env.close()


if __name__ == "__main__":
    main()
