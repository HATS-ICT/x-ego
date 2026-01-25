#!/usr/bin/env python3
"""
Single-agent online PPO training for ViZDoom with parallel environments (CPU),
to learn a decent "actor" policy you can later transfer to multi-agent MAPPO.

What you get:
- A Gymnasium-compatible ViZDoom env (image obs, discrete actions)
- SubprocVecEnv parallelization (many Doom instances)
- Stable-Baselines3 PPO training (online, on-policy)
- Periodic checkpointing + best-model saving
- Quick evaluation run

Install (typical):
  pip install vizdoom gymnasium opencv-python stable-baselines3[extra]

Run:
  python train_vizdoom_ppo_parallel.py \
    --config /path/to/scenarios/my_way_home.cfg \
    --total-timesteps 30000000 \
    --n-envs 32 \
    --save-dir ./runs_mwh

Notes:
- Use a navigation-ish scenario first (e.g., MyWayHome). This script does NOT do
  any manual curriculum, demos, or offline RL.
- Increase --n-envs to use your Threadripper. GPU mostly helps PPO update speed.
"""

import os
import time
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2

import gymnasium as gym
from gymnasium import spaces

from vizdoom import DoomGame, Mode, ScreenResolution, ScreenFormat, Button, GameVariable

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed


# -----------------------------
# Environment
# -----------------------------

@dataclass
class DoomEnvConfig:
    cfg_path: str
    frame_skip: int = 4
    render: bool = False
    obs_width: int = 84
    obs_height: int = 84
    grayscale: bool = True
    max_episode_steps: int = 2100  # ~35s at 60Hz with frame_skip=4
    seed: int = 0


def _build_action_map(buttons: List[Button]) -> List[List[float]]:
    """
    Build a small discrete action set (button combos). This is "standard RL Doom":
    limited discrete actions that still allow navigation/combat.

    You can edit this list later without changing the rest of the training stack.
    """
    # Helper: create an action vector for the selected buttons
    btn_to_idx = {b: i for i, b in enumerate(buttons)}

    def act(*pressed: Button) -> List[float]:
        a = [0.0] * len(buttons)
        for p in pressed:
            if p in btn_to_idx:
                a[btn_to_idx[p]] = 1.0
        return a

    # A conservative navigation-first set:
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
        act(Button.USE),  # open door / use switch
    ]

    # If the scenario supports ATTACK and you want it:
    if Button.ATTACK in buttons:
        actions += [
            act(Button.ATTACK),
            act(Button.ATTACK, Button.MOVE_FORWARD),
        ]

    return actions


class VizDoomGymEnv(gym.Env):
    """
    Minimal Gymnasium wrapper over ViZDoom.
    Observation: (C, H, W) uint8 image (grayscale or RGB)
    Action: Discrete over a small set of button combinations
    Reward: from ViZDoom's game.get_last_reward()
    """

    metadata = {"render_modes": ["human"], "render_fps": 35}

    def __init__(self, cfg: DoomEnvConfig):
        super().__init__()
        self.cfg = cfg

        self.game = DoomGame()
        self.game.load_config(cfg.cfg_path)

        # Keep things deterministic-ish per env when seeding:
        self.game.set_seed(cfg.seed)

        # Make sure we're in PLAYER mode for RL:
        self.game.set_mode(Mode.PLAYER)

        # Use a fast screen format:
        self.game.set_screen_resolution(ScreenResolution.RES_160X120)
        self.game.set_screen_format(ScreenFormat.GRAY8 if cfg.grayscale else ScreenFormat.RGB24)

        # If you want to see it, set render=True. (Note: rendering slows stepping.)
        if cfg.render:
            self.game.set_window_visible(True)
        else:
            self.game.set_window_visible(False)

        self.game.init()

        self.buttons = self.game.get_available_buttons()
        self.actions = _build_action_map(self.buttons)

        # Observation space
        c = 1 if cfg.grayscale else 3
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(c, cfg.obs_height, cfg.obs_width), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(len(self.actions))

        self._elapsed_steps = 0

    def _get_obs(self) -> np.ndarray:
        state = self.game.get_state()
        # If episode ended, get_state can be None; guard it.
        if state is None or state.screen_buffer is None:
            # Return zeros to satisfy interface; episode should terminate.
            return np.zeros(self.observation_space.shape, dtype=np.uint8)

        buf = state.screen_buffer  # shape: (H, W) for GRAY8 OR (C, H, W) for some formats
        # ViZDoom returns:
        # - GRAY8: (H, W)
        # - RGB24: (C, H, W) in some bindings; in others (H, W, C). We'll handle both.

        if self.cfg.grayscale:
            if buf.ndim == 2:
                img = buf
            else:
                # If it comes with channel dim, squeeze it
                img = buf[0] if buf.shape[0] in (1, 3) else buf[..., 0]
            img = cv2.resize(img, (self.cfg.obs_width, self.cfg.obs_height), interpolation=cv2.INTER_AREA)
            obs = img[None, :, :]  # (1, H, W)
        else:
            if buf.ndim == 3 and buf.shape[0] == 3:
                img = np.transpose(buf, (1, 2, 0))  # (H, W, 3)
            elif buf.ndim == 3 and buf.shape[-1] == 3:
                img = buf  # already (H, W, 3)
            else:
                raise RuntimeError(f"Unexpected RGB buffer shape: {buf.shape}")

            img = cv2.resize(img, (self.cfg.obs_width, self.cfg.obs_height), interpolation=cv2.INTER_AREA)
            obs = np.transpose(img, (2, 0, 1))  # (3, H, W)

        return obs.astype(np.uint8)

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._elapsed_steps = 0

        if seed is not None:
            # Mix provided seed with base cfg seed for reproducibility
            mixed = (int(seed) * 1000003 + self.cfg.seed) & 0x7FFFFFFF
            self.game.set_seed(mixed)

        self.game.new_episode()
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: int):
        self._elapsed_steps += 1

        # Repeat action for frame_skip steps (standard Doom RL trick)
        reward = self.game.make_action(self.actions[int(action)], self.cfg.frame_skip)
        terminated = self.game.is_episode_finished()
        truncated = False

        if not terminated:
            if self._elapsed_steps >= self.cfg.max_episode_steps:
                truncated = True
                # Force end the episode cleanly
                self.game.close()
                # Re-init quickly (ViZDoom doesn't always like force-ending; simplest is re-init)
                self.game = DoomGame()
                self.game.load_config(self.cfg.cfg_path)
                self.game.set_seed(self.cfg.seed)
                self.game.set_mode(Mode.PLAYER)
                self.game.set_screen_resolution(ScreenResolution.RES_160X120)
                self.game.set_screen_format(ScreenFormat.GRAY8 if self.cfg.grayscale else ScreenFormat.RGB24)
                self.game.set_window_visible(self.cfg.render)
                self.game.init()

        obs = self._get_obs()
        info = {}
        return obs, float(reward), terminated, truncated, info

    def render(self):
        # Rendering is handled by ViZDoom window visibility; nothing needed here.
        return None

    def close(self):
        try:
            self.game.close()
        except Exception:
            pass


# -----------------------------
# Callbacks
# -----------------------------

class PrintTrainStatsCallback(BaseCallback):
    """
    Prints periodic stats without needing TensorBoard.
    """
    def __init__(self, print_every_steps: int = 200_000, verbose: int = 1):
        super().__init__(verbose)
        self.print_every_steps = print_every_steps
        self._next = print_every_steps
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            elapsed = time.time() - self.start_time
            fps = int(self.num_timesteps / max(elapsed, 1e-9))
            if self.verbose:
                print(f"[train] steps={self.num_timesteps:,}  fps~{fps:,}")
            self._next += self.print_every_steps
        return True


# -----------------------------
# VecEnv factory
# -----------------------------

def make_env(rank: int, base_cfg: DoomEnvConfig, seed: int):
    def _init():
        cfg = DoomEnvConfig(**vars(base_cfg))
        cfg.seed = seed + rank * 1000
        env = VizDoomGymEnv(cfg)
        # Optional: add TimeLimit wrapper for truncation instead of internal logic.
        return env
    return _init


# -----------------------------
# Main training script
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to ViZDoom .cfg scenario")
    parser.add_argument("--save-dir", type=str, default="./runs_vizdoom_ppo")
    parser.add_argument("--total-timesteps", type=int, default=30_000_000)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=2100)
    parser.add_argument("--grayscale", action="store_true", default=True)
    parser.add_argument("--rgb", action="store_true", default=False)
    parser.add_argument("--render-eval", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()

    if args.rgb:
        grayscale = False
    else:
        grayscale = True  # default

    os.makedirs(args.save_dir, exist_ok=True)

    set_random_seed(args.seed)

    base_cfg = DoomEnvConfig(
        cfg_path=args.config,
        frame_skip=args.frame_skip,
        render=False,
        grayscale=grayscale,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
    )

    # Parallel envs for training
    env_fns = [make_env(i, base_cfg, args.seed) for i in range(args.n_envs)]
    vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    vec_env = VecMonitor(vec_env)  # logs episode returns/lengths
    vec_env = VecTransposeImage(vec_env)  # SB3 CNN expects (C,H,W)

    # Separate eval env (single instance, optionally render)
    eval_cfg = DoomEnvConfig(**vars(base_cfg))
    eval_cfg.render = args.render_eval
    eval_env = VizDoomGymEnv(eval_cfg)
    eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env)
    eval_env = VecTransposeImage(gym.vector.SyncVectorEnv([lambda: eval_env]))

    # PPO hyperparams that tend to work well for Doom-like pixel tasks
    # (You will still tune, but this is a solid start.)
    n_steps = 128  # per env rollout length
    batch_size = 4096  # large batches usually help PPO stability
    # Ensure batch_size divides (n_steps * n_envs)
    rollout_size = n_steps * args.n_envs
    if batch_size > rollout_size:
        batch_size = rollout_size
    # Make batch_size a factor of rollout_size
    for b in [4096, 2048, 1024, 512, 256]:
        if rollout_size % b == 0:
            batch_size = min(batch_size, b)
            break

    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        learning_rate=2.5e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,   # keep exploration alive early
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=os.path.join(args.save_dir, "tb"),
        device=args.device,
    )

    # Checkpoint every N steps (total env steps)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, 2_000_000 // args.n_envs),  # roughly every 2M steps overall
        save_path=os.path.join(args.save_dir, "checkpoints"),
        name_prefix="ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Eval callback to keep best model
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=os.path.join(args.save_dir, "best"),
        log_path=os.path.join(args.save_dir, "eval"),
        eval_freq=max(1, 1_000_000 // args.n_envs),
        n_eval_episodes=10,
        deterministic=True,
        render=args.render_eval,
    )

    print_cb = PrintTrainStatsCallback(print_every_steps=2_000_000)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, eval_cb, print_cb],
        log_interval=args.log_interval,
        progress_bar=True,
    )

    final_path = os.path.join(args.save_dir, "final_model.zip")
    model.save(final_path)
    print(f"Saved final model to: {final_path}")

    # Quick evaluation rollout
    print("Running a quick evaluation (10 episodes)...")
    obs = eval_env.reset()
    ep_returns = []
    ep_return = 0.0
    episodes = 0
    while episodes < 10:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)
        ep_return += float(rewards[0])
        if bool(dones[0]):
            ep_returns.append(ep_return)
            ep_return = 0.0
            episodes += 1
    print(f"Eval returns: mean={np.mean(ep_returns):.3f}  std={np.std(ep_returns):.3f}")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
