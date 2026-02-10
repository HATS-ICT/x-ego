#!/usr/bin/env python3
"""
PettingZoo ParallelEnv wrapper for ViZDoom multiplayer coop (host + joiners),
based on your demo script but refactored into a single reset()/step() API.

Design goals:
- Works as a PettingZoo ParallelEnv (simultaneous actions)
- Uses multiprocessing workers (one ViZDoom instance per agent)
- Observation = state.screen_buffer (uint8 HxWxC)
- Action space = Discrete(N) mapping to predefined button vectors
- Sync mode (Mode.PLAYER) and skip=1 recommended

Notes:
- ViZDoom multiplayer requires all players to call new_episode(); handled in reset.
- In lockstep, a single slow worker can stall. Remove sleeps/prints.
- Rendering windows are optional; set window_visible=False for training.
"""

from __future__ import annotations

import os
import signal
import traceback
from dataclasses import dataclass
from itertools import count
# Global counter so all envs get unique ports across get_env_fun calls.
_ENV_PORT_COUNTER = count(0)
from multiprocessing import Process, Pipe
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import deque

import numpy as np
import vizdoom as vzd

# PettingZoo / Gymnasium
from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces

# BenchMARL / TorchRL
import copy
from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING
from torchrl.data import CompositeSpec
from torchrl.envs import EnvBase, PettingZooWrapper


# -----------------------------
# Configuration dataclass
# -----------------------------
@dataclass
class VZCoopConfig:
    config_path: str
    num_players: int = 2  # includes host as player_0
    host_ip: str = "127.0.0.1"

    # ViZDoom runtime
    mode: vzd.Mode = vzd.Mode.PLAYER
    ticrate: int = vzd.DEFAULT_TICRATE
    skip: int = 1  # IMPORTANT: keep 1 for easiest lockstep
    screen_resolution: vzd.ScreenResolution = vzd.ScreenResolution.RES_320X240
    window_visible: bool = False
    console_enabled: bool = False

    # Optional engine args
    timelimit_minutes: Optional[int] = None  # if not None, set +timelimit
    extra_game_args: str = ""  # appended to each instance

    # Port handling (avoid collisions when multiple envs are spawned)
    base_port: int = 5029
    port_stride: int = 10

    # Optional: window tiling for debugging
    tile_windows: bool = False
    win_x: int = 80
    win_y: int = 80


# -----------------------------
# Action set (Phase 1 discrete)
# -----------------------------
def build_default_buttons_and_actions() -> Tuple[List[vzd.Button], List[List[int]]]:
    """
    A reasonable small discrete action set. Adjust to your scenario if needed.

    IMPORTANT:
    - The action vectors must match the exact order of available buttons.
    - Here we explicitly set available buttons (recommended).
    """
    buttons = [
        vzd.Button.MOVE_FORWARD,
        vzd.Button.TURN_LEFT,
        vzd.Button.TURN_RIGHT,
        vzd.Button.STRAFE,
        vzd.Button.ATTACK,
        vzd.Button.USE,
    ]

    # Action vectors correspond to the buttons list above (len=7):
    # [FWD, TL, TR, SL, SR, ATT, USE]
    actions = [
        [0, 0, 0, 0, 0, 0],  # 0: NOOP
        [1, 0, 0, 0, 0, 0],  # 1: forward
        [0, 1, 0, 0, 0, 0],  # 2: turn left
        [0, 0, 1, 0, 0, 0],  # 3: turn right
        [0, 0, 0, 1, 0, 0],  # 4: strafe
        [0, 0, 0, 0, 1, 0],  # 5: attack
        [0, 0, 0, 0, 0, 1],  # 6: use 
    ]
    return buttons, actions


# -----------------------------
# Worker protocol
# -----------------------------
# Manager -> Worker messages:
# ("reset", seed:int|None)
# ("step", action_index:int)
# ("close",)
#
# Worker -> Manager messages:
# ("reset_ok", obs:np.ndarray, info:dict)
# ("step_ok", obs:np.ndarray, reward:float, done:bool, info:dict)
# ("error", msg:str, tb:str)


def _build_window_args(cfg: VZCoopConfig, p_index: int, game: vzd.DoomGame) -> str:
    if not cfg.tile_windows:
        return f"+name Player{p_index} +colorset 0"
    w = game.get_screen_width()
    h = game.get_screen_height()
    x = cfg.win_x + (p_index % 4) * w
    y = cfg.win_y + (p_index // 4) * h
    return f"+name Player{p_index} +colorset 0 +win_x {x} +win_y {y}"


def _init_game_instance(
    cfg: VZCoopConfig,
    is_host: bool,
    player_index: int,
    available_buttons: List[vzd.Button],
) -> Tuple[vzd.DoomGame, List[List[int]]]:
    game = vzd.DoomGame()
    game = vzd.DoomGame()
    config_path = cfg.config_path
    if isinstance(config_path, str) and config_path and not config_path.startswith("/"):
        config_path = os.path.join(vzd.scenarios_path, config_path)
    game.load_config(config_path)

    # Make sure we control button layout deterministically
    game.set_available_buttons(available_buttons)

    game.set_mode(cfg.mode)
    game.set_ticrate(cfg.ticrate)

    game.set_screen_resolution(cfg.screen_resolution)
    game.set_console_enabled(cfg.console_enabled)
    game.set_window_visible(cfg.window_visible)

    # Multiplayer args
    if is_host:
        # -host N waits for N total players
        args = f"-host {cfg.num_players} -netmode 0 "
        if cfg.timelimit_minutes is not None:
            args += f"+timelimit {cfg.timelimit_minutes} "
        # coop: do NOT set -deathmatch
        game.add_game_args(args)
    else:
        game.add_game_args(f"-join {cfg.host_ip}")

    # Name + optional window placement
    game.add_game_args(_build_window_args(cfg, player_index, game))

    if cfg.extra_game_args:
        game.add_game_args(cfg.extra_game_args)

    game.init()

    # Our discrete action mapping
    _, actions = build_default_buttons_and_actions()
    return game, actions


def _safe_get_obs(game: vzd.DoomGame) -> np.ndarray:
    state = game.get_state()
    if state is None or state.screen_buffer is None:
        # If episode finished, state can be None; return a dummy array.
        # Manager will mark done, so this is mainly to keep shapes consistent.
        h = game.get_screen_height()
        w = game.get_screen_width()
        return np.zeros((h, w, 3), dtype=np.uint8)
    # ViZDoom returns CHW or HWC depending on build; normalize to HWC uint8.
    buf = state.screen_buffer
    arr = np.asarray(buf)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        # likely CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return arr


def _worker_main(
    conn,
    cfg: VZCoopConfig,
    is_host: bool,
    player_index: int,
):
    # Make SIGINT not kill child mid-critical section; manager handles shutdown.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        buttons, _ = build_default_buttons_and_actions()
        game, actions = _init_game_instance(cfg, is_host, player_index, buttons)

        # Ensure episode exists
        # In multiplayer, do not call new_episode until all are ready; manager will send reset.
        while True:
            msg = conn.recv()
            cmd = msg[0]

            if cmd == "reset":
                # In multiplayer, all players must call new_episode().
                # Some single-player maps lack Player 2/3/4 starts; the
                # engine may sporadically fail on episode restart.  Retry
                # with a full game re-init to recover.
                max_retries = 3
                for _attempt in range(max_retries):
                    try:
                        game.new_episode()
                        break
                    except Exception:
                        try:
                            game.close()
                        except Exception:
                            pass
                        game, actions = _init_game_instance(
                            cfg, is_host, player_index, buttons
                        )
                else:
                    raise RuntimeError(
                        f"Player {player_index}: failed to start new episode "
                        f"after {max_retries} retries"
                    )
                obs = _safe_get_obs(game)
                info = {}
                conn.send(("reset_ok", obs, info))

            elif cmd == "step":
                if game.is_episode_finished():
                    done = True
                    reward = 0.0
                    obs = _safe_get_obs(game)
                    info = {"episode_time": int(game.get_episode_time())}
                    conn.send(("step_ok", obs, reward, done, info))
                    continue
                a_idx = int(msg[1])
                if a_idx < 0 or a_idx >= len(actions):
                    raise ValueError(f"Invalid action index {a_idx} (n={len(actions)})")

                # DEBUG: see that we actually reach step and with what
                print(f"[worker {player_index}] step a_idx={a_idx}", flush=True)

                try:
                    reward = float(game.make_action(actions[a_idx], cfg.skip))
                except vzd.vizdoom.ViZDoomErrorException:
                    # VizDoom native engine crashed (signal 11 / segfault).
                    # Re-init the game instance and signal episode-done so
                    # the manager resets all workers on the next step.
                    print(
                        f"[worker {player_index}] ViZDoom crashed during "
                        f"make_action — reinitialising game instance",
                        flush=True,
                    )
                    try:
                        game.close()
                    except Exception:
                        pass
                    game, actions = _init_game_instance(
                        cfg, is_host, player_index, buttons
                    )
                    obs = _safe_get_obs(game)
                    conn.send(("step_ok", obs, 0.0, True, {"vizdoom_crash": True}))
                    continue

                # DEBUG: see if make_action returns
                print(f"[worker {player_index}] make_action done, reward={reward}", flush=True)

                # if game.is_player_dead():
                #     game.respawn_player()

                done = bool(game.is_episode_finished())
                obs = _safe_get_obs(game)
                info = {}
                conn.send(("step_ok", obs, reward, done, info))

            elif cmd == "close":
                try:
                    game.close()
                finally:
                    conn.close()
                break

            else:
                raise ValueError(f"Unknown command: {cmd}")

    except Exception as e:
        tb = traceback.format_exc()
        try:
            conn.send(("error", str(e), tb))
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


# -----------------------------
# PettingZoo ParallelEnv wrapper
# -----------------------------
class VizDoomCoopParallelEnv(ParallelEnv):
    metadata = {"name": "vizdoom_coop_parallel_v0", "render_modes": ["human"]}

    def __init__(self, cfg: VZCoopConfig, frame_stack: int = 1):
        assert cfg.num_players >= 1
        self.cfg = cfg

        # How many past observations to stack along a new time dimension.
        # T=1 reproduces the original single-frame behaviour.
        self._frame_stack = int(frame_stack)
        assert self._frame_stack >= 1, "frame_stack must be >= 1"

        self._seconds_per_step = float(self.cfg.skip) / float(self.cfg.ticrate)

        self.possible_agents = [f"player_{i}" for i in range(cfg.num_players)]
        self.agents = self.possible_agents[:]

        # Action mapping
        self._buttons, self._actions = build_default_buttons_and_actions()
        self._n_actions = len(self._actions)

        # We need obs shape. Use resolution mapping:
        # ViZDoom 320x240 means (240,320,3)
        # We'll instantiate a tiny dummy DoomGame to read width/height reliably.
        # tmp = vzd.DoomGame()
        # tmp.load_config(cfg.config_path)
        # tmp.set_screen_resolution(cfg.screen_resolution)
        # tmp.set_window_visible(False)
        # tmp.init()
        # h, w = tmp.get_screen_height(), tmp.get_screen_width()
        # tmp.close()
        _res_to_hw = {
            vzd.ScreenResolution.RES_160X120: (120, 160),
            vzd.ScreenResolution.RES_320X240: (240, 320),
            vzd.ScreenResolution.RES_640X480: (480, 640),
            # add more if you ever use them
        }

        h, w = _res_to_hw[self.cfg.screen_resolution]

        # Base single-frame shape (H, W, C) for raw observation.
        self._frame_shape_hwc = (h, w, 3)

        # Stacked video shape: (T, C, H, W) for encoder compatibility.
        self._video_shape_tchw = (self._frame_stack, 3, h, w)

        self.observation_spaces = {
            a: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=255, shape=self._frame_shape_hwc, dtype=np.uint8
                    ),
                    "video": spaces.Box(
                        low=0, high=255, shape=self._video_shape_tchw, dtype=np.uint8
                    ),
                    "obs_start_time": spaces.Box(
                        low=0.0, high=np.finfo(np.float32).max, shape=(), dtype=np.float32
                    ),
                    "obs_end_time": spaces.Box(
                        low=0.0, high=np.finfo(np.float32).max, shape=(), dtype=np.float32
                    ),
                }
            )
            for a in self.possible_agents
        }
        self.action_spaces = {
            a: spaces.Discrete(self._n_actions) for a in self.possible_agents
        }

        # Worker infra
        self._parent_conns: Dict[str, Any] = {}
        self._procs: List[Process] = []
        self._started = False

        # For truncation control (optional)
        self._step_count = 0
        self._max_steps: Optional[int] = None  # set externally if you want

        # Per-agent observation buffers for frame stacking.
        # Each entry is a deque of at most self._frame_stack single-frame observations.
        self._obs_buffers: Dict[str, deque] = {}

    def _init_obs_buffers(self, first_obs: Dict[str, np.ndarray]) -> None:
        """
        Initialise per-agent deques on reset. We fill them either with T copies
        of the first frame (standard in RL) or just the first frame when T=1.
        """
        self._obs_buffers = {}
        for a, o in first_obs.items():
            buf = deque(maxlen=self._frame_stack)
            # Replicate the first observation so that the initial stacked tensor
            # has meaningful content at all time indices.
            for _ in range(self._frame_stack):
                buf.append(o)
            self._obs_buffers[a] = buf

    def _get_stacked_obs(self, agent: str) -> np.ndarray:
        """
        Return the stacked video for a single agent.
        Shape:
            - (T, C, H, W)
        """
        buf = self._obs_buffers.get(agent)
        if buf is None or len(buf) == 0:
            # Should not normally happen, but fall back to zeros with correct shape.
            return np.zeros(self._video_shape_tchw, dtype=np.uint8)

        # Convert each HWC frame to CHW, then stack along time: (T, C, H, W).
        frames_chw = [np.transpose(frame, (2, 0, 1)) for frame in buf]
        return np.stack(frames_chw, axis=0)

    def _get_time_window(self, current_step_index: int) -> Tuple[float, float]:
        end_time = current_step_index * self._seconds_per_step
        start_time = max(0.0, end_time - (self._frame_stack - 1) * self._seconds_per_step)
        return start_time, end_time

    def _start_workers(self):
        if self._started:
            return

        # Start joiners first, then host (matches official pattern and your demo)
        # We'll create one Pipe per agent.
        # IMPORTANT: host is player_0
        conns: Dict[str, Any] = {}
        procs: List[Process] = []

        # Joiners
        for i in range(1, self.cfg.num_players):
            agent = f"player_{i}"
            parent_conn, child_conn = Pipe(duplex=True)
            p = Process(
                target=_worker_main,
                args=(child_conn, self.cfg, False, i),
                daemon=True,
            )
            p.start()
            conns[agent] = parent_conn
            procs.append(p)

        # Host last
        agent0 = "player_0"
        parent_conn, child_conn = Pipe(duplex=True)
        p0 = Process(
            target=_worker_main,
            args=(child_conn, self.cfg, True, 0),
            daemon=True,
        )
        p0.start()
        conns[agent0] = parent_conn
        procs.append(p0)

        self._parent_conns = conns
        self._procs = procs
        self._started = True

    def _recv_ok(self, agent: str):
        msg = self._parent_conns[agent].recv()
        if msg[0] == "error":
            _, m, tb = msg
            raise RuntimeError(f"[{agent}] worker error: {m}\n{tb}")
        return msg

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self._start_workers()
        self.agents = self.possible_agents[:]
        self._step_count = 0

        # Send reset to all workers
        for a in self.agents:
            self._parent_conns[a].send(("reset", seed))

        obs: Dict[str, np.ndarray] = {}
        infos: Dict[str, dict] = {}
        for a in self.agents:
            msg = self._recv_ok(a)
            assert msg[0] == "reset_ok"
            _, o, info = msg
            obs[a] = o
            infos[a] = info

        # (Re)initialise frame-stacking buffers and return both raw + video observations.
        self._init_obs_buffers(obs)
        stacked_obs = {
            a: {
                "observation": obs[a],
                "video": self._get_stacked_obs(a),
                "obs_start_time": np.float32(self._get_time_window(0)[0]),
                "obs_end_time": np.float32(self._get_time_window(0)[1]),
            }
            for a in self.agents
        }

        return stacked_obs, infos

    def step(self, actions: Dict[str, int]):
        if len(self.agents) == 0:
            raise RuntimeError("step() called after episode ended; call reset().")

        # DEBUG
        print("[manager] step called with actions:", actions, flush=True)

        for a in self.agents:
            if a not in actions:
                raise ValueError(f"Missing action for live agent {a}")

        for a in self.agents:
            print(f"[manager] sending step to {a}", flush=True)
            self._parent_conns[a].send(("step", int(actions[a])))

        obs = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        done_any = False
        for a in self.agents:
            print(f"[manager] waiting for reply from {a}", flush=True)
            msg = self._recv_ok(a)
            print(f"[manager] got reply from {a}: {msg[0]}", flush=True)
            assert msg[0] == "step_ok"
            _, o, r, done, info = msg
            # Update frame buffer with the newest single-frame observation.
            if a not in self._obs_buffers:
                # Shouldn't happen in normal flow, but create a fresh buffer if needed.
                buf = deque(maxlen=self._frame_stack)
                for _ in range(self._frame_stack - 1):
                    buf.append(o)
                self._obs_buffers[a] = buf
            self._obs_buffers[a].append(o)

            obs[a] = {
                "observation": o,
                "video": self._get_stacked_obs(a),
                "obs_start_time": np.float32(self._get_time_window(self._step_count + 1)[0]),
                "obs_end_time": np.float32(self._get_time_window(self._step_count + 1)[1]),
            }
            rewards[a] = float(r)
            terminations[a] = bool(done)
            infos[a] = info
            truncations[a] = False
            done_any = done_any or bool(done)

        self._step_count += 1
        if self._max_steps is not None and self._step_count >= self._max_steps:
            # Force truncate
            for a in self.agents:
                truncations[a] = True
                terminations[a] = False
            done_any = True

        # Multiplayer often ends for everyone together; be robust and end if ANY says done.
        if done_any:
            # Mark all as ended so trainers don't hang waiting
            for a in self.agents:
                # If we truncated, keep as set above; else terminate all
                if not truncations[a]:
                    terminations[a] = True
            self.agents = []

            # Drop buffers once the episode is over to free memory.
            self._obs_buffers.clear()

        return obs, rewards, terminations, truncations, infos

    def render(self):
        # In this multiprocessing design, each worker can display its own window if cfg.window_visible=True.
        # If you want a single combined render output, you'd need to request frames from one agent and show them here.
        return None

    def close(self):
        if not self._started:
            return
        # Ask workers to close
        for a, c in self._parent_conns.items():
            try:
                c.send(("close",))
            except Exception:
                pass
        # Join processes
        for p in self._procs:
            try:
                p.join(timeout=2.0)
            except Exception:
                pass
        self._parent_conns.clear()
        self._procs.clear()
        self._started = False


# -----------------------------
# BenchMARL Task wrapper
# -----------------------------
# from scripts.doom_wrappers.DoomPettingzoo import DoomPettingZooTask

# task = DoomPettingZooTask.DOOM_PZ.get_from_yaml(
#     path="/home/sohans/X_EGO/x-ego/configs/task/doom_pettingzoo.yaml"
# )
class DoomPettingZooTask(Task):
    DOOM_PZ = None

    @staticmethod
    def associated_class():
        return DoomPettingZooEnv


class DoomPettingZooEnv(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        frame_stack = config.pop("frame_stack", 1)
        wrapper_kwargs = config.pop("wrapper_kwargs", {})
        max_steps = config.pop("max_steps", None)

        if isinstance(config.get("mode"), str):
            config["mode"] = getattr(vzd.Mode, config["mode"])
        if isinstance(config.get("screen_resolution"), str):
            config["screen_resolution"] = getattr(
                vzd.ScreenResolution, config["screen_resolution"]
            )

        cfg = VZCoopConfig(**config)

        # Optional: allow shifting the base network port via environment variable
        # so that multiple training processes can run in parallel without port
        # collisions (e.g. in tmux panes).
        port_offset_str = os.environ.get("DOOM_PORT_OFFSET")
        if port_offset_str is not None:
            try:
                port_offset = int(port_offset_str)
                cfg.base_port = int(cfg.base_port) + port_offset
            except ValueError:
                # If the env var is malformed, just ignore it and use the default.
                pass

        def make_env() -> EnvBase:
            env_idx = next(_ENV_PORT_COUNTER)
            cfg_local = copy.deepcopy(cfg)
            if cfg_local.base_port is not None:
                port = int(cfg_local.base_port) + env_idx * int(cfg_local.port_stride)
                extra = cfg_local.extra_game_args or ""
                if " -port " not in f" {extra} ":
                    cfg_local.extra_game_args = f"{extra} -port {port}".strip()
            env = VizDoomCoopParallelEnv(cfg_local, frame_stack=frame_stack)
            if max_steps is not None:
                env._max_steps = int(max_steps)
            return PettingZooWrapper(
                env,
                categorical_actions=True,
                device=device,
                seed=seed,
                return_state=False,
                **wrapper_kwargs,
            )

        return make_env

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def has_state(self) -> bool:
        return False

    def has_render(self, env: EnvBase) -> bool:
        return False

    def max_steps(self, env: EnvBase) -> int:
        if hasattr(env, "max_steps") and env.max_steps is not None:
            return int(env.max_steps)
        # TorchRL rollout expects an int; use a safe default if not configured.
        return 1000

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        if hasattr(env, "group_map"):
            return env.group_map
        if hasattr(env, "possible_agents"):
            return {"agents": list(env.possible_agents)}
        return {"agents": []}

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        return None

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key not in {"observation", "video", "obs_start_time", "obs_end_time"}:
                    del group_obs_spec[key]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        observation_spec = env.observation_spec.clone()
        if "info" not in observation_spec.keys(True, True):
            return None
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "info":
                    del group_obs_spec[key]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        return observation_spec

    def action_spec(self, env: EnvBase) -> CompositeSpec:
        return env.full_action_spec

    @staticmethod
    def env_name() -> str:
        return "doom_pettingzoo"


# -----------------------------
# Quick smoke test
# -----------------------------
if __name__ == "__main__":
    cfg = VZCoopConfig(
        config_path=os.path.join(vzd.scenarios_path, "deadly_corridor.cfg"),
        num_players=4,
        window_visible=False,      # set False for training
        tile_windows=True,
        timelimit_minutes=1,
        extra_game_args="",       # you can add +sv_nomonsters 0 etc
    )

    env = VizDoomCoopParallelEnv(cfg)
    obs, infos = env.reset()

    print("Agents:", env.possible_agents)
    print("Obs shapes:", {a: o.shape for a, o in obs.items()})

    # Random rollout
    rng = np.random.default_rng(0)
    for t in range(200):
        print("t =", t, "agents:", env.agents)
        if len(env.agents) == 0:
            break
        acts = {a: int(rng.integers(0, env.action_spaces[a].n)) for a in env.agents}
        obs, rew, term, trunc, info = env.step(acts)
        print("  term:", term, "trunc:", trunc)
        if any(term.values()) or any(trunc.values()):
            break
    print("loop done, calling env.close()")
    env.close()
    print("Done.")
