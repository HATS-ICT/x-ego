#!/usr/bin/env python3
"""
Find the largest physical and virtual contrastive batch settings per encoder.

This script uses dummy video tensors but runs the real ContrastiveModel encoder,
projector, sigmoid pair loss, embedding-cache replay, optimizer step, and
scheduler-free training path. It first probes physical data.batch_size with
training.contrastive_accumulate_batches=1, then uses the largest passing
physical batch size to probe contrastive_accumulate_batches.

For contrastive data loading, data.batch_size means videos per physical
microbatch. Dummy agent_counts are only used to preserve team boundaries for the
alignment matrix, and always sum to data.batch_size.

Known manual physical video batch sizes skip the physical-batch probe and are
used directly for the contrastive_accumulate_batches search.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from omegaconf import OmegaConf

from src.models.contrastive_model import ContrastiveModel
from src.utils.config_utils import load_cfg


MODEL_ALIASES = {
    "resnet": "resnet50",
}

# DEFAULT_MODELS = ("clip", "vjepa2", "siglip2", "dinov3", "resnet50")
DEFAULT_MODELS = ("vjepa2", "dinov3", "resnet50")
MANUAL_PHYSICAL_VIDEO_BATCH_SIZES = {
    "dinov3": 32,
    "resnet50": 32,
}
VJEPA2_IMAGE_SIZE = 256
DEFAULT_IMAGE_SIZE = 224
DEFAULT_PEAK_MEMORY_LIMIT_GB = 40.0
DEFAULT_MAX_VIRTUAL_VIDEO_BATCH_SIZE = 4096
ATTEMPT_RESULT_PREFIX = "X_EGO_ATTEMPT_RESULT="


@dataclass(frozen=True)
class ProbeResult:
    model_type: str
    max_physical_batch_size: int
    first_physical_batch_fail_at: int | None
    max_accumulate_batches: int
    first_oom_at: int | None
    agents_per_sample: int
    videos_per_microbatch: int
    max_virtual_videos: int
    physical_peak_memory_gb: float | None
    peak_memory_gb: float | None
    error: str | None = None

    def to_dict(self) -> dict:
        matrix_size = self.max_virtual_videos
        return {
            "model_type": self.model_type,
            "optimal": {
                "data.batch_size": self.max_physical_batch_size,
                "training.accumulate_grad_batches": 1,
                "training.contrastive_accumulate_batches": self.max_accumulate_batches,
            },
            "physical_batch": {
                "max_videos": self.max_physical_batch_size,
                "first_fail_at": self.first_physical_batch_fail_at,
                "max_agents_per_team": self.agents_per_sample,
                "videos_per_microbatch": self.videos_per_microbatch,
                "peak_memory_gb": self.physical_peak_memory_gb,
            },
            "virtual_batch": {
                "max_accumulate_batches": self.max_accumulate_batches,
                "first_fail_at": self.first_oom_at,
                "max_videos": self.max_virtual_videos,
                "videos_per_microbatch": self.videos_per_microbatch,
                "peak_memory_gb": self.peak_memory_gb,
            },
            "matrix": {
                "max_agents_per_team": self.agents_per_sample,
                "total_videos": matrix_size,
                "shape": [matrix_size, matrix_size],
                "num_entries": matrix_size * matrix_size,
                "formula": (
                    "data.batch_size * training.contrastive_accumulate_batches"
                ),
            },
            "status": "ok" if self.error is None else "error",
            "error": self.error,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe max contrastive_accumulate_batches before CUDA OOM."
    )
    parser.add_argument(
        "--config",
        default="configs/train/contrastive.yaml",
        help="Base contrastive YAML config to use.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model aliases to probe. Use resnet or resnet50 for torchvision ResNet-50.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Fixed physical video batch size. When omitted, known manual "
            "per-model values are used before falling back to physical search."
        ),
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=128,
        help="Hard cap for the physical video batch-size search.",
    )
    parser.add_argument(
        "--agents-per-sample",
        type=int,
        default=5,
        help="Maximum dummy agents per contrastive team when forming agent_counts.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Smallest contrastive_accumulate_batches value to test.",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=64,
        help="Hard cap for contrastive_accumulate_batches search.",
    )
    parser.add_argument(
        "--max-virtual-video-batch-size",
        type=int,
        default=DEFAULT_MAX_VIRTUAL_VIDEO_BATCH_SIZE,
        help="Hard cap for data.batch_size * contrastive_accumulate_batches.",
    )
    parser.add_argument(
        "--peak-memory-limit-gb",
        type=float,
        default=DEFAULT_PEAK_MEMORY_LIMIT_GB,
        help="Treat settings above this CUDA peak memory as failed.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Frames per dummy video. Defaults to fixed_duration_seconds * target_fps.",
    )
    parser.add_argument(
        "--finetune-last-k-layers",
        type=int,
        default=None,
        help="Override model.encoder.finetune_last_k_layers.",
    )
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable CUDA bf16 autocast during probe.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=10.0,
        help=(
            "Maximum seconds allowed per physical microbatch in a virtual-batch "
            "attempt. Total attempt timeout is this value times "
            "contrastive_accumulate_batches."
        ),
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=180.0,
        help="Extra seconds allowed for Python startup, imports, model loading, and GPU placement.",
    )
    parser.add_argument(
        "--gpu-name",
        default=None,
        help="GPU name for the output JSON filename, e.g. 4090, a40, a100.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifact",
        help="Directory for the final JSON result file.",
    )
    parser.add_argument("--_attempt-model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_attempt-accumulate", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def canonical_model_type(model_type: str) -> str:
    return MODEL_ALIASES.get(model_type, model_type)


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "unknown-gpu"


def prompt_gpu_name(args: argparse.Namespace) -> str:
    if args._attempt_model is not None:
        return args.gpu_name or "attempt"
    if args.gpu_name:
        return args.gpu_name
    return input("GPU name for this run (e.g. 4090, a40, a100): ").strip()


def manual_physical_video_batch_size(args: argparse.Namespace, model_type: str) -> int | None:
    if args.batch_size is not None:
        return int(args.batch_size)
    return MANUAL_PHYSICAL_VIDEO_BATCH_SIZES.get(model_type)


def cuda_memory_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1024**3


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        isinstance(exc, torch.cuda.OutOfMemoryError)
        or "out of memory" in text
        or "cuda error: out of memory" in text
        or "cublas_status_alloc_failed" in text
    )


def dummy_image_size(model_type: str) -> int:
    return VJEPA2_IMAGE_SIZE if model_type == "vjepa2" else DEFAULT_IMAGE_SIZE


def build_cfg(args: argparse.Namespace, model_type: str, accumulate_batches: int):
    cfg = load_cfg(args.config)
    cfg.model.encoder.model_type = model_type
    cfg.training.accumulate_grad_batches = 1
    cfg.training.contrastive_accumulate_batches = accumulate_batches
    cfg.optimization.scheduler = None

    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.finetune_last_k_layers is not None:
        cfg.model.encoder.finetune_last_k_layers = args.finetune_last_k_layers

    # The real train setup adds this path before model construction. It is only
    # stored on the model here, so a dummy value is enough for this probe.
    if "path" not in cfg:
        cfg = OmegaConf.merge(cfg, {"path": {"exp": "."}})
    elif "exp" not in cfg.path:
        cfg.path.exp = "."

    return cfg


def make_dummy_batch(
    *,
    batch_size: int,
    agents_per_sample: int,
    frames: int,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    agent_counts = make_dummy_agent_counts(
        total_videos=batch_size,
        max_agents_per_team=agents_per_sample,
        device=device,
    )
    video = torch.randn(
        batch_size,
        frames,
        3,
        image_size,
        image_size,
        device=device,
        dtype=dtype,
    )
    return {
        "video": video,
        "agent_counts": agent_counts,
    }


def make_dummy_agent_counts(
    *,
    total_videos: int,
    max_agents_per_team: int,
    device: torch.device,
) -> torch.Tensor:
    """Create team chunks whose total videos match the physical batch size."""
    if total_videos <= 0:
        raise ValueError(f"total_videos must be positive, got {total_videos}")
    if max_agents_per_team <= 0:
        raise ValueError(f"max_agents_per_team must be positive, got {max_agents_per_team}")

    counts = []
    remaining = int(total_videos)
    while remaining > 0:
        take = min(int(max_agents_per_team), remaining)
        counts.append(take)
        remaining -= take
    return torch.tensor(counts, dtype=torch.long, device=device)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.detach().to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def run_one_virtual_step(
    *,
    model: ContrastiveModel,
    optimizer: torch.optim.Optimizer,
    cfg,
    accumulate_batches: int,
    agents_per_sample: int,
    frames: int,
    use_bf16: bool,
) -> None:
    device = next(model.parameters()).device
    image_size = dummy_image_size(cfg.model.encoder.model_type)
    input_dtype = torch.float32
    autocast_enabled = use_bf16 and device.type == "cuda"

    def autocast_context():
        if autocast_enabled:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    cache = []

    for _ in range(accumulate_batches):
        batch = make_dummy_batch(
            batch_size=int(cfg.data.batch_size),
            agents_per_sample=agents_per_sample,
            frames=frames,
            image_size=image_size,
            device=device,
            dtype=input_dtype,
        )
        with torch.no_grad():
            with autocast_context():
                projected = model._compute_projected_embeddings(batch).detach()
        cache.append(
            {
                "batch": move_batch_to_device(batch, torch.device("cpu")),
                "projected": projected,
                "agent_counts": batch["agent_counts"].detach().clone(),
            }
        )

    cached_projected = torch.cat(
        [entry["projected"] for entry in cache],
        dim=0,
    ).detach().requires_grad_(True)
    agent_counts = torch.cat([entry["agent_counts"] for entry in cache], dim=0)
    labels = model.create_alignment_matrix(agent_counts, cached_projected.device)

    with autocast_context():
        contrastive_loss, _ = model.compute_contrastive_loss(cached_projected, labels)
        loss = model.contrastive_loss_weight * contrastive_loss
    loss.backward()

    projected_grads = cached_projected.grad.detach()
    start = 0
    for entry in cache:
        end = start + entry["projected"].shape[0]
        grad_chunk = projected_grads[start:end]
        start = end

        with autocast_context():
            replay_batch = move_batch_to_device(entry["batch"], device)
            projected = model._compute_projected_embeddings(replay_batch)
            replay_loss = torch.sum(projected * grad_chunk)
        replay_loss.backward()

    if cfg.training.gradient_clip_val is not None and cfg.training.gradient_clip_val > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_val)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Make sure async CUDA allocation failures surface inside the tested setting.
    if device.type == "cuda":
        torch.cuda.synchronize()


def can_run_setting(
    args: argparse.Namespace,
    model_type: str,
    batch_size: int,
    accumulate_batches: int,
) -> tuple[bool, float | None, str | None]:
    clear_memory()
    cfg = build_cfg(args, model_type, accumulate_batches)
    cfg.data.batch_size = batch_size
    frames = args.frames or int(cfg.data.fixed_duration_seconds * cfg.data.target_fps)
    use_bf16 = not args.no_bf16
    model = None
    optimizer = None
    timeout_timer = None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ContrastiveModel(cfg).to(device)
        optimizer = model.configure_optimizers()["optimizer"]

        def exit_on_attempt_timeout() -> None:
            peak = cuda_memory_gb()
            print(
                ATTEMPT_RESULT_PREFIX + json.dumps(
                    {"ok": False, "peak": peak, "error": "TIMEOUT"},
                    sort_keys=True,
                ),
                flush=True,
            )
            os._exit(124)

        attempt_timeout_seconds = args.timeout_seconds * accumulate_batches
        if args.timeout_seconds is not None and args.timeout_seconds > 0:
            timeout_timer = threading.Timer(attempt_timeout_seconds, exit_on_attempt_timeout)
            timeout_timer.daemon = True
            timeout_timer.start()

        run_one_virtual_step(
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            accumulate_batches=accumulate_batches,
            agents_per_sample=args.agents_per_sample,
            frames=frames,
            use_bf16=use_bf16,
        )
        if timeout_timer is not None:
            timeout_timer.cancel()
        peak = cuda_memory_gb()
        del optimizer, model
        clear_memory()
        return True, peak, None
    except BaseException as exc:
        if timeout_timer is not None:
            timeout_timer.cancel()
        peak = cuda_memory_gb()
        oom = is_oom_error(exc)
        error = "OOM" if oom else f"{type(exc).__name__}: {exc}"
        del exc, optimizer, model
        clear_memory()
        return False, peak, error


def attempt_failure_is_boundary(error: str | None) -> bool:
    return error in {"OOM", "TIMEOUT", "PEAK_MEMORY_LIMIT"}


def child_attempt_command(
    args: argparse.Namespace,
    model_type: str,
    batch_size: int,
    accumulate_batches: int,
) -> list[str]:
    command = [
        sys.executable,
        __file__,
        "--config",
        args.config,
        "--batch-size",
        str(batch_size),
        "--agents-per-sample",
        str(args.agents_per_sample),
        "--start",
        str(args.start),
        "--max",
        str(args.max),
        "--_attempt-model",
        model_type,
        "--_attempt-accumulate",
        str(accumulate_batches),
    ]
    if args.frames is not None:
        command.extend(["--frames", str(args.frames)])
    if args.finetune_last_k_layers is not None:
        command.extend(["--finetune-last-k-layers", str(args.finetune_last_k_layers)])
    if args.no_bf16:
        command.append("--no-bf16")
    command.extend(["--timeout-seconds", str(args.timeout_seconds)])
    command.extend(["--max-virtual-video-batch-size", str(args.max_virtual_video_batch_size)])
    command.extend(["--peak-memory-limit-gb", str(args.peak_memory_limit_gb)])
    return command


def run_setting_with_timeout(
    args: argparse.Namespace,
    model_type: str,
    batch_size: int,
    accumulate_batches: int,
) -> tuple[bool, float | None, str | None]:
    command = child_attempt_command(args, model_type, batch_size, accumulate_batches)
    try:
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            timeout=args.startup_timeout_seconds + args.timeout_seconds * accumulate_batches,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, None, "STARTUP_TIMEOUT"

    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    result_line = None
    for line in output.splitlines():
        if line.startswith(ATTEMPT_RESULT_PREFIX):
            result_line = line[len(ATTEMPT_RESULT_PREFIX):]

    if result_line is None:
        lowered_output = output.lower()
        if "out of memory" in lowered_output or "cublas_status_alloc_failed" in lowered_output:
            return False, None, "OOM"
        return False, None, f"NO_RESULT: {output.strip()[-500:]}"

    try:
        payload = json.loads(result_line)
    except json.JSONDecodeError as exc:
        return False, None, f"BAD_RESULT: {exc}"

    if completed.returncode != 0 and payload.get("error") is None:
        return False, payload.get("peak"), f"EXIT_{completed.returncode}"

    return bool(payload["ok"]), payload.get("peak"), payload.get("error")


def next_probe_value(current: int, start: int) -> int:
    return max(current * 2, start + 1)


def search_largest_passing_value(
    *,
    label: str,
    start: int,
    max_value: int,
    try_value,
    value_suffix=None,
    peak_memory_limit_gb: float | None = None,
) -> tuple[int, int | None, float | None, str | None]:
    low = 0
    low_peak = None
    high = None
    first_error = None
    probe = max(start, 1)

    while probe <= max_value:
        suffix = "" if value_suffix is None else value_suffix(probe)
        print(f"{label} trying {probe}{suffix}", flush=True)
        ok, peak, error = try_value(probe)
        if ok and peak_memory_limit_gb is not None and peak is not None and peak > peak_memory_limit_gb:
            ok = False
            error = "PEAK_MEMORY_LIMIT"
        peak_text = "n/a" if peak is None else f"{peak:.2f} GB"
        if ok:
            print(f"{label} OK at {probe}{suffix} peak={peak_text}", flush=True)
            low = probe
            low_peak = peak
            probe = next_probe_value(probe, start)
        else:
            print(f"{label} FAIL at {probe}{suffix} peak={peak_text} error={error}", flush=True)
            if not attempt_failure_is_boundary(error):
                return low, high, low_peak, error
            high = probe
            first_error = error
            break

    if high is None:
        if low >= max_value:
            return low, None, low_peak, None
        return low, None, low_peak, None if low > 0 else first_error

    while high - low > 1:
        mid = (low + high) // 2
        suffix = "" if value_suffix is None else value_suffix(mid)
        print(f"{label} binary trying {mid}{suffix}", flush=True)
        ok, peak, error = try_value(mid)
        if ok and peak_memory_limit_gb is not None and peak is not None and peak > peak_memory_limit_gb:
            ok = False
            error = "PEAK_MEMORY_LIMIT"
        peak_text = "n/a" if peak is None else f"{peak:.2f} GB"
        if ok:
            print(f"{label} OK at {mid}{suffix} peak={peak_text}", flush=True)
            low = mid
            low_peak = peak
        else:
            print(f"{label} FAIL at {mid}{suffix} peak={peak_text} error={error}", flush=True)
            if not attempt_failure_is_boundary(error):
                return low, high, low_peak, error
            high = mid
            first_error = error

    return low, high, low_peak, None if low > 0 else first_error


def find_max_for_model(args: argparse.Namespace, model_type: str) -> ProbeResult:
    model_type = canonical_model_type(model_type)
    manual_batch_size = manual_physical_video_batch_size(args, model_type)
    start_batch_size = manual_batch_size
    if start_batch_size is None:
        start_batch_size = int(load_cfg(args.config).data.batch_size)

    print(f"\n=== Probing {model_type} ===", flush=True)
    print(
        "start_physical_video_batch_size="
        f"{start_batch_size}, max_physical_video_batch_size={args.max_batch_size}, "
        f"max_agents_per_team={args.agents_per_sample}",
        flush=True,
    )

    if manual_batch_size is not None:
        physical_low = manual_batch_size
        physical_high = None
        physical_peak = None
        physical_error = None
        print(
            f"[{model_type}] using manual physical_video_batch_size={physical_low}; "
            "skipping physical batch probe",
            flush=True,
        )
    else:
        physical_max = min(args.max_batch_size, args.max_virtual_video_batch_size)
        physical_low, physical_high, physical_peak, physical_error = search_largest_passing_value(
            label=f"[{model_type}] physical_video_batch_size",
            start=start_batch_size,
            max_value=physical_max,
            try_value=lambda value: run_setting_with_timeout(
                args,
                model_type,
                batch_size=value,
                accumulate_batches=1,
            ),
            peak_memory_limit_gb=args.peak_memory_limit_gb,
        )

    if physical_low <= 0 or physical_error is not None:
        return ProbeResult(
            model_type=model_type,
            max_physical_batch_size=physical_low,
            first_physical_batch_fail_at=physical_high,
            max_accumulate_batches=0,
            first_oom_at=None,
            agents_per_sample=args.agents_per_sample,
            videos_per_microbatch=physical_low,
            max_virtual_videos=0,
            physical_peak_memory_gb=physical_peak,
            peak_memory_gb=None,
            error=physical_error,
        )

    print(
        f"[{model_type}] using physical_video_batch_size={physical_low} "
        "for contrastive_accumulate_batches search",
        flush=True,
    )

    max_accumulate_by_virtual_videos = args.max_virtual_video_batch_size // physical_low
    accum_max = min(args.max, max_accumulate_by_virtual_videos)
    if accum_max < args.start:
        return ProbeResult(
            model_type=model_type,
            max_physical_batch_size=physical_low,
            first_physical_batch_fail_at=physical_high,
            max_accumulate_batches=0,
            first_oom_at=None,
            agents_per_sample=args.agents_per_sample,
            videos_per_microbatch=physical_low,
            max_virtual_videos=0,
            physical_peak_memory_gb=physical_peak,
            peak_memory_gb=None,
            error="VIRTUAL_VIDEO_BATCH_LIMIT",
        )

    accum_low, accum_high, accum_peak, accum_error = search_largest_passing_value(
        label=f"[{model_type}] contrastive_accumulate_batches",
        start=args.start,
        max_value=accum_max,
        try_value=lambda value: run_setting_with_timeout(
            args,
            model_type,
            batch_size=physical_low,
            accumulate_batches=value,
        ),
        value_suffix=lambda value: (
            f" virtual_video_batch_size={physical_low * value}"
        ),
        peak_memory_limit_gb=args.peak_memory_limit_gb,
    )

    return ProbeResult(
        model_type=model_type,
        max_physical_batch_size=physical_low,
        first_physical_batch_fail_at=physical_high,
        max_accumulate_batches=accum_low,
        first_oom_at=accum_high,
        agents_per_sample=args.agents_per_sample,
        videos_per_microbatch=physical_low,
        max_virtual_videos=physical_low * accum_low,
        physical_peak_memory_gb=physical_peak,
        peak_memory_gb=accum_peak,
        error=accum_error,
    )


def print_summary(results: Iterable[ProbeResult]) -> None:
    print("\n=== Summary ===", flush=True)
    for result in results:
        peak = "n/a" if result.peak_memory_gb is None else f"{result.peak_memory_gb:.2f} GB"
        physical_peak = (
            "n/a"
            if result.physical_peak_memory_gb is None
            else f"{result.physical_peak_memory_gb:.2f} GB"
        )
        physical_fail = (
            "not reached"
            if result.first_physical_batch_fail_at is None
            else str(result.first_physical_batch_fail_at)
        )
        first_oom = "not reached" if result.first_oom_at is None else str(result.first_oom_at)
        status = "OK" if result.error is None else f"ERROR: {result.error}"
        print(
            f"{result.model_type}: max_physical_video_batch_size="
            f"{result.max_physical_batch_size}, first_physical_fail_at={physical_fail}, "
            f"physical_peak={physical_peak}, max_contrastive_accumulate_batches="
            f"{result.max_accumulate_batches}, first_accumulate_fail_at={first_oom}, "
            f"videos_per_microbatch={result.videos_per_microbatch}, "
            f"max_virtual_videos={result.max_virtual_videos}, peak_at_max={peak}, {status}",
            flush=True,
        )


def write_results_json(
    *,
    args: argparse.Namespace,
    gpu_name: str,
    results: list[ProbeResult],
    elapsed_seconds: float,
) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    gpu_slug = slugify(gpu_name)
    output_path = output_dir / f"contra_accumulate_probe_{gpu_slug}_{timestamp}.json"

    payload = {
        "gpu_name": gpu_name,
        "gpu_slug": gpu_slug,
        "created_at": timestamp,
        "elapsed_seconds": elapsed_seconds,
        "config": {
            "config_path": args.config,
            "models": [canonical_model_type(model) for model in args.models],
            "start_batch_size": args.batch_size if args.batch_size is not None else int(load_cfg(args.config).data.batch_size),
            "manual_physical_video_batch_sizes": MANUAL_PHYSICAL_VIDEO_BATCH_SIZES,
            "max_batch_size": args.max_batch_size,
            "max_virtual_video_batch_size": args.max_virtual_video_batch_size,
            "peak_memory_limit_gb": args.peak_memory_limit_gb,
            "agents_per_sample": args.agents_per_sample,
            "start_accumulate_batches": args.start,
            "max_accumulate_batches": args.max,
            "frames": args.frames,
            "finetune_last_k_layers": args.finetune_last_k_layers,
            "bf16": not args.no_bf16,
            "timeout_seconds_per_microbatch": args.timeout_seconds,
            "startup_timeout_seconds": args.startup_timeout_seconds,
        },
        "results": [result.to_dict() for result in results],
    }

    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    if args._attempt_model is not None:
        ok, peak, error = can_run_setting(
            args,
            canonical_model_type(args._attempt_model),
            int(args.batch_size),
            int(args._attempt_accumulate),
        )
        print(
            ATTEMPT_RESULT_PREFIX + json.dumps(
                {"ok": ok, "peak": peak, "error": error},
                sort_keys=True,
            ),
            flush=True,
        )
        raise SystemExit(0 if ok else 1)

    gpu_name = prompt_gpu_name(args)

    start_batch_size = args.batch_size
    if start_batch_size is None:
        start_batch_size = int(load_cfg(args.config).data.batch_size)
    if args.max_batch_size < start_batch_size:
        raise ValueError(
            "--max-batch-size must be greater than or equal to the starting "
            f"batch size ({start_batch_size})"
        )
    if args.max_virtual_video_batch_size < start_batch_size:
        raise ValueError(
            "--max-virtual-video-batch-size must be greater than or equal to "
            f"the starting batch size ({start_batch_size})"
        )

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available; this will not find GPU OOM limits.", flush=True)

    start_time = time.time()
    results = []
    for model_type in args.models:
        results.append(find_max_for_model(args, model_type))
    elapsed_seconds = time.time() - start_time

    print_summary(results)
    output_path = write_results_json(
        args=args,
        gpu_name=gpu_name,
        results=results,
        elapsed_seconds=elapsed_seconds,
    )
    print(f"\nWrote JSON results: {output_path}", flush=True)
    print(f"Total probe time: {elapsed_seconds / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
