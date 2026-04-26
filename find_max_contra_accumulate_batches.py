#!/usr/bin/env python3
"""
Find the largest contrastive virtual-batch accumulation setting per encoder.

This script uses dummy video tensors but runs the real ContrastiveModel encoder,
projector, sigmoid pair loss, embedding-cache replay, optimizer step, and
scheduler-free training path. It increases training.contrastive_accumulate_batches
until CUDA runs out of memory, then prints the largest passing value.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
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

DEFAULT_MODELS = ("clip", "vjepa2", "siglip2", "dinov3", "resnet50")
VJEPA2_IMAGE_SIZE = 256
DEFAULT_IMAGE_SIZE = 224
ATTEMPT_RESULT_PREFIX = "X_EGO_ATTEMPT_RESULT="


@dataclass(frozen=True)
class ProbeResult:
    model_type: str
    max_accumulate_batches: int
    first_oom_at: int | None
    batch_size: int
    agents_per_sample: int
    total_agents_per_microbatch: int
    peak_memory_gb: float | None
    error: str | None = None


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
        default=4,
        help="Physical dataloader batch size. Defaults to config data.batch_size.",
    )
    parser.add_argument(
        "--agents-per-sample",
        type=int,
        default=5,
        help="Dummy alive agents per contrastive sample.",
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
        default=128,
        help="Hard cap for contrastive_accumulate_batches search.",
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
    parser.add_argument("--_attempt-model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_attempt-accumulate", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def canonical_model_type(model_type: str) -> str:
    return MODEL_ALIASES.get(model_type, model_type)


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
    total_agents = batch_size * agents_per_sample
    video = torch.randn(
        total_agents,
        frames,
        3,
        image_size,
        image_size,
        device=device,
        dtype=dtype,
    )
    return {
        "video": video,
        "agent_counts": torch.full(
            (batch_size,),
            agents_per_sample,
            dtype=torch.long,
            device=device,
        ),
    }


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
                "batch": batch,
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
            projected = model._compute_projected_embeddings(entry["batch"])
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
    accumulate_batches: int,
) -> tuple[bool, float | None, str | None]:
    clear_memory()
    cfg = build_cfg(args, model_type, accumulate_batches)
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
    return error in {"OOM", "TIMEOUT"}


def child_attempt_command(
    args: argparse.Namespace,
    model_type: str,
    accumulate_batches: int,
) -> list[str]:
    command = [
        sys.executable,
        __file__,
        "--config",
        args.config,
        "--batch-size",
        str(args.batch_size if args.batch_size is not None else load_cfg(args.config).data.batch_size),
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
    return command


def run_setting_with_timeout(
    args: argparse.Namespace,
    model_type: str,
    accumulate_batches: int,
) -> tuple[bool, float | None, str | None]:
    command = child_attempt_command(args, model_type, accumulate_batches)
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


def find_max_for_model(args: argparse.Namespace, model_type: str) -> ProbeResult:
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = int(load_cfg(args.config).data.batch_size)

    model_type = canonical_model_type(model_type)
    print(f"\n=== Probing {model_type} ===", flush=True)
    print(
        "physical_batch_size="
        f"{batch_size}, agents_per_sample={args.agents_per_sample}, "
        f"videos_per_microbatch={batch_size * args.agents_per_sample}",
        flush=True,
    )

    low = 0
    low_peak = None
    probe = max(args.start, 1)
    high = None
    first_error = None

    while probe <= args.max:
        print(f"[{model_type}] trying contrastive_accumulate_batches={probe}", flush=True)
        ok, peak, error = run_setting_with_timeout(args, model_type, probe)
        peak_text = "n/a" if peak is None else f"{peak:.2f} GB"
        if ok:
            print(f"[{model_type}] OK at {probe} peak={peak_text}", flush=True)
            low = probe
            low_peak = peak
            probe = next_probe_value(probe, args.start)
        else:
            print(f"[{model_type}] FAIL at {probe} peak={peak_text} error={error}", flush=True)
            if not attempt_failure_is_boundary(error):
                return ProbeResult(
                    model_type=model_type,
                    max_accumulate_batches=low,
                    first_oom_at=None,
                    batch_size=batch_size,
                    agents_per_sample=args.agents_per_sample,
                    total_agents_per_microbatch=batch_size * args.agents_per_sample,
                    peak_memory_gb=low_peak,
                    error=error,
                )
            high = probe
            first_error = error
            break

    if high is None:
        return ProbeResult(
            model_type=model_type,
            max_accumulate_batches=low,
            first_oom_at=None,
            batch_size=batch_size,
            agents_per_sample=args.agents_per_sample,
            total_agents_per_microbatch=batch_size * args.agents_per_sample,
            peak_memory_gb=low_peak,
            error=None if low > 0 else first_error,
        )

    while high - low > 1:
        mid = (low + high) // 2
        print(f"[{model_type}] binary trying {mid}", flush=True)
        ok, peak, error = run_setting_with_timeout(args, model_type, mid)
        peak_text = "n/a" if peak is None else f"{peak:.2f} GB"
        if ok:
            print(f"[{model_type}] OK at {mid} peak={peak_text}", flush=True)
            low = mid
            low_peak = peak
        else:
            print(f"[{model_type}] FAIL at {mid} peak={peak_text} error={error}", flush=True)
            if not attempt_failure_is_boundary(error):
                return ProbeResult(
                    model_type=model_type,
                    max_accumulate_batches=low,
                    first_oom_at=high,
                    batch_size=batch_size,
                    agents_per_sample=args.agents_per_sample,
                    total_agents_per_microbatch=batch_size * args.agents_per_sample,
                    peak_memory_gb=low_peak,
                    error=error,
                )
            high = mid
            first_error = error

    return ProbeResult(
        model_type=model_type,
        max_accumulate_batches=low,
        first_oom_at=high,
        batch_size=batch_size,
        agents_per_sample=args.agents_per_sample,
        total_agents_per_microbatch=batch_size * args.agents_per_sample,
        peak_memory_gb=low_peak,
        error=None if low > 0 else first_error,
    )


def print_summary(results: Iterable[ProbeResult]) -> None:
    print("\n=== Summary ===", flush=True)
    for result in results:
        peak = "n/a" if result.peak_memory_gb is None else f"{result.peak_memory_gb:.2f} GB"
        first_oom = "not reached" if result.first_oom_at is None else str(result.first_oom_at)
        status = "OK" if result.error is None else f"ERROR: {result.error}"
        print(
            f"{result.model_type}: max_contrastive_accumulate_batches="
            f"{result.max_accumulate_batches}, first_oom_at={first_oom}, "
            f"videos_per_microbatch={result.total_agents_per_microbatch}, "
            f"peak_at_max={peak}, {status}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    if args._attempt_model is not None:
        ok, peak, error = can_run_setting(
            args,
            canonical_model_type(args._attempt_model),
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

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available; this will not find GPU OOM limits.", flush=True)

    start_time = time.time()
    results = []
    for model_type in args.models:
        results.append(find_max_for_model(args, model_type))

    print_summary(results)
    print(f"\nTotal probe time: {(time.time() - start_time) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
