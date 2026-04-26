#!/usr/bin/env python3
"""
Run one formal contrastive accumulation window for every model/map setup.

This is a pre-flight check for the large contrastive training sweep. Each run
uses the same model, map, physical batch size, and contrastive accumulation
setting as formal training, but limits training to exactly one accumulation
window so the forward/cache/flush/backward path is exercised once.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ["PYTHONIOENCODING"] = "utf-8"


MODEL_TYPES = ["siglip2", "dinov3", "clip", "vjepa2", "resnet50"]
MAPS = ["dust2", "inferno", "mirage"]
UI_MASK = "all"
NUM_WORKERS = 8
CHECK_TORCH_COMPILE = sys.platform.startswith("linux")

TRAINING_SETTINGS = {
    "clip": (128, 8),
    "siglip2": (64, 16),
    "dinov3": (32, 32),
    "vjepa2": (32, 32),
    "resnet50": (32, 32),
}


@dataclass
class TestResult:
    model_type: str
    map_name: str
    batch_size: int
    accumulate_batches: int
    success: bool
    error_msg: Optional[str] = None

    @property
    def virtual_videos(self) -> int:
        return self.batch_size * self.accumulate_batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test formal contrastive model/map setups for one accumulation window."
    )
    parser.add_argument("--models", nargs="+", default=MODEL_TYPES, choices=MODEL_TYPES)
    parser.add_argument("--maps", nargs="+", default=MAPS, choices=MAPS)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--keep-wandb", action="store_true", help="Keep WandB enabled.")
    return parser.parse_args()


def compile_overrides() -> list[str]:
    return [f"training.torch_compile={str(CHECK_TORCH_COMPILE).lower()}"]


def run_command(cmd: list[str], description: str) -> tuple[bool, str, str]:
    print(f"\n{'=' * 80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=Path(__file__).parent,
        )
    except Exception as exc:
        print(f"EXCEPTION: {exc}")
        return False, "", str(exc)

    if result.returncode != 0:
        print(f"FAILED with return code {result.returncode}")
        stderr_tail = result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr
        stdout_tail = result.stdout[-1500:] if len(result.stdout) > 1500 else result.stdout
        print(f"STDOUT tail:\n{stdout_tail}")
        print(f"STDERR tail:\n{stderr_tail}")
        return False, result.stdout, result.stderr

    print("SUCCESS")
    return True, result.stdout, result.stderr


def test_contrastive_setup(
    model_type: str,
    map_name: str,
    num_workers: int,
    keep_wandb: bool,
) -> TestResult:
    batch_size, accumulate_batches = TRAINING_SETTINGS[model_type]
    run_name = f"setup-contra-{model_type}-{map_name}-b{batch_size}-a{accumulate_batches}"

    cmd = [
        sys.executable,
        "main.py",
        "--mode",
        "train",
        "--task",
        "contrastive",
        f"model.encoder.model_type={model_type}",
        f"data.map={map_name}",
        f"data.ui_mask={UI_MASK}",
        f"data.batch_size={batch_size}",
        f"data.num_workers={num_workers}",
        "training.max_epochs=1",
        "training.max_steps=1",
        f"training.limit_train_batches={accumulate_batches}",
        "training.limit_val_batches=0",
        "training.limit_test_batches=0",
        "training.accumulate_grad_batches=1",
        f"training.contrastive_accumulate_batches={accumulate_batches}",
        "training.log_every_n_steps=1",
        "checkpoint.epoch.save_top_k=0",
        "checkpoint.epoch.save_last=true",
        "checkpoint.step.save_top_k=0",
        f"meta.exp_name=setup_contrastive_only",
        f"meta.run_name={run_name}",
        *compile_overrides(),
    ]
    if not keep_wandb:
        cmd.append("wandb.enabled=false")

    success, stdout, stderr = run_command(
        cmd,
        (
            f"{model_type} on {map_name}: batch_size={batch_size}, "
            f"accumulate={accumulate_batches}, virtual_videos={batch_size * accumulate_batches}"
        ),
    )

    return TestResult(
        model_type=model_type,
        map_name=map_name,
        batch_size=batch_size,
        accumulate_batches=accumulate_batches,
        success=success,
        error_msg=stderr[-800:] if not success else None,
    )


def print_summary(results: list[TestResult]) -> bool:
    print("\n" + "=" * 80)
    print("CONTRASTIVE SETUP TEST SUMMARY")
    print("=" * 80)

    total_failed = 0
    for model_type in MODEL_TYPES:
        model_results = [result for result in results if result.model_type == model_type]
        if not model_results:
            continue

        batch_size, accumulate_batches = TRAINING_SETTINGS[model_type]
        print(
            f"\n{model_type} "
            f"(batch_size={batch_size}, accumulate={accumulate_batches}, "
            f"virtual_videos={batch_size * accumulate_batches}):"
        )
        for result in model_results:
            status = "PASS" if result.success else "FAIL"
            print(f"  {result.map_name}: {status}")
            if not result.success:
                total_failed += 1
                if result.error_msg:
                    preview = result.error_msg[:300].replace("\n", " ")
                    print(f"    Error: {preview}...")

    total_passed = len(results) - total_failed
    print(f"\nTotal: {total_passed} passed, {total_failed} failed")
    return total_failed == 0


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("X-EGO Contrastive Setup Test")
    print(f"Models: {', '.join(args.models)}")
    print(f"Maps: {', '.join(args.maps)}")
    print(f"UI mask: {UI_MASK}")
    print(f"num_workers: {args.num_workers}")
    print(f"torch.compile check: {'enabled' if CHECK_TORCH_COMPILE else 'disabled'}")
    print(f"WandB: {'enabled' if args.keep_wandb else 'disabled'}")
    print("Training settings:")
    for model_type in args.models:
        batch_size, accumulate_batches = TRAINING_SETTINGS[model_type]
        print(
            f"  {model_type}: batch_size={batch_size}, "
            f"accumulate={accumulate_batches}, virtual_videos={batch_size * accumulate_batches}"
        )
    print("=" * 80)

    results: list[TestResult] = []
    for model_type in args.models:
        for map_name in args.maps:
            results.append(
                test_contrastive_setup(
                    model_type=model_type,
                    map_name=map_name,
                    num_workers=args.num_workers,
                    keep_wandb=args.keep_wandb,
                )
            )

    sys.exit(0 if print_summary(results) else 1)


if __name__ == "__main__":
    main()
