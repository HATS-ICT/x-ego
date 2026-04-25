#!/usr/bin/env python3
"""
Test script to verify the full pipeline works for all model/map setups.

Tests three settings for each model and map with task self_location_0s:
1. Contrastive only (Stage 1)
2. Baseline downstream (off-the-shelf model)
3. Downstream with saved contrastive checkpoint (Stage 2)

All tests run in dev mode for quick verification.
"""

import subprocess
import sys

# Fix Windows console encoding to support Unicode characters
if sys.platform == "win32":
    # Set UTF-8 encoding for stdout and stderr
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    # Also set environment variable for subprocesses
    import os
    os.environ["PYTHONIOENCODING"] = "utf-8"

from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# Available model types from cheatsheet
MODEL_TYPES = ["siglip2", "dinov3", "clip", "vjepa2", "resnet50"]
MAPS = ["dust2", "inferno", "mirage"]
TASK_ID = "self_location_0s"
CHECK_TORCH_COMPILE = sys.platform.startswith("linux")


@dataclass
class TestResult:
    name: str
    model_type: str
    map_name: str
    success: bool
    error_msg: Optional[str] = None
    checkpoint_path: Optional[str] = None


def run_command(cmd: list[str], description: str) -> tuple[bool, str, str]:
    """Run a command and return (success, stdout, stderr)."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("="*60)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=Path(__file__).parent
        )

        if result.returncode != 0:
            print(f"FAILED with return code {result.returncode}")
            print(f"STDERR:\n{result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr}")
            return False, result.stdout, result.stderr

        print("SUCCESS")
        return True, result.stdout, result.stderr

    except Exception as e:
        print(f"EXCEPTION: {e}")
        return False, "", str(e)


def compile_overrides() -> list[str]:
    """Enable torch.compile coverage on Linux setup tests."""
    return [f"training.torch_compile={str(CHECK_TORCH_COMPILE).lower()}"]


def find_checkpoint(stdout: str, stderr: str, model_type: str, map_name: str) -> Optional[str]:
    """Extract the checkpoint path from the experiment created by this run."""
    combined = (stdout or "") + (stderr or "")

    # First, try to find the experiment directory from the output
    experiment_dir = None
    for line in combined.split("\n"):
        if "Created experiment directory" in line or "experiment directory:" in line.lower():
            # Try to extract the path - look for the path after the colon or equals sign
            if ":" in line:
                # Extract everything after the colon
                path_part = line.split(":", 1)[1].strip()
                # Remove any trailing punctuation
                path_part = path_part.strip("'\":,.")
                potential_path = Path(path_part)
                if potential_path.exists() and potential_path.is_dir():
                    experiment_dir = potential_path
                    break
            # Also try splitting by spaces and looking for paths
            parts = line.split()
            for part in parts:
                if "output" in part.lower() or "contrastive" in part.lower():
                    # Check if it's a valid path
                    potential_path = Path(part.strip("'\":,."))
                    if potential_path.exists() and potential_path.is_dir():
                        experiment_dir = potential_path
                        break
                    # Also check if it contains the full path
                    if "\\" in part or "/" in part:
                        potential_path = Path(part.strip("'\":,."))
                        if potential_path.exists() and potential_path.is_dir():
                            experiment_dir = potential_path
                            break

    # If we found the experiment directory, look for checkpoints there
    if experiment_dir:
        if f"contrastive-{model_type}-{map_name}" not in experiment_dir.name:
            return None
        checkpoint_dir = experiment_dir / "checkpoint"
        if checkpoint_dir.exists():
            last_ckpt = checkpoint_dir / "last.ckpt"
            if last_ckpt.exists():
                return str(last_ckpt)

    return None


def test_contrastive(model_type: str, map_name: str) -> TestResult:
    """Test contrastive learning (Stage 1)."""
    cmd = [
        sys.executable, "main.py",
        "--mode", "dev",
        "--task", "contrastive",
        f"model.encoder.model_type={model_type}",
        f"data.map={map_name}",
        f"meta.run_name=contrastive-{model_type}-{map_name}",
        *compile_overrides(),
    ]

    success, stdout, stderr = run_command(
        cmd,
        f"Contrastive training with {model_type} on {map_name}"
    )

    checkpoint_path = find_checkpoint(stdout, stderr, model_type, map_name) if success else None

    return TestResult(
        name="contrastive",
        model_type=model_type,
        map_name=map_name,
        success=success,
        error_msg=stderr[-500:] if not success else None,
        checkpoint_path=checkpoint_path
    )


def test_baseline_downstream(model_type: str, map_name: str) -> TestResult:
    """Test baseline downstream (off-the-shelf encoder)."""
    cmd = [
        sys.executable, "main.py",
        "--mode", "dev",
        "--task", "downstream",
        f"task.task_id={TASK_ID}",
        f"model.encoder.model_type={model_type}",
        f"data.map={map_name}",
        f"meta.run_name=probe-{TASK_ID}-{model_type}-{map_name}",
        *compile_overrides(),
    ]

    success, stdout, stderr = run_command(
        cmd,
        f"Baseline downstream ({model_type}) on {TASK_ID} / {map_name}"
    )

    return TestResult(
        name="baseline_downstream",
        model_type=model_type,
        map_name=map_name,
        success=success,
        error_msg=stderr[-500:] if not success else None
    )


def test_downstream_with_checkpoint(model_type: str, map_name: str, checkpoint_path: str) -> TestResult:
    """Test downstream with pretrained contrastive checkpoint."""
    cmd = [
        sys.executable, "main.py",
        "--mode", "dev",
        "--task", "downstream",
        f"task.task_id={TASK_ID}",
        f"model.encoder.model_type={model_type}",
        f"data.map={map_name}",
        f"model.stage1_checkpoint={checkpoint_path}",
        f"meta.run_name=probe-{TASK_ID}-{model_type}-{map_name}-stage1",
        *compile_overrides(),
    ]

    success, stdout, stderr = run_command(
        cmd,
        f"Downstream with checkpoint ({model_type}) on {TASK_ID} / {map_name}"
    )

    return TestResult(
        name="downstream_with_checkpoint",
        model_type=model_type,
        map_name=map_name,
        success=success,
        error_msg=stderr[-500:] if not success else None
    )


def print_summary(results: list[TestResult]):
    """Print a summary of all test results."""
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    # Group by model type and map
    by_model_map = {}
    for r in results:
        key = (r.model_type, r.map_name)
        if key not in by_model_map:
            by_model_map[key] = []
        by_model_map[key].append(r)

    total_passed = 0
    total_failed = 0

    for model_type in MODEL_TYPES:
        model_results = [key for key in by_model_map if key[0] == model_type]
        if not model_results:
            continue

        print(f"\n{model_type}:")
        for map_name in MAPS:
            key = (model_type, map_name)
            if key not in by_model_map:
                continue

            print(f"  {map_name}:")
            for r in by_model_map[key]:
                status = "PASS" if r.success else "FAIL"
                symbol = "CHECK" if r.success else "FAIL"
                print(f"    {symbol} {r.name}: {status}")

                if r.success:
                    total_passed += 1
                else:
                    total_failed += 1
                    if r.error_msg:
                        # Print truncated error
                        error_preview = r.error_msg[:200].replace("\n", " ")
                        print(f"        Error: {error_preview}...")

    print(f"\n{'='*80}")
    print(f"Total: {total_passed} passed, {total_failed} failed")
    print("="*80)

    return total_failed == 0


def main():
    """Run all tests sequentially."""
    print("="*80)
    print("X-EGO Pipeline Test: All Model/Map Setups")
    print(f"Task: {TASK_ID}")
    print(f"Models: {', '.join(MODEL_TYPES)}")
    print(f"Maps: {', '.join(MAPS)}")
    print(f"torch.compile check: {'enabled' if CHECK_TORCH_COMPILE else 'disabled'}")
    print("="*80)

    results: list[TestResult] = []

    for model_type in MODEL_TYPES:
        for map_name in MAPS:
            print(f"\n{'#'*80}")
            print(f"# Testing model: {model_type} | map: {map_name}")
            print("#"*80)

            # 1. Contrastive only
            contrastive_result = test_contrastive(model_type, map_name)
            results.append(contrastive_result)

            # 2. Baseline downstream
            baseline_result = test_baseline_downstream(model_type, map_name)
            results.append(baseline_result)

            # 3. Downstream with checkpoint (only if contrastive succeeded and produced checkpoint)
            if contrastive_result.success and contrastive_result.checkpoint_path:
                checkpoint_result = test_downstream_with_checkpoint(
                    model_type,
                    map_name,
                    contrastive_result.checkpoint_path
                )
                results.append(checkpoint_result)
            else:
                # Record as skipped/failed
                results.append(TestResult(
                    name="downstream_with_checkpoint",
                    model_type=model_type,
                    map_name=map_name,
                    success=False,
                    error_msg="Skipped: No checkpoint from contrastive training"
                ))

    # Print summary
    all_passed = print_summary(results)

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
