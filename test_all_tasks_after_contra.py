#!/usr/bin/env python3
"""
Smoke-test downstream dev runs using main_contra_with_accu checkpoints.

The script first prints checkpoint availability, then runs:
1. one selected model/map checkpoint across all benchmark tasks;
2. one selected task across every model/map checkpoint.
"""

import argparse
import csv
import gc
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Fix Windows console encoding to support Unicode characters.
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ["PYTHONIOENCODING"] = "utf-8"

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_BASE_PATH = Path(os.getenv("DATA_BASE_PATH", PROJECT_ROOT / "data"))
OUTPUT_BASE_PATH = Path(os.getenv("OUTPUT_BASE_PATH", PROJECT_ROOT / "output"))

MODELS = [
    "siglip2",
    "dinov3",
    "clip",
    "vjepa2",
    "resnet50",
]

MAPS = [
    "dust2",
    "inferno",
    "mirage",
]

UI_MASK = "all"
CHECKPOINT_FILENAME = "checkpoint/last.ckpt"

ALL_TASKS_MODEL = "siglip2"
ALL_TASKS_MAP = "dust2"
ONE_TASK_ID = "self_location_0s"

STAGE1_CHECKPOINTS = {
    ("clip", "dust2"): "main_contra_with_accu-clip-dust2-ui-all-260426-061227-kdhc",
    ("clip", "inferno"): "main_contra_with_accu-clip-inferno-ui-all-260426-062557-v5dk",
    ("clip", "mirage"): "main_contra_with_accu-clip-mirage-ui-all-260426-063549-w4kf",
    ("dinov3", "dust2"): "main_contra_with_accu-dinov3-dust2-ui-all-260426-064234-29fw",
    ("dinov3", "inferno"): "main_contra_with_accu-dinov3-inferno-ui-all-260426-065505-e4pl",
    ("dinov3", "mirage"): "main_contra_with_accu-dinov3-mirage-ui-all-260427-052436-jkwm",
    ("resnet50", "dust2"): "main_contra_with_accu-resnet50-dust2-ui-all-260427-071310-4l22",
    ("resnet50", "inferno"): "main_contra_with_accu-resnet50-inferno-ui-all-260427-073350-f0ra",
    ("resnet50", "mirage"): "main_contra_with_accu-resnet50-mirage-ui-all-260427-074405-vq44",
    ("siglip2", "dust2"): "main_contra_with_accu-siglip2-dust2-ui-all-260427-075806-ylp0",
    ("siglip2", "inferno"): "main_contra_with_accu-siglip2-inferno-ui-all-260427-080246-9h16",
    ("siglip2", "mirage"): "main_contra_with_accu-siglip2-mirage-ui-all-260427-080317-eixo",
    ("vjepa2", "dust2"): "main_contra_with_accu-vjepa2-dust2-ui-all-260427-080726-ujzu",
    ("vjepa2", "inferno"): "main_contra_with_accu-vjepa2-inferno-ui-all-260427-081410-64sg",
    ("vjepa2", "mirage"): "main_contra_with_accu-vjepa2-mirage-ui-all-260427-083324-13tq",
}


@dataclass
class TaskDefinition:
    task_id: str
    task_name: str
    category: str
    ml_form: str
    use_in_benchmark: str


@dataclass
class TestResult:
    scenario: str
    model: str
    map_name: str
    task_id: str
    success: bool
    error_msg: Optional[str] = None
    skipped: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test downstream dev runs with contrastive checkpoints."
    )
    parser.add_argument(
        "--all-tasks-model",
        default=ALL_TASKS_MODEL,
        choices=MODELS,
        help=f"Model for the all-tasks smoke test (default: {ALL_TASKS_MODEL})",
    )
    parser.add_argument(
        "--all-tasks-map",
        default=ALL_TASKS_MAP,
        choices=MAPS,
        help=f"Map for the all-tasks smoke test (default: {ALL_TASKS_MAP})",
    )
    parser.add_argument(
        "--one-task-id",
        default=ONE_TASK_ID,
        help=f"Task ID for the all-checkpoints smoke test (default: {ONE_TASK_ID})",
    )
    parser.add_argument(
        "--ui-mask",
        default=UI_MASK,
        choices=["none", "minimap_only", "all"],
        help=f"UI mask setting (default: {UI_MASK})",
    )
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help="Only print checkpoint availability and planned commands.",
    )
    parser.add_argument(
        "--extra-overrides",
        nargs="*",
        default=[],
        help="Additional main.py config overrides appended to every dev run.",
    )
    return parser.parse_args()


def checkpoint_path(model: str, map_name: str) -> Path:
    folder = STAGE1_CHECKPOINTS[(model, map_name)]
    return OUTPUT_BASE_PATH / folder / CHECKPOINT_FILENAME


def read_checkpoint_metadata(path: Path) -> str:
    """Return a compact epoch/step summary from a Lightning checkpoint."""
    try:
        import torch
    except ImportError as exc:
        return f"metadata unavailable: torch import failed: {exc}"

    try:
        try:
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        epoch = checkpoint.get("epoch", "unknown")
        global_step = checkpoint.get("global_step", "unknown")
        loops = checkpoint.get("loops", {})
        fit_loop = loops.get("fit_loop", {}) if isinstance(loops, dict) else {}
        loop_epoch = fit_loop.get("epoch_progress", {}) if isinstance(fit_loop, dict) else {}
        metadata = f"epoch={epoch}, global_step={global_step}"
        if loop_epoch:
            metadata += f", epoch_progress={loop_epoch}"
        return metadata
    except Exception as exc:
        return f"metadata unavailable: {exc}"
    finally:
        try:
            del checkpoint
        except UnboundLocalError:
            pass
        gc.collect()


def task_definitions_path(map_name: str) -> Path:
    return DATA_BASE_PATH / map_name / "labels" / "task_definitions.csv"


def load_task_definitions(map_name: str) -> list[TaskDefinition]:
    tasks = []
    path = task_definitions_path(map_name)

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append(
                TaskDefinition(
                    task_id=row["task_id"],
                    task_name=row["task_name"],
                    category=row["category"],
                    ml_form=row["ml_form"],
                    use_in_benchmark=row.get(
                        "use_in_benchmark", row.get("implemented", "yes")
                    ),
                )
            )

    return tasks


def benchmark_tasks(map_name: str) -> list[TaskDefinition]:
    return [
        task
        for task in load_task_definitions(map_name)
        if task.use_in_benchmark.lower() == "yes"
    ]


def check_checkpoint_availability() -> dict[tuple[str, str], Path]:
    print("=" * 80)
    print("Checkpoint Availability")
    print(f"OUTPUT_BASE_PATH: {OUTPUT_BASE_PATH}")
    print("=" * 80)

    available = {}
    missing = []

    for map_name in MAPS:
        print(f"\n{map_name}:")
        for model in MODELS:
            path = checkpoint_path(model, map_name)
            if path.exists():
                metadata = read_checkpoint_metadata(path)
                print(f"  FOUND   {model}: {path} ({metadata})")
                available[(model, map_name)] = path
            else:
                print(f"  MISSING {model}: {path}")
                missing.append((model, map_name, path))

    print("\n" + "=" * 80)
    print(f"Found {len(available)}/{len(STAGE1_CHECKPOINTS)} expected last checkpoints")
    if missing:
        print("Missing checkpoints:")
        for model, map_name, path in missing:
            print(f"  - {model}/{map_name}: {path}")
    else:
        print("No missing checkpoints")
    print("=" * 80)

    return available


def build_command(
    model: str,
    map_name: str,
    task_id: str,
    ui_mask: str,
    stage1_checkpoint: Path,
    extra_overrides: list[str],
) -> list[str]:
    return [
        sys.executable,
        "main.py",
        "--mode",
        "dev",
        "--task",
        "downstream",
        f"task.task_id={task_id}",
        f"data.map={map_name}",
        f"model.encoder.model_type={model}",
        f"data.ui_mask={ui_mask}",
        f"model.stage1_checkpoint={stage1_checkpoint}",
        "model.encoder.trainable=false",
        *extra_overrides,
    ]


def run_command(cmd: list[str], description: str, skip_runs: bool) -> tuple[bool, str]:
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(str(part) for part in cmd)}")
    print("=" * 60)
    sys.stdout.flush()

    if skip_runs:
        print("SKIPPED (--skip-runs)")
        return True, ""

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=PROJECT_ROOT,
        )
    except Exception as exc:
        print(f"EXCEPTION: {exc}")
        return False, str(exc)

    if result.returncode != 0:
        print(f"FAILED with return code {result.returncode}")
        stderr = result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr
        stdout = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
        if stdout:
            print(f"STDOUT tail:\n{stdout}")
        if stderr:
            print(f"STDERR tail:\n{stderr}")
        return False, stderr or stdout or f"Exit code: {result.returncode}"

    print("SUCCESS")
    return True, ""


def test_task(
    scenario: str,
    model: str,
    map_name: str,
    task_id: str,
    args: argparse.Namespace,
    available: dict[tuple[str, str], Path],
) -> TestResult:
    stage1_checkpoint = available.get((model, map_name))
    if stage1_checkpoint is None:
        return TestResult(
            scenario=scenario,
            model=model,
            map_name=map_name,
            task_id=task_id,
            success=False,
            error_msg="Missing checkpoint",
            skipped=True,
        )

    cmd = build_command(
        model=model,
        map_name=map_name,
        task_id=task_id,
        ui_mask=args.ui_mask,
        stage1_checkpoint=stage1_checkpoint,
        extra_overrides=args.extra_overrides,
    )
    success, error_msg = run_command(
        cmd,
        f"{scenario}: {model}/{map_name}/{task_id}",
        args.skip_runs,
    )
    return TestResult(
        scenario=scenario,
        model=model,
        map_name=map_name,
        task_id=task_id,
        success=success,
        error_msg=error_msg if not success else None,
    )


def run_all_tasks_for_one_checkpoint(
    args: argparse.Namespace,
    available: dict[tuple[str, str], Path],
) -> list[TestResult]:
    tasks = benchmark_tasks(args.all_tasks_map)
    print("\n" + "#" * 80)
    print(
        f"# Phase 1: all benchmark tasks for "
        f"{args.all_tasks_model}/{args.all_tasks_map}"
    )
    print(f"# Tasks: {len(tasks)}")
    print("#" * 80)

    results = []
    for index, task in enumerate(tasks, 1):
        print(f"\nTask {index}/{len(tasks)}: {task.task_id}")
        results.append(
            test_task(
                scenario="all_tasks_one_checkpoint",
                model=args.all_tasks_model,
                map_name=args.all_tasks_map,
                task_id=task.task_id,
                args=args,
                available=available,
            )
        )
    return results


def run_one_task_for_all_checkpoints(
    args: argparse.Namespace,
    available: dict[tuple[str, str], Path],
) -> list[TestResult]:
    print("\n" + "#" * 80)
    print(f"# Phase 2: task {args.one_task_id} for all model/map checkpoints")
    print(f"# Checkpoints: {len(STAGE1_CHECKPOINTS)}")
    print("#" * 80)

    results = []
    for map_name in MAPS:
        task_ids = {task.task_id for task in benchmark_tasks(map_name)}
        if args.one_task_id not in task_ids:
            print(f"\nSKIP {map_name}: task {args.one_task_id} is not a benchmark task")
            for model in MODELS:
                results.append(
                    TestResult(
                        scenario="one_task_all_checkpoints",
                        model=model,
                        map_name=map_name,
                        task_id=args.one_task_id,
                        success=False,
                        error_msg="Task is not a benchmark task for this map",
                        skipped=True,
                    )
                )
            continue

        for model in MODELS:
            results.append(
                test_task(
                    scenario="one_task_all_checkpoints",
                    model=model,
                    map_name=map_name,
                    task_id=args.one_task_id,
                    args=args,
                    available=available,
                )
            )
    return results


def print_summary(results: list[TestResult]) -> bool:
    print("\n" + "=" * 80)
    print("Smoke Test Summary")
    print("=" * 80)

    passed = 0
    failed = 0
    skipped = 0

    for result in results:
        if result.skipped:
            status = "SKIP"
            skipped += 1
        elif result.success:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1
        print(
            f"{status:4} {result.scenario}: "
            f"{result.model}/{result.map_name}/{result.task_id}"
        )

    print("-" * 80)
    print(f"Total: {passed} passed, {failed} failed, {skipped} skipped")

    failures = [result for result in results if not result.success and not result.skipped]
    if failures:
        print("\nFailures:")
        for result in failures:
            print(
                f"  - {result.scenario} {result.model}/{result.map_name}/"
                f"{result.task_id}: {result.error_msg[:300] if result.error_msg else ''}"
            )

    return failed == 0 and skipped == 0


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("X-EGO Downstream After Contrastive Smoke Test")
    print(f"DATA_BASE_PATH: {DATA_BASE_PATH}")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"UI Mask: {args.ui_mask}")
    print("=" * 80)

    available = check_checkpoint_availability()

    results = []
    results.extend(run_all_tasks_for_one_checkpoint(args, available))
    results.extend(run_one_task_for_all_checkpoints(args, available))

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
