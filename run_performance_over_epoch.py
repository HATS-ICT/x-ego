#!/usr/bin/env python3
"""
Run downstream experiments across epochs to track performance over training.

This script trains downstream tasks using checkpoints from different epochs
to analyze how contrastive pre-training affects downstream performance over time.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# Fix Windows console encoding
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ["PYTHONIOENCODING"] = "utf-8"

load_dotenv()

OUTPUT_BASE_PATH = os.getenv("OUTPUT_BASE_PATH")
if OUTPUT_BASE_PATH is None:
    raise ValueError("OUTPUT_BASE_PATH not found in .env file")

# ============================================================
# Configuration
# ============================================================

# Tasks to evaluate
EVAL_TASKS = [
    "self_kill_5s",
    "self_kill_10s",
    "self_death_5s",
    "self_death_10s",
    "self_inCombat",
    "global_anyKill_5s",
    "global_anyKill_10s",
    "teammate_inCombat",
    "global_bombSite",
    "global_roundWinner",
]

# Checkpoint patterns for each model type
# Format: (folder_pattern, checkpoint_files)
# checkpoint_files is a list of (epoch, filename) tuples
MODEL_CHECKPOINTS = {
    "siglip2": {
        "folder": "main_ui_cover-siglip2-ui-all-260122-064933-md8t",
        "checkpoints": [
            (0, "main_ui_cover-siglip2-ui-all-e00-s000644-l26.9483.ckpt"),
            (1, "main_ui_cover-siglip2-ui-all-e01-s001288-l25.9370.ckpt"),
            (2, "main_ui_cover-siglip2-ui-all-e02-s001932-l26.5292.ckpt"),
            (3, "main_ui_cover-siglip2-ui-all-e03-s002576-l24.7947.ckpt"),
            (4, "main_ui_cover-siglip2-ui-all-e04-s003220-l24.4860.ckpt"),
            (9, "main_ui_cover-siglip2-ui-all-e09-s006440-l21.5981.ckpt"),
            (14, "main_ui_cover-siglip2-ui-all-e14-s009660-l19.9258.ckpt"),
            (19, "main_ui_cover-siglip2-ui-all-e19-s012880-l18.9469.ckpt"),
            (24, "main_ui_cover-siglip2-ui-all-e24-s016100-l18.6912.ckpt"),
            (29, "main_ui_cover-siglip2-ui-all-e29-s019320-l17.6591.ckpt"),
            (34, "main_ui_cover-siglip2-ui-all-e34-s022540-l17.0135.ckpt"),
            # (39, "main_ui_cover-siglip2-ui-all-e39-s025760-l16.8452.ckpt"),  # excluded
        ],
    },
    "dinov2": {
        "folder": "main_ui_cover-dinov2-ui-all-260122-035704-my8c",
        "checkpoints": [
            (0, "main_ui_cover-dinov2-ui-all-e00-s000644-l13.9043.ckpt"),
            (1, "main_ui_cover-dinov2-ui-all-e01-s001288-l13.7108.ckpt"),
            (2, "main_ui_cover-dinov2-ui-all-e02-s001932-l12.7184.ckpt"),
            (3, "main_ui_cover-dinov2-ui-all-e03-s002576-l12.5499.ckpt"),
            (4, "main_ui_cover-dinov2-ui-all-e04-s003220-l12.7051.ckpt"),
            (9, "main_ui_cover-dinov2-ui-all-e09-s006440-l11.4426.ckpt"),
            (14, "main_ui_cover-dinov2-ui-all-e14-s009660-l11.9038.ckpt"),
            (19, "main_ui_cover-dinov2-ui-all-e19-s012880-l11.2415.ckpt"),
            (24, "main_ui_cover-dinov2-ui-all-e24-s016100-l11.2832.ckpt"),
            (29, "main_ui_cover-dinov2-ui-all-e29-s019320-l11.1802.ckpt"),
            (34, "main_ui_cover-dinov2-ui-all-e34-s022540-l11.1013.ckpt"),
            # (39, "main_ui_cover-dinov2-ui-all-e39-s025760-l11.1983.ckpt"),  # excluded
        ],
    },
    "clip": {
        "folder": "main_ui_cover-clip-ui-all-260124-084053-wxbo",
        "checkpoints": [
            (0, "contrastive-clip-e00-s000403-l29.0643.ckpt"),
            (1, "contrastive-clip-e01-s000806-l33.5950.ckpt"),
            (2, "contrastive-clip-e02-s001209-l33.8955.ckpt"),
            (3, "contrastive-clip-e03-s001612-l32.5149.ckpt"),
            (4, "contrastive-clip-e04-s002015-l33.3639.ckpt"),
            (9, "contrastive-clip-e09-s004030-l29.3537.ckpt"),
            (14, "contrastive-clip-e14-s006045-l24.7550.ckpt"),
            (19, "contrastive-clip-e19-s008060-l22.4778.ckpt"),
            (24, "contrastive-clip-e24-s010075-l19.8016.ckpt"),
            (29, "contrastive-clip-e29-s012090-l18.9290.ckpt"),
            (34, "contrastive-clip-e34-s014105-l18.1432.ckpt"),
            # (39, "contrastive-clip-e39-s016120-l17.7273.ckpt"),  # excluded
        ],
    },
}

# UI mask setting
UI_MASK = "all"


# ============================================================
# Functions
# ============================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run downstream experiments across epochs."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=list(MODEL_CHECKPOINTS.keys()),
        help="Model type to run experiments for",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for the experiment (default: 1)",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=None,
        help="Start from a specific epoch (default: None, start from beginning)",
    )
    parser.add_argument(
        "--start-task",
        type=str,
        default=None,
        help="Start from a specific task within an epoch (default: None)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=32,
        help="Number of data loader workers (default: 32)",
    )
    return parser.parse_args()


def run_experiment(
    model_type: str,
    task_id: str,
    epoch: int,
    checkpoint_path: Path,
    seed: int,
    index: int,
    total: int,
    num_workers: int,
    dry_run: bool = False,
) -> bool:
    """Run a single downstream experiment."""
    run_name = f"epoch_probe-{model_type}-{task_id}-all-e{epoch:02d}-s{seed}"

    print(f"\n{'='*70}")
    print(f"[{index}/{total}] seed={seed}, epoch={epoch}, task={task_id}")
    print(f"Run name: {run_name}")
    print("=" * 70)

    cmd = [
        sys.executable,
        "main.py",
        "--mode",
        "train",
        "--task",
        "downstream",
        f"task.task_id={task_id}",
        f"model.encoder.model_type={model_type}",
        f"data.ui_mask={UI_MASK}",
        f"model.stage1_checkpoint={checkpoint_path}",
        f"meta.run_name={run_name}",
        f"meta.seed={seed}",
        f"data.num_workers={num_workers}",
    ]

    print(f"Command: {' '.join(cmd)}")

    if dry_run:
        print("[DRY RUN] Skipping execution")
        return True

    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    if result.returncode != 0:
        print(f"FAILED with return code {result.returncode}")
        return False

    print("SUCCESS")
    return True


def main():
    args = parse_args()

    model_type = args.model_type
    model_config = MODEL_CHECKPOINTS[model_type]
    folder = model_config["folder"]
    checkpoints = model_config["checkpoints"]

    # Filter checkpoints if start_epoch is specified
    if args.start_epoch is not None:
        checkpoints = [(e, c) for e, c in checkpoints if e >= args.start_epoch]

    seed = args.seed

    print("=" * 70)
    print("Performance Over Epoch Experiment")
    print("=" * 70)
    print(f"Model type: {model_type}")
    print(f"Checkpoint folder: {folder}")
    print(f"Epochs: {[e for e, _ in checkpoints]}")
    print(f"Tasks: {EVAL_TASKS}")
    print(f"Seed: {seed}")
    print(f"UI mask: {UI_MASK}")
    if args.dry_run:
        print("[DRY RUN MODE]")
    print("=" * 70)

    # Calculate total experiments
    num_epochs = len(checkpoints)
    num_tasks = len(EVAL_TASKS)
    total_experiments = num_epochs * num_tasks

    print(f"\nTotal experiments: {num_epochs} epochs × {num_tasks} tasks = {total_experiments}")

    results = []
    experiment_idx = 0

    # Outer loop: epochs
    for epoch, checkpoint_file in checkpoints:
        checkpoint_path = Path(OUTPUT_BASE_PATH) / folder / "checkpoint" / checkpoint_file

        print(f"\n{'-'*70}")
        print(f"Epoch {epoch} checkpoint: {checkpoint_file}")
        print("-" * 70)

        # Inner loop: tasks
        for task_id in EVAL_TASKS:
            # Skip to start_task if specified (only for first epoch)
            if (
                args.start_task is not None
                and args.start_epoch is not None
                and epoch == args.start_epoch
            ):
                if EVAL_TASKS.index(task_id) < EVAL_TASKS.index(args.start_task):
                    continue

            experiment_idx += 1

            success = run_experiment(
                model_type=model_type,
                task_id=task_id,
                epoch=epoch,
                checkpoint_path=checkpoint_path,
                seed=seed,
                index=experiment_idx,
                total=total_experiments,
                num_workers=args.num_workers,
                dry_run=args.dry_run,
            )

            results.append((seed, epoch, task_id, success))

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r[3])
    failed = len(results) - passed

    if failed > 0:
        print("\nFailed experiments:")
        for seed, epoch, task_id, success in results:
            if not success:
                print(f"  - seed={seed}, epoch={epoch}, task={task_id}")

    print(f"\nTotal: {passed} passed, {failed} failed out of {len(results)} experiments")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

# Resume
# python run_performance_over_epoch.py --model-type siglip2 --seed 3 --start-epoch 1 --start-task global_anyKill_10s