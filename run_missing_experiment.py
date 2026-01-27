#!/usr/bin/env python3
"""
Run missing downstream experiments.

Configure missing experiments in MISSING_EXPERIMENTS below.
"""

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OUTPUT_BASE_PATH = os.getenv("OUTPUT_BASE_PATH")
if OUTPUT_BASE_PATH is None:
    raise ValueError("OUTPUT_BASE_PATH not found in .env file")

# ============================================================
# Configuration: Missing experiments (model, task_id, init_type)
#   init_type: "baseline" (no checkpoint) or "finetuned" (with checkpoint)
# ============================================================
MISSING_EXPERIMENTS = [
    ("clip", "self_speed", "finetuned"),
    ("clip", "teammate_speed", "finetuned"),
]

# Stage1 checkpoints for finetuned experiments
STAGE1_CHECKPOINTS = {
    "clip": "main_ui_cover-clip-ui-all-260124-084053-wxbo",
    "dinov2": "main_ui_cover-dinov2-ui-all-260122-035704-my8c",
    "siglip2": "main_ui_cover-siglip2-ui-all-260122-064933-md8t",
    "vjepa2": "main_ui_cover-vjepa2-ui-all-260122-072237-nrz4",
}

# ============================================================


def run_experiment(model: str, task_id: str, init_type: str, index: int, total: int) -> bool:
    """Run a single downstream experiment."""
    print(f"\n{'='*60}")
    print(f"[{index}/{total}] {model} - {task_id} - {init_type}")
    print("=" * 60)

    cmd = [
        sys.executable, "main.py",
        "--mode", "train",
        "--task", "downstream",
        f"task.task_id={task_id}",
        f"model.encoder.model_type={model}",
        "data.ui_mask=all",
    ]

    if init_type == "baseline":
        cmd.append(f"meta.run_name=probe-{model}-{task_id}-all")
    elif init_type == "finetuned":
        checkpoint_folder = STAGE1_CHECKPOINTS.get(model)
        if checkpoint_folder is None:
            print(f"ERROR: No checkpoint found for model '{model}'")
            return False
        checkpoint_path = Path(OUTPUT_BASE_PATH) / checkpoint_folder / "checkpoint" / "last.ckpt"
        cmd.append(f"model.stage1_checkpoint={checkpoint_path}")
        cmd.append(f"meta.run_name=probe-{model}-{task_id}-all-finetuned")
    else:
        print(f"ERROR: Unknown init_type '{init_type}'")
        return False

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print(f"FAILED with return code {result.returncode}")
        return False
    
    print("SUCCESS")
    return True


def main():
    print("=" * 60)
    print("Running missing experiments")
    print("=" * 60)

    total = len(MISSING_EXPERIMENTS)
    results = []

    for i, (model, task_id, init_type) in enumerate(MISSING_EXPERIMENTS, 1):
        success = run_experiment(model, task_id, init_type, i, total)
        results.append((model, task_id, init_type, success))

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print("=" * 60)
    
    passed = sum(1 for r in results if r[3])
    failed = total - passed
    
    for model, task_id, init_type, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {model} - {task_id} - {init_type}")
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
