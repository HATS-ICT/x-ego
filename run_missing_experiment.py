#!/usr/bin/env python3
"""
Run missing downstream experiments.

Configure missing experiments in MISSING_EXPERIMENTS below.
"""

import os
import random
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
    # siglip2 finetuned missing 1 each (count=3, need 1 more)
    ("siglip2", "enemy_aliveCount", "finetuned"),
    ("siglip2", "self_inCombat", "finetuned"),
    ("siglip2", "teammate_inCombat", "finetuned"),
    ("siglip2", "self_speed", "finetuned"),
    ("siglip2", "teammate_speed", "finetuned"),
    ("siglip2", "global_bombPlanted", "finetuned"),
    ("siglip2", "global_bombSite", "finetuned"),
    ("siglip2", "global_willPlant", "finetuned"),
    ("siglip2", "global_postPlantOutcome", "finetuned"),
    ("siglip2", "global_roundWinner", "finetuned"),
    ("siglip2", "global_roundOutcome", "finetuned"),
    # dinov2 baseline missing 1 each (count=3, need 1 more)
    ("dinov2", "teammate_location_20s", "baseline"),
    ("dinov2", "enemy_location_0s", "baseline"),
    ("dinov2", "enemy_location_5s", "baseline"),
    ("dinov2", "enemy_location_10s", "baseline"),
    ("dinov2", "enemy_location_20s", "baseline"),
    ("dinov2", "self_kill_5s", "baseline"),
    ("dinov2", "self_kill_10s", "baseline"),
    ("dinov2", "self_kill_20s", "baseline"),
    ("dinov2", "self_death_5s", "baseline"),
    ("dinov2", "self_death_10s", "baseline"),
    ("dinov2", "self_death_20s", "baseline"),
    ("dinov2", "global_anyKill_5s", "baseline"),
    ("dinov2", "global_anyKill_10s", "baseline"),
    ("dinov2", "global_anyKill_20s", "baseline"),
    ("dinov2", "teammate_aliveCount", "baseline"),
    ("dinov2", "enemy_aliveCount", "baseline"),
    ("dinov2", "self_inCombat", "baseline"),
    ("dinov2", "teammate_inCombat", "baseline"),
    ("dinov2", "self_speed", "baseline"),
    ("dinov2", "teammate_speed", "baseline"),
    ("dinov2", "global_bombPlanted", "baseline"),
    ("dinov2", "global_bombSite", "baseline"),
    ("dinov2", "global_willPlant", "baseline"),
    ("dinov2", "global_postPlantOutcome", "baseline"),
    ("dinov2", "global_roundWinner", "baseline"),
    ("dinov2", "global_roundOutcome", "baseline"),
    # dinov2 finetuned missing 1 each (count=3, need 1 more)
    ("dinov2", "teammate_location_20s", "finetuned"),
    ("dinov2", "enemy_location_0s", "finetuned"),
    ("dinov2", "enemy_location_5s", "finetuned"),
    ("dinov2", "enemy_location_10s", "finetuned"),
    ("dinov2", "enemy_location_20s", "finetuned"),
    ("dinov2", "self_kill_5s", "finetuned"),
    ("dinov2", "self_kill_10s", "finetuned"),
    ("dinov2", "self_kill_20s", "finetuned"),
    ("dinov2", "self_death_5s", "finetuned"),
    ("dinov2", "self_death_10s", "finetuned"),
    ("dinov2", "self_death_20s", "finetuned"),
    ("dinov2", "global_anyKill_5s", "finetuned"),
    ("dinov2", "global_anyKill_10s", "finetuned"),
    ("dinov2", "global_anyKill_20s", "finetuned"),
    ("dinov2", "teammate_aliveCount", "finetuned"),
    ("dinov2", "enemy_aliveCount", "finetuned"),
    ("dinov2", "self_inCombat", "finetuned"),
    ("dinov2", "teammate_inCombat", "finetuned"),
    ("dinov2", "self_speed", "finetuned"),
    ("dinov2", "teammate_speed", "finetuned"),
    ("dinov2", "global_bombPlanted", "finetuned"),
    ("dinov2", "global_bombSite", "finetuned"),
    ("dinov2", "global_willPlant", "finetuned"),
    ("dinov2", "global_postPlantOutcome", "finetuned"),
    ("dinov2", "global_roundWinner", "finetuned"),
    ("dinov2", "global_roundOutcome", "finetuned"),
    # vjepa2 baseline missing 1 (count=3, need 1 more)
    ("vjepa2", "global_anyKill_5s", "baseline"),
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

    seed = random.randint(0, 2**31 - 1)
    cmd = [
        sys.executable, "main.py",
        "--mode", "train",
        "--task", "downstream",
        f"task.task_id={task_id}",
        f"model.encoder.model_type={model}",
        "data.ui_mask=all",
        f"meta.seed={seed}",
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
