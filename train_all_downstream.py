#!/usr/bin/env python3
"""
Train downstream tasks on all implemented tasks.

Runs downstream training (stage 2) on all tasks defined in
data/labels/task_definitions.csv in train mode.
"""

import subprocess
import sys

# Fix Windows console encoding to support Unicode characters
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    import os
    os.environ['PYTHONIOENCODING'] = 'utf-8'
else:
    import os

import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# CONFIGURATION - Modify these settings as needed
# =============================================================================

# Resume from a specific task (set to None to start from beginning)
# Example: "self_inCombat" to start from task 24/35
# START_FROM_TASK: Optional[str] = "self_inCombat"
START_FROM_TASK: Optional[str] = None

# Model type to use for the encoder
MODEL_TYPE = "siglip2"

# UI mask setting: "none", "minimap_only", or "all"
UI_MASK = "none"

# Stage 1 checkpoint path (set to None for baseline/off-the-shelf HuggingFace)
# Example: "contra-siglip2-260115-123456-abc1/checkpoints/last.ckpt"
STAGE1_CHECKPOINT: Optional[str] = None

# WandB group name (optional, set to None to disable)
WANDB_GROUP: Optional[str] = None

# Additional config overrides (list of "key=value" strings)
EXTRA_OVERRIDES: list[str] = []

# =============================================================================

# Read DATA_BASE_PATH from .env file
DATA_BASE_PATH = os.getenv("DATA_BASE_PATH")
if DATA_BASE_PATH is None:
    raise ValueError("DATA_BASE_PATH not found in .env file")

TASK_DEFINITIONS_PATH = Path(DATA_BASE_PATH) / "labels" / "task_definitions.csv"


@dataclass
class TaskDefinition:
    task_id: str
    task_name: str
    category: str
    ml_form: str
    implemented: str


@dataclass
class TrainResult:
    task_id: str
    task_name: str
    category: str
    success: bool
    error_msg: Optional[str] = None


def load_task_definitions() -> list[TaskDefinition]:
    """Load task definitions from CSV."""
    tasks = []
    
    with open(TASK_DEFINITIONS_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append(TaskDefinition(
                task_id=row['task_id'],
                task_name=row['task_name'],
                category=row['category'],
                ml_form=row['ml_form'],
                implemented=row['implemented']
            ))
    
    return tasks


def run_command(cmd: list[str], description: str) -> tuple[bool, str]:
    """Run a command with live output and return (success, error_msg)."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    sys.stdout.flush()
    
    try:
        # Run with live output to terminal
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent
        )
        
        if result.returncode != 0:
            print(f"\nFAILED with return code {result.returncode}")
            return False, f"Exit code: {result.returncode}"
        
        print("\nSUCCESS")
        return True, ""
        
    except Exception as e:
        print(f"\nEXCEPTION: {e}")
        return False, str(e)


def train_task(task: TaskDefinition) -> TrainResult:
    """Train downstream on a single task."""
    # Build run name: model-task-ui_mask
    run_name = f"probe-{MODEL_TYPE}-{task.task_id}-{UI_MASK}"
    
    cmd = [
        sys.executable, "main.py",
        "--mode", "train",
        "--task", "downstream",
        f"task.task_id={task.task_id}",
        f"model.encoder.model_type={MODEL_TYPE}",
        f"data.ui_mask={UI_MASK}",
        f"meta.run_name={run_name}",
    ]
    
    # Add stage 1 checkpoint if specified
    if STAGE1_CHECKPOINT is not None:
        cmd.append(f"model.stage1_checkpoint={STAGE1_CHECKPOINT}")
    
    # Add WandB group if specified
    if WANDB_GROUP is not None:
        cmd.append(f"wandb.group={WANDB_GROUP}")
    
    # Add any extra overrides
    cmd.extend(EXTRA_OVERRIDES)
    
    description = f"Downstream training ({MODEL_TYPE}) on {task.task_id} [ui_mask={UI_MASK}]"
    if STAGE1_CHECKPOINT:
        description += f" [stage1: {Path(STAGE1_CHECKPOINT).parent.name}]"
    
    success, error_msg = run_command(cmd, description)
    
    return TrainResult(
        task_id=task.task_id,
        task_name=task.task_name,
        category=task.category,
        success=success,
        error_msg=error_msg if not success else None
    )


def print_summary(results: list[TrainResult]):
    """Print a summary of all training results."""
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    # Group by category
    by_category: dict[str, list[TrainResult]] = {}
    for r in results:
        if r.category not in by_category:
            by_category[r.category] = []
        by_category[r.category].append(r)
    
    total_passed = 0
    total_failed = 0
    failed_tasks: list[TrainResult] = []
    
    for category in sorted(by_category.keys()):
        print(f"\n{category.upper()}:")
        for r in by_category[category]:
            status = "PASS" if r.success else "FAIL"
            symbol = "CHECK" if r.success else "X"
            print(f"  {symbol} {r.task_id}: {status}")
            
            if r.success:
                total_passed += 1
            else:
                total_failed += 1
                failed_tasks.append(r)
    
    print(f"\n{'='*80}")
    print(f"Total: {total_passed} passed, {total_failed} failed")
    print("="*80)
    
    # Print detailed failures
    if failed_tasks:
        print("\nFAILED TASKS DETAILS:")
        print("-"*80)
        for r in failed_tasks:
            print(f"\n{r.task_id} ({r.task_name}):")
            if r.error_msg:
                print(f"  Error: {r.error_msg[:300]}...")
    
    return total_failed == 0


def main():
    """Run downstream training on all tasks sequentially."""
    print("="*80)
    print("X-EGO Downstream Training: All Tasks")
    print(f"Model: {MODEL_TYPE}")
    print(f"UI Mask: {UI_MASK}")
    if STAGE1_CHECKPOINT:
        print(f"Stage 1 Checkpoint: {STAGE1_CHECKPOINT}")
    else:
        print("Stage 1 Checkpoint: None (baseline/off-the-shelf)")
    if WANDB_GROUP:
        print(f"WandB Group: {WANDB_GROUP}")
    print(f"Task definitions: {TASK_DEFINITIONS_PATH}")
    print("="*80)
    
    # Load tasks
    tasks = load_task_definitions()
    
    # Filter to implemented tasks only
    implemented_tasks = [t for t in tasks if t.implemented.lower() == 'yes']
    
    print(f"\nFound {len(implemented_tasks)} implemented tasks to train")
    for t in implemented_tasks:
        print(f"  - {t.task_id} ({t.category})")
    
    results: list[TrainResult] = []
    
    # Find starting index if resuming
    start_idx = 0
    if START_FROM_TASK is not None:
        for idx, t in enumerate(implemented_tasks):
            if t.task_id == START_FROM_TASK:
                start_idx = idx
                print(f"\nResuming from task: {START_FROM_TASK} (index {start_idx + 1}/{len(implemented_tasks)})")
                break
        else:
            print(f"\nWARNING: Task '{START_FROM_TASK}' not found, starting from beginning")
    
    for i, task in enumerate(implemented_tasks[start_idx:], start_idx + 1):
        print(f"\n{'#'*80}")
        print(f"# Task {i}/{len(implemented_tasks)}: {task.task_id}")
        print(f"# Category: {task.category} | ML Form: {task.ml_form}")
        print('#'*80)
        
        result = train_task(task)
        results.append(result)
    
    # Print summary
    all_passed = print_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
