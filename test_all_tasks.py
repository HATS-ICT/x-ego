#!/usr/bin/env python3
"""
Test script to verify baseline downstream training works for all tasks.

Runs baseline downstream (siglip, off-the-shelf) on all tasks defined in
data/labels/task_definitions.csv in dev mode.
"""

import subprocess
import sys

# Fix Windows console encoding to support Unicode characters
if sys.platform == 'win32':
    # Set UTF-8 encoding for stdout and stderr
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    # Also set environment variable for subprocesses
    import os
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


MODEL_TYPE = "siglip2"
TASK_DEFINITIONS_PATH = Path(__file__).parent / "data" / "labels" / "task_definitions.csv"


@dataclass
class TaskDefinition:
    task_id: str
    task_name: str
    category: str
    ml_form: str
    implemented: str


@dataclass
class TestResult:
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


def run_command(cmd: list[str], description: str) -> tuple[bool, str, str]:
    """Run a command and return (success, stdout, stderr)."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
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


def test_task(task: TaskDefinition) -> TestResult:
    """Test baseline downstream on a single task."""
    cmd = [
        sys.executable, "main.py",
        "--mode", "dev",
        "--task", "downstream",
        f"task.task_id={task.task_id}",
        f"model.encoder.model_type={MODEL_TYPE}"
    ]
    
    success, stdout, stderr = run_command(
        cmd,
        f"Baseline downstream ({MODEL_TYPE}) on {task.task_id}"
    )
    
    return TestResult(
        task_id=task.task_id,
        task_name=task.task_name,
        category=task.category,
        success=success,
        error_msg=stderr[-500:] if not success else None
    )


def print_summary(results: list[TestResult]):
    """Print a summary of all test results."""
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    # Group by category
    by_category: dict[str, list[TestResult]] = {}
    for r in results:
        if r.category not in by_category:
            by_category[r.category] = []
        by_category[r.category].append(r)
    
    total_passed = 0
    total_failed = 0
    failed_tasks: list[TestResult] = []
    
    for category in sorted(by_category.keys()):
        print(f"\n{category.upper()}:")
        for r in by_category[category]:
            status = "PASS" if r.success else "FAIL"
            symbol = "CHECK" if r.success else "✗"
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
                # Print truncated error
                print(f"  Error: {r.error_msg[:300]}...")
    
    return total_failed == 0


def main():
    """Run all task tests sequentially."""
    print("="*80)
    print("X-EGO Pipeline Test: All Tasks")
    print(f"Model: {MODEL_TYPE} (baseline)")
    print(f"Task definitions: {TASK_DEFINITIONS_PATH}")
    print("="*80)
    
    # Load tasks
    tasks = load_task_definitions()
    
    # Filter to implemented tasks only
    implemented_tasks = [t for t in tasks if t.implemented.lower() == 'yes']
    
    print(f"\nFound {len(implemented_tasks)} implemented tasks to test")
    for t in implemented_tasks:
        print(f"  - {t.task_id} ({t.category})")
    
    results: list[TestResult] = []
    
    for i, task in enumerate(implemented_tasks, 1):
        print(f"\n{'#'*80}")
        print(f"# Task {i}/{len(implemented_tasks)}: {task.task_id}")
        print(f"# Category: {task.category} | ML Form: {task.ml_form}")
        print('#'*80)
        
        result = test_task(task)
        results.append(result)
    
    # Print summary
    all_passed = print_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
