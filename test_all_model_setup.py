#!/usr/bin/env python3
"""
Test script to verify the full pipeline works for all model setups.

Tests three settings with task self_location_0s:
1. Contrastive only (Stage 1)
2. Baseline downstream (off-the-shelf model)
3. Downstream with saved contrastive checkpoint (Stage 2)

All tests run in dev mode for quick verification.
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

from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# Available model types from cheatsheet
# MODEL_TYPES = ["siglip2", "dinov2", "dinov3", "clip", "vivit", "videomae", "vjepa2"]
# MODEL_TYPES = ["siglip2", "dinov2", "vjepa2"]
MODEL_TYPES = ["dinov2"]
TASK_ID = "self_location_0s"


@dataclass
class TestResult:
    name: str
    model_type: str
    success: bool
    error_msg: Optional[str] = None
    checkpoint_path: Optional[str] = None


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
            encoding='utf-8',
            errors='replace',
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


def find_checkpoint(stdout: str, stderr: str, model_type: str) -> Optional[str]:
    """Extract checkpoint path from training output, filtering by model type."""
    combined = (stdout or "") + (stderr or "")
    
    # First, try to find the experiment directory from the output
    experiment_dir = None
    for line in combined.split('\n'):
        if 'Created experiment directory' in line or 'experiment directory:' in line.lower():
            # Try to extract the path - look for the path after the colon or equals sign
            if ':' in line:
                # Extract everything after the colon
                path_part = line.split(':', 1)[1].strip()
                # Remove any trailing punctuation
                path_part = path_part.strip("'\":,.")
                potential_path = Path(path_part)
                if potential_path.exists() and potential_path.is_dir():
                    experiment_dir = potential_path
                    break
            # Also try splitting by spaces and looking for paths
            parts = line.split()
            for part in parts:
                if 'output' in part.lower() or 'contrastive' in part.lower():
                    # Check if it's a valid path
                    potential_path = Path(part.strip("'\":,."))
                    if potential_path.exists() and potential_path.is_dir():
                        experiment_dir = potential_path
                        break
                    # Also check if it contains the full path
                    if '\\' in part or '/' in part:
                        potential_path = Path(part.strip("'\":,."))
                        if potential_path.exists() and potential_path.is_dir():
                            experiment_dir = potential_path
                            break
    
    # Look for checkpoint path patterns in output
    for line in combined.split('\n'):
        if '.ckpt' in line and ('Saving' in line or 'checkpoint' in line.lower()):
            # Try to extract the path
            parts = line.split()
            for part in parts:
                if '.ckpt' in part:
                    # Clean up the path
                    path = part.strip("'\"")
                    if Path(path).exists():
                        return path
    
    # If we found the experiment directory, look for checkpoints there
    if experiment_dir:
        checkpoint_dir = experiment_dir / "checkpoint"
        if checkpoint_dir.exists():
            ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
            if ckpt_files:
                # Return the most recently modified checkpoint from this experiment
                return str(max(ckpt_files, key=lambda p: p.stat().st_mtime))
    
    # Alternative: look in output directory for checkpoints matching the model type
    output_dir = Path(__file__).parent / "output"
    if output_dir.exists():
        # Look for experiment directories that match the model type
        # Experiment names are like: contrastive-{model_type}-{timestamp}-{hash}
        matching_dirs = []
        for exp_dir in output_dir.iterdir():
            if exp_dir.is_dir() and f'contrastive-{model_type}' in exp_dir.name:
                matching_dirs.append(exp_dir)
        
        # If we found matching directories, look for checkpoints in the most recent one
        if matching_dirs:
            # Sort by modification time, most recent first
            matching_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            most_recent_exp = matching_dirs[0]
            checkpoint_dir = most_recent_exp / "checkpoint"
            if checkpoint_dir.exists():
                ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
                if ckpt_files:
                    # Return the most recently modified checkpoint
                    return str(max(ckpt_files, key=lambda p: p.stat().st_mtime))
        
        # Fallback: look for any checkpoint, but prefer ones matching model type
        ckpt_files = list(output_dir.rglob("*.ckpt"))
        if ckpt_files:
            # Filter by model type in parent directory name
            matching_ckpts = [c for c in ckpt_files if f'contrastive-{model_type}' in str(c)]
            if matching_ckpts:
                return str(max(matching_ckpts, key=lambda p: p.stat().st_mtime))
            # If no matching checkpoints, return the most recent (but warn)
            return str(max(ckpt_files, key=lambda p: p.stat().st_mtime))
    
    return None


def test_contrastive(model_type: str) -> TestResult:
    """Test contrastive learning (Stage 1)."""
    cmd = [
        sys.executable, "main.py",
        "--mode", "dev",
        "--task", "contrastive",
        f"model.encoder.model_type={model_type}"
    ]
    
    success, stdout, stderr = run_command(
        cmd, 
        f"Contrastive training with {model_type}"
    )
    
    checkpoint_path = find_checkpoint(stdout, stderr, model_type) if success else None
    
    return TestResult(
        name="contrastive",
        model_type=model_type,
        success=success,
        error_msg=stderr[-500:] if not success else None,
        checkpoint_path=checkpoint_path
    )


def test_baseline_downstream(model_type: str) -> TestResult:
    """Test baseline downstream (off-the-shelf encoder)."""
    cmd = [
        sys.executable, "main.py",
        "--mode", "dev",
        "--task", "downstream",
        f"task.task_id={TASK_ID}",
        f"model.encoder.model_type={model_type}"
    ]
    
    success, stdout, stderr = run_command(
        cmd,
        f"Baseline downstream ({model_type}) on {TASK_ID}"
    )
    
    return TestResult(
        name="baseline_downstream",
        model_type=model_type,
        success=success,
        error_msg=stderr[-500:] if not success else None
    )


def test_downstream_with_checkpoint(model_type: str, checkpoint_path: str) -> TestResult:
    """Test downstream with pretrained contrastive checkpoint."""
    cmd = [
        sys.executable, "main.py",
        "--mode", "dev",
        "--task", "downstream",
        f"task.task_id={TASK_ID}",
        f"model.encoder.model_type={model_type}",
        f"model.stage1_checkpoint={checkpoint_path}"
    ]
    
    success, stdout, stderr = run_command(
        cmd,
        f"Downstream with checkpoint ({model_type}) on {TASK_ID}"
    )
    
    return TestResult(
        name="downstream_with_checkpoint",
        model_type=model_type,
        success=success,
        error_msg=stderr[-500:] if not success else None
    )


def print_summary(results: list[TestResult]):
    """Print a summary of all test results."""
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    # Group by model type
    by_model = {}
    for r in results:
        if r.model_type not in by_model:
            by_model[r.model_type] = []
        by_model[r.model_type].append(r)
    
    total_passed = 0
    total_failed = 0
    
    for model_type in MODEL_TYPES:
        if model_type not in by_model:
            continue
            
        print(f"\n{model_type}:")
        for r in by_model[model_type]:
            status = "PASS" if r.success else "FAIL"
            symbol = "CHECK" if r.success else "✗"
            print(f"  {symbol} {r.name}: {status}")
            
            if r.success:
                total_passed += 1
            else:
                total_failed += 1
                if r.error_msg:
                    # Print truncated error
                    error_preview = r.error_msg[:200].replace('\n', ' ')
                    print(f"      Error: {error_preview}...")
    
    print(f"\n{'='*80}")
    print(f"Total: {total_passed} passed, {total_failed} failed")
    print("="*80)
    
    return total_failed == 0


def main():
    """Run all tests sequentially."""
    print("="*80)
    print("X-EGO Pipeline Test: All Model Setups")
    print(f"Task: {TASK_ID}")
    print(f"Models: {', '.join(MODEL_TYPES)}")
    print("="*80)
    
    results: list[TestResult] = []
    
    for model_type in MODEL_TYPES:
        print(f"\n{'#'*80}")
        print(f"# Testing model: {model_type}")
        print('#'*80)
        
        # 1. Contrastive only
        contrastive_result = test_contrastive(model_type)
        results.append(contrastive_result)
        
        # 2. Baseline downstream
        baseline_result = test_baseline_downstream(model_type)
        results.append(baseline_result)
        
        # 3. Downstream with checkpoint (only if contrastive succeeded and produced checkpoint)
        if contrastive_result.success and contrastive_result.checkpoint_path:
            checkpoint_result = test_downstream_with_checkpoint(
                model_type, 
                contrastive_result.checkpoint_path
            )
            results.append(checkpoint_result)
        else:
            # Record as skipped/failed
            results.append(TestResult(
                name="downstream_with_checkpoint",
                model_type=model_type,
                success=False,
                error_msg="Skipped: No checkpoint from contrastive training"
            ))
    
    # Print summary
    all_passed = print_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
