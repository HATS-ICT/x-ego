#!/usr/bin/env python3
"""
Evaluate all existing downstream experiments in the output folder.

Runs test evaluation on all experiment folders that don't already have test results.
Skips 'pre-icml' folder and any experiments that already have test_results_*.json files.
"""

import subprocess
import sys
from pathlib import Path

# Fix Windows console encoding to support Unicode characters
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    import os
    os.environ['PYTHONIOENCODING'] = 'utf-8'


# =============================================================================
# CONFIGURATION
# =============================================================================

# Folders to exclude from evaluation
EXCLUDE_FOLDERS = {'pre-icml'}

# Output directory containing experiments
OUTPUT_DIR = Path(__file__).parent / 'output'

# =============================================================================


def has_test_results(exp_dir: Path) -> bool:
    """Check if experiment already has test results."""
    # Check for any test_results_*.json files
    test_results = list(exp_dir.glob('test_results_*.json'))
    return len(test_results) > 0


def is_valid_experiment(exp_dir: Path) -> bool:
    """Check if directory is a valid downstream experiment."""
    # Must have hparam.yaml
    if not (exp_dir / 'hparam.yaml').exists():
        return False
    
    # Must have checkpoint directory with at least one .ckpt file
    checkpoint_dir = exp_dir / 'checkpoint'
    if not checkpoint_dir.exists():
        return False
    
    ckpt_files = list(checkpoint_dir.glob('*.ckpt'))
    if not ckpt_files:
        return False
    
    return True


def get_experiments_to_evaluate() -> list[Path]:
    """Get list of experiment directories that need evaluation."""
    experiments = []
    
    if not OUTPUT_DIR.exists():
        print(f"Output directory not found: {OUTPUT_DIR}")
        return experiments
    
    for item in OUTPUT_DIR.iterdir():
        # Skip non-directories
        if not item.is_dir():
            continue
        
        # Skip excluded folders
        if item.name in EXCLUDE_FOLDERS:
            print(f"Skipping excluded folder: {item.name}")
            continue
        
        # Skip if not a valid experiment
        if not is_valid_experiment(item):
            print(f"Skipping invalid experiment: {item.name}")
            continue
        
        # Skip if already has test results
        if has_test_results(item):
            print(f"Skipping (already evaluated): {item.name}")
            continue
        
        experiments.append(item)
    
    return experiments


def evaluate_experiment(exp_name: str) -> tuple[bool, str]:
    """Run test evaluation on a single experiment."""
    cmd = [
        sys.executable, 'main.py',
        '--mode', 'test',
        '--task', 'downstream',
        f'meta.resume_exp={exp_name}',
    ]
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {exp_name}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    sys.stdout.flush()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent
        )
        
        if result.returncode != 0:
            return False, f"Exit code: {result.returncode}"
        
        return True, ""
    
    except Exception as e:
        return False, str(e)


def main():
    """Evaluate all downstream experiments that need evaluation."""
    print("="*80)
    print("X-EGO: Evaluate Existing Downstream Experiments")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Excluded folders: {EXCLUDE_FOLDERS}")
    print("="*80)
    
    # Get experiments to evaluate
    experiments = get_experiments_to_evaluate()
    
    if not experiments:
        print("\nNo experiments need evaluation.")
        return
    
    # Sort by name for consistent ordering
    experiments.sort(key=lambda x: x.name)
    
    print(f"\nFound {len(experiments)} experiments to evaluate:")
    for exp in experiments:
        print(f"  - {exp.name}")
    
    # Evaluate each experiment
    results = []
    for i, exp_dir in enumerate(experiments, 1):
        print(f"\n{'#'*80}")
        print(f"# Experiment {i}/{len(experiments)}: {exp_dir.name}")
        print('#'*80)
        
        success, error_msg = evaluate_experiment(exp_dir.name)
        results.append((exp_dir.name, success, error_msg))
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success, _ in results if success)
    failed = sum(1 for _, success, _ in results if not success)
    
    for exp_name, success, error_msg in results:
        status = "PASS" if success else "FAIL"
        print(f"  {status}: {exp_name}")
        if not success and error_msg:
            print(f"         Error: {error_msg}")
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    print("="*80)
    
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
