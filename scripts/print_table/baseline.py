#!/usr/bin/env python3
"""
Print a formatted table of baseline experiment results.
Reads test results from all baseline experiments and creates a summary table.

Usage:
    # Auto-detect paths from current directory
    python scripts/print_table/baseline.py
    
    # Specify custom output directory
    python scripts/print_table/baseline.py /path/to/output/exp_baseline
    
    # Use environment variable
    export BASELINE_OUTPUT_DIR=/path/to/output/exp_baseline
    python scripts/print_table/baseline.py

The script will:
1. Scan all experiment folders in exp_baseline
2. Parse folder names to extract task, task_form, and model information
3. Read test_results.json from each experiment
4. Generate formatted tables for coord-gen and multi-label-cls tasks
5. Save the table to baseline_results_table.txt in the output directory
"""
from pathlib import Path
import json
import re
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get base paths from environment
OUTPUT_BASE_PATH = os.getenv("OUTPUT_BASE_PATH")
SRC_BASE_PATH = os.getenv("SRC_BASE_PATH")

# Determine paths based on priority:
# 1. Command line argument
# 2. Environment variable OUTPUT_BASE_PATH
# 3. Auto-detect from current working directory
if len(sys.argv) > 1:
    OUTPUT_DIR = Path(sys.argv[1])
    OUTPUT_TABLE_FILE = OUTPUT_DIR.parent / "baseline_results_table.txt"
elif OUTPUT_BASE_PATH:
    OUTPUT_DIR = Path(OUTPUT_BASE_PATH) / "exp_baseline"
    OUTPUT_TABLE_FILE = Path(OUTPUT_BASE_PATH) / "baseline_results_table.txt"
else:
    # Fallback: auto-detect based on current working directory
    CURRENT_DIR = Path.cwd()
    if 'x-ego' in CURRENT_DIR.parts:
        # Find the x-ego root directory
        parts = CURRENT_DIR.parts
        x_ego_idx = parts.index('x-ego')
        PROJECT_ROOT = Path(*parts[:x_ego_idx + 1])
    else:
        # Default to script's parent directory structure
        PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    OUTPUT_DIR = PROJECT_ROOT / "output" / "exp_baseline"
    OUTPUT_TABLE_FILE = PROJECT_ROOT / "output" / "baseline_results_table.txt"


def parse_folder_name(folder_name):
    """
    Parse folder name to extract task, task_form, and model.
    
    Format: baseline-{task}-{task_form}-{model}-{timestamp}-{hash}
    Example: baseline-en-forecast-cgen-dinov2-251003-053126-16yy
    
    Returns:
        dict with keys: task, task_form, model, full_name
        or None if parsing fails
    """
    # Pattern: baseline-{task}-{task_form}-{model}-{rest}
    pattern = r'^baseline-([^-]+-[^-]+)-([^-]+)-([^-]+)-\d{6}-\d{6}-[a-z0-9]+$'
    match = re.match(pattern, folder_name)
    
    if match:
        return {
            'task': match.group(1),
            'task_form': match.group(2),
            'model': match.group(3),
            'full_name': folder_name
        }
    return None


def find_test_results_files(exp_dir):
    """
    Find test_results.json files in the experiment directory.
    Looks in test_analysis subdirectories for both 'best' and 'last' checkpoints.
    
    Returns:
        dict with keys 'best' and 'last', values are Path objects or None
    """
    test_analysis_dir = exp_dir / "test_analysis"
    results = {'best': None, 'last': None}
    
    if not test_analysis_dir.exists():
        return results
    
    # Try to find both 'best' and 'last' checkpoints
    for checkpoint_type in ['best', 'last']:
        for subdir in test_analysis_dir.iterdir():
            if subdir.is_dir() and checkpoint_type in subdir.name:
                results_file = subdir / "test_results.json"
                if results_file.exists():
                    results[checkpoint_type] = results_file
                    break
    
    return results


def load_test_results(results_file):
    """Load test results from JSON file."""
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {results_file}: {e}")
        return None


def extract_cgen_metrics(results):
    """Extract key metrics from coord-gen results."""
    metrics = {
        'overall_mae': results.get('overall_mae', None),
        'overall_mse': results.get('overall_mse', None),
        'chamfer_dist': results.get('geometric_distances', {}).get('chamfer_distance_mean', None),
        'wasserstein_dist': results.get('geometric_distances', {}).get('wasserstein_distance_mean', None),
        'num_samples': results.get('num_samples', None),
    }
    
    # Team-specific metrics
    team_metrics = results.get('team_specific_metrics', {})
    if 'CT' in team_metrics:
        metrics['ct_mae'] = team_metrics['CT'].get('mae', None)
        metrics['ct_chamfer'] = team_metrics['CT'].get('geometric_distances', {}).get('chamfer_distance_mean', None)
    if 'T' in team_metrics:
        metrics['t_mae'] = team_metrics['T'].get('mae', None)
        metrics['t_chamfer'] = team_metrics['T'].get('geometric_distances', {}).get('chamfer_distance_mean', None)
    
    return metrics


def extract_mlcls_metrics(results):
    """Extract key metrics from multi-label-cls results."""
    metrics = {
        'hamming_loss': results.get('hamming_loss', None),
        'hamming_accuracy': results.get('hamming_accuracy', None),
        'micro_f1': results.get('micro_f1', None),
        'macro_f1': results.get('macro_f1', None),
        'subset_accuracy': results.get('subset_accuracy', None),
        'num_samples': results.get('num_samples', None),
    }
    
    # Team-specific metrics
    team_metrics = results.get('team_specific_metrics', {})
    if 'CT' in team_metrics:
        metrics['ct_hamming_acc'] = team_metrics['CT'].get('hamming_accuracy', None)
        metrics['ct_micro_f1'] = team_metrics['CT'].get('micro_f1', None)
    if 'T' in team_metrics:
        metrics['t_hamming_acc'] = team_metrics['T'].get('hamming_accuracy', None)
        metrics['t_micro_f1'] = team_metrics['T'].get('micro_f1', None)
    
    return metrics


def collect_all_results():
    """Collect results from all baseline experiments."""
    results_data = []
    
    if not OUTPUT_DIR.exists():
        print(f"Error: Output directory not found: {OUTPUT_DIR}")
        return results_data
    
    # Iterate through all experiment directories
    for exp_dir in sorted(OUTPUT_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        # Parse folder name
        parsed = parse_folder_name(exp_dir.name)
        if not parsed:
            print(f"Skipping (couldn't parse): {exp_dir.name}")
            continue
        
        # Find test results files (both best and last)
        results_files = find_test_results_files(exp_dir)
        
        if not results_files['best'] and not results_files['last']:
            print(f"Warning: No test_results.json found in {exp_dir.name}")
            continue
        
        # Process both checkpoint types
        for checkpoint_type in ['best', 'last']:
            results_file = results_files[checkpoint_type]
            if not results_file:
                continue
            
            # Load results
            results = load_test_results(results_file)
            if not results:
                continue
            
            # Extract metrics based on task form
            task_form = results.get('task_form', parsed['task_form'])
            if task_form in ['coord-gen', 'cgen']:
                metrics = extract_cgen_metrics(results)
                metrics['task_form'] = 'cgen'
            elif task_form in ['multi-label-cls', 'mlcls']:
                metrics = extract_mlcls_metrics(results)
                metrics['task_form'] = 'mlcls'
            else:
                print(f"Unknown task form: {task_form} in {exp_dir.name}")
                continue
            
            # Combine parsed info with metrics
            result_entry = {
                **parsed,
                **metrics,
                'checkpoint': checkpoint_type,
                'results_file': str(results_file)
            }
            results_data.append(result_entry)
    
    return results_data


def format_value(value, precision=4):
    """Format a numeric value for display."""
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"


def create_cgen_table(results_data, checkpoint_type):
    """Create formatted table for coord-gen results filtered by checkpoint type."""
    # Filter for coord-gen results with specific checkpoint type
    cgen_results = [r for r in results_data if r['task_form'] == 'cgen' and r.get('checkpoint') == checkpoint_type]
    
    if not cgen_results:
        return f"No coord-gen results found for checkpoint: {checkpoint_type}.\n"
    
    # Group by task
    tasks = sorted(set(r['task'] for r in cgen_results))
    models = sorted(set(r['model'] for r in cgen_results))
    
    lines = []
    lines.append("=" * 127)
    lines.append(f"COORDINATE GENERATION (CGEN) RESULTS - {checkpoint_type.upper()} CHECKPOINT")
    lines.append("=" * 127)
    lines.append("")
    
    for task in tasks:
        task_results = [r for r in cgen_results if r['task'] == task]
        if not task_results:
            continue
        
        lines.append(f"Task: {task}")
        lines.append("-" * 127)
        lines.append(f"{'Model':<12} {'MAE':>8} {'MSE':>8} {'Chamfer':>10} {'Wasser':>10} {'CT-MAE':>8} {'T-MAE':>8} {'Samples':>8}")
        lines.append("-" * 127)
        
        for model in models:
            model_results = [r for r in task_results if r['model'] == model]
            if not model_results:
                continue
            
            # Should only be one result per model now
            for r in model_results:
                lines.append(
                    f"{model:<12} "
                    f"{format_value(r.get('overall_mae')):>8} "
                    f"{format_value(r.get('overall_mse')):>8} "
                    f"{format_value(r.get('chamfer_dist')):>10} "
                    f"{format_value(r.get('wasserstein_dist')):>10} "
                    f"{format_value(r.get('ct_mae')):>8} "
                    f"{format_value(r.get('t_mae')):>8} "
                    f"{r.get('num_samples', 'N/A'):>8}"
                )
        
        lines.append("")
    
    return "\n".join(lines)


def create_mlcls_table(results_data, checkpoint_type):
    """Create formatted table for multi-label-cls results filtered by checkpoint type."""
    # Filter for multi-label-cls results with specific checkpoint type
    mlcls_results = [r for r in results_data if r['task_form'] == 'mlcls' and r.get('checkpoint') == checkpoint_type]
    
    if not mlcls_results:
        return f"No multi-label-cls results found for checkpoint: {checkpoint_type}.\n"
    
    # Group by task
    tasks = sorted(set(r['task'] for r in mlcls_results))
    models = sorted(set(r['model'] for r in mlcls_results))
    
    lines = []
    lines.append("=" * 137)
    lines.append(f"MULTI-LABEL CLASSIFICATION (MLCLS) RESULTS - {checkpoint_type.upper()} CHECKPOINT")
    lines.append("=" * 137)
    lines.append("")
    
    for task in tasks:
        task_results = [r for r in mlcls_results if r['task'] == task]
        if not task_results:
            continue
        
        lines.append(f"Task: {task}")
        lines.append("-" * 137)
        lines.append(f"{'Model':<12} {'H-Loss':>8} {'H-Acc':>8} {'Subset-Acc':>10} {'Micro-F1':>10} {'Macro-F1':>10} {'CT-HAcc':>8} {'T-HAcc':>8} {'Samples':>8}")
        lines.append("-" * 137)
        
        for model in models:
            model_results = [r for r in task_results if r['model'] == model]
            if not model_results:
                continue
            
            # Should only be one result per model now
            for r in model_results:
                lines.append(
                    f"{model:<12} "
                    f"{format_value(r.get('hamming_loss')):>8} "
                    f"{format_value(r.get('hamming_accuracy')):>8} "
                    f"{format_value(r.get('subset_accuracy')):>10} "
                    f"{format_value(r.get('micro_f1')):>10} "
                    f"{format_value(r.get('macro_f1')):>10} "
                    f"{format_value(r.get('ct_hamming_acc')):>8} "
                    f"{format_value(r.get('t_hamming_acc')):>8} "
                    f"{r.get('num_samples', 'N/A'):>8}"
                )
        
        lines.append("")
    
    return "\n".join(lines)


def main():
    """Main function to generate and save results table."""
    print("=" * 80)
    print("BASELINE EXPERIMENT RESULTS ANALYZER")
    print("=" * 80)
    
    # Show which path source is being used
    if len(sys.argv) > 1:
        print("Path source: Command line argument")
    elif OUTPUT_BASE_PATH:
        print(f"Path source: .env file (OUTPUT_BASE_PATH={OUTPUT_BASE_PATH})")
    else:
        print("Path source: Auto-detected from current directory")
    
    print(f"Looking for results in: {OUTPUT_DIR}")
    print(f"Will save table to: {OUTPUT_TABLE_FILE}")
    print()
    
    print("Collecting baseline experiment results...")
    results_data = collect_all_results()
    
    if not results_data:
        print("No results found.")
        return
    
    print(f"Found {len(results_data)} experiments with results.")
    
    # Create tables for both best and last checkpoints
    cgen_table_best = create_cgen_table(results_data, 'best')
    mlcls_table_best = create_mlcls_table(results_data, 'best')
    
    cgen_table_last = create_cgen_table(results_data, 'last')
    mlcls_table_last = create_mlcls_table(results_data, 'last')
    
    # Combine and save
    full_output = f"""
BASELINE EXPERIMENT RESULTS SUMMARY
Generated from: {OUTPUT_DIR}

{cgen_table_best}

{mlcls_table_best}

{cgen_table_last}

{mlcls_table_last}

Notes:
- MAE/MSE: Mean Absolute/Squared Error
- Chamfer/Wasser: Chamfer/Wasserstein Distance
- H-Loss/H-Acc: Hamming Loss/Accuracy
- Subset-Acc: Subset Accuracy (exact match of all labels)
- CT/T: Counter-Terrorist/Terrorist team metrics
- Best checkpoint: Model with best validation performance
- Last checkpoint: Model from final training epoch
"""
    
    # Print to console
    print(full_output)
    
    # Save to file
    OUTPUT_TABLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TABLE_FILE, 'w') as f:
        f.write(full_output)
    
    print(f"\nResults table saved to: {OUTPUT_TABLE_FILE}")


if __name__ == "__main__":
    main()

