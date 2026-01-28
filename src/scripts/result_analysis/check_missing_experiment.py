#!/usr/bin/env python3
"""
Check Missing Experiments

Identifies missing experiments and counts repeats for each experiment setting.
An experiment setting is defined by: (model_name, task_id, init_type)
where init_type is 'baseline' or 'finetuned' based on model.stage1_checkpoint in hparam.yaml.
"""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table
from rich import box


# Constants
MODELS = ['siglip2', 'clip', 'dinov2', 'vjepa2']
UI_MASKS = ['all', 'none', 'minimap_only']


def load_task_ids(task_definitions_path: Path) -> List[str]:
    """Load task IDs from task_definitions.csv."""
    df = pd.read_csv(task_definitions_path)
    return df['task_id'].tolist()


def read_hparam(exp_dir: Path) -> Optional[Dict]:
    """Read hparam.yaml from experiment directory."""
    hparam_path = exp_dir / 'hparam.yaml'
    if hparam_path.exists():
        with open(hparam_path, 'r') as f:
            return yaml.safe_load(f)
    return None


def is_finetuned(hparam: Optional[Dict]) -> bool:
    """Determine if experiment is finetuned based on hparam."""
    if hparam is None:
        return False
    model_cfg = hparam.get('model', {})
    stage1_checkpoint = model_cfg.get('stage1_checkpoint')
    return stage1_checkpoint is not None


def get_ui_mask(hparam: Optional[Dict]) -> str:
    """Get ui_mask setting from hparam."""
    if hparam is None:
        return 'unknown'
    data_cfg = hparam.get('data', {})
    return data_cfg.get('ui_mask', 'unknown')


def get_model_type(hparam: Optional[Dict]) -> str:
    """Get model type from hparam."""
    if hparam is None:
        return 'unknown'
    model_cfg = hparam.get('model', {})
    encoder_cfg = model_cfg.get('encoder', {})
    return encoder_cfg.get('model_type', 'unknown')


def get_task_id(hparam: Optional[Dict]) -> str:
    """Get task_id from hparam."""
    if hparam is None:
        return 'unknown'
    task_cfg = hparam.get('task', {})
    return task_cfg.get('task_id', 'unknown')


def is_new_format(exp_name: str) -> bool:
    """Check if experiment name matches the new format."""
    parts = exp_name.split('-')
    if len(parts) < 5:
        return False
    return parts[1] in MODELS


def check_experiment_complete(exp_dir: Path) -> Tuple[bool, bool]:
    """
    Check if experiment has required result files.
    
    Returns:
        Tuple of (has_best, has_last)
    """
    has_best = (exp_dir / 'test_results_best.json').exists()
    has_last = (exp_dir / 'test_results_last.json').exists()
    return has_best, has_last


def collect_experiments(
    output_dir: Path,
    ui_mask_filter: str
) -> Dict[Tuple[str, str, str], List[Dict]]:
    """
    Collect all experiments and organize by setting.
    
    Args:
        output_dir: Path to output directory
        ui_mask_filter: UI mask to filter ('all', 'none', 'minimap_only')
    
    Returns:
        Dict mapping (model, task_id, init_type) to list of experiment info dicts
    """
    experiments = defaultdict(list)
    
    for exp_dir in output_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # Only process folders starting with 'probe-'
        if not exp_dir.name.startswith('probe-'):
            continue
        
        # Only process new format
        if not is_new_format(exp_dir.name):
            continue
        
        # Read hparam to get experiment details
        hparam = read_hparam(exp_dir)
        if hparam is None:
            continue
        
        # Get experiment settings from hparam (more reliable than parsing folder name)
        model = get_model_type(hparam)
        task_id = get_task_id(hparam)
        ui_mask = get_ui_mask(hparam)
        init_type = 'finetuned' if is_finetuned(hparam) else 'baseline'
        
        # Filter by ui_mask
        if ui_mask != ui_mask_filter:
            continue
        
        # Check completeness
        has_best, has_last = check_experiment_complete(exp_dir)
        
        exp_info = {
            'folder': exp_dir.name,
            'model': model,
            'task_id': task_id,
            'ui_mask': ui_mask,
            'init_type': init_type,
            'has_best': has_best,
            'has_last': has_last,
            'complete': has_best and has_last,
        }
        
        key = (model, task_id, init_type)
        experiments[key].append(exp_info)
    
    return experiments


def print_experiment_status(
    experiments: Dict[Tuple[str, str, str], List[Dict]],
    task_ids: List[str],
    ui_mask: str,
    console: Console
) -> Dict[Tuple[str, str, str], List[Dict]]:
    """Print experiment status tables.
    
    Returns:
        Dict of incomplete experiments for potential deletion.
    """
    
    # Build expected experiments
    expected = set()
    for model in MODELS:
        for task_id in task_ids:
            for init_type in ['baseline', 'finetuned']:
                expected.add((model, task_id, init_type))
    
    # Categorize experiments
    complete_experiments = {}  # key -> count of complete experiments
    incomplete_experiments = {}  # key -> list of incomplete exp info
    missing_experiments = set()  # keys with no experiments at all
    
    for key in expected:
        if key not in experiments:
            missing_experiments.add(key)
        else:
            exp_list = experiments[key]
            complete_count = sum(1 for e in exp_list if e['complete'])
            incomplete_list = [e for e in exp_list if not e['complete']]
            
            complete_experiments[key] = complete_count
            if incomplete_list:
                incomplete_experiments[key] = incomplete_list
    
    # Print summary
    console.print()
    console.rule(f"[bold blue]Experiment Status Summary (ui_mask={ui_mask})[/bold blue]")
    console.print()
    
    total_expected = len(expected)
    total_with_complete = sum(1 for k, v in complete_experiments.items() if v > 0)
    total_missing = len(missing_experiments)
    total_incomplete = len(incomplete_experiments)
    
    console.print(f"  Total expected experiment settings: {total_expected}")
    console.print(f"  Settings with at least 1 complete: {total_with_complete}")
    console.print(f"  Settings entirely missing: {total_missing}")
    console.print(f"  Settings with incomplete runs: {total_incomplete}")
    console.print()
    
    # Print repeat counts table
    console.rule("[bold green]Experiment Repeat Counts[/bold green]")
    console.print()
    
    # Create table for repeat counts
    table = Table(
        title=f"Complete Experiment Counts (ui_mask={ui_mask})",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )
    
    table.add_column("Model", style="cyan")
    table.add_column("Task ID", style="green")
    table.add_column("Baseline", justify="center")
    table.add_column("Finetuned", justify="center")
    
    # Group by model and task
    for model in MODELS:
        for task_id in task_ids:
            baseline_key = (model, task_id, 'baseline')
            finetuned_key = (model, task_id, 'finetuned')
            
            baseline_count = complete_experiments.get(baseline_key, 0)
            finetuned_count = complete_experiments.get(finetuned_key, 0)
            
            # Color code based on count
            def format_count(count):
                if count == 0:
                    return "[red]0[/red]"
                elif count == 1:
                    return "[yellow]1[/yellow]"
                else:
                    return f"[green]{count}[/green]"
            
            table.add_row(
                model,
                task_id,
                format_count(baseline_count),
                format_count(finetuned_count)
            )
    
    console.print(table)
    console.print()
    
    # Print missing experiments
    if missing_experiments:
        console.rule("[bold red]Missing Experiments (No Folder Found)[/bold red]")
        console.print()
        
        missing_table = Table(
            title="Missing Experiment Settings",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        missing_table.add_column("Model", style="cyan")
        missing_table.add_column("Task ID", style="green")
        missing_table.add_column("Init Type", style="yellow")
        
        for model, task_id, init_type in sorted(missing_experiments):
            missing_table.add_row(model, task_id, init_type)
        
        console.print(missing_table)
        console.print()
        console.print(f"[red]Total missing: {len(missing_experiments)}[/red]")
        console.print()
    
    # Print incomplete experiments (has folder but missing result files)
    if not incomplete_experiments:
        return {}
    
    console.rule("[bold yellow]Incomplete Experiments (Missing Result Files)[/bold yellow]")
    console.print()
    
    incomplete_table = Table(
        title="Incomplete Experiments",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )
    
    incomplete_table.add_column("Model", style="cyan")
    incomplete_table.add_column("Task ID", style="green")
    incomplete_table.add_column("Init Type", style="yellow")
    incomplete_table.add_column("Folder", style="dim")
    incomplete_table.add_column("Best", justify="center")
    incomplete_table.add_column("Last", justify="center")
    
    for key in sorted(incomplete_experiments.keys()):
        for exp in incomplete_experiments[key]:
            model, task_id, init_type = key
            best_status = "[green]Y[/green]" if exp['has_best'] else "[red]N[/red]"
            last_status = "[green]Y[/green]" if exp['has_last'] else "[red]N[/red]"
            
            incomplete_table.add_row(
                model,
                task_id,
                init_type,
                exp['folder'],
                best_status,
                last_status
            )
    
    console.print(incomplete_table)
    console.print()
    
    total_incomplete_runs = sum(len(v) for v in incomplete_experiments.values())
    console.print(f"[yellow]Total incomplete runs: {total_incomplete_runs}[/yellow]")
    console.print()
    
    return incomplete_experiments


def prompt_delete_incomplete(
    incomplete_experiments: Dict[Tuple[str, str, str], List[Dict]],
    output_dir: Path,
    console: Console
) -> None:
    """Prompt user to delete incomplete experiment folders."""
    if not incomplete_experiments:
        return
    
    # Collect all incomplete folders
    incomplete_folders = []
    for exp_list in incomplete_experiments.values():
        for exp in exp_list:
            incomplete_folders.append(exp['folder'])
    
    console.print()
    console.rule("[bold red]Delete Incomplete Experiments?[/bold red]")
    console.print()
    console.print(f"Found [yellow]{len(incomplete_folders)}[/yellow] incomplete experiment folders.")
    console.print()
    
    response = console.input("[bold]Do you want to delete these folders? (y/N): [/bold]")
    
    if response.lower() in ['y', 'yes']:
        console.print()
        deleted_count = 0
        for folder_name in incomplete_folders:
            folder_path = output_dir / folder_name
            if folder_path.exists():
                try:
                    shutil.rmtree(folder_path)
                    console.print(f"  [red]Deleted:[/red] {folder_name}")
                    deleted_count += 1
                except Exception as e:
                    console.print(f"  [red]Failed to delete {folder_name}: {e}[/red]")
        console.print()
        console.print(f"[green]Successfully deleted {deleted_count} folders.[/green]")
    else:
        console.print()
        console.print("[dim]No folders deleted.[/dim]")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Check for missing experiments and count repeats.'
    )
    parser.add_argument(
        '--ui-mask', '-u',
        type=str,
        default='all',
        choices=['all', 'minimap_only', 'none'],
        help='UI mask setting to filter (default: all)'
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    console = Console()
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent.parent
    output_dir = project_root / 'output'
    task_definitions_path = project_root / 'data' / 'labels' / 'task_definitions.csv'
    
    # Load task IDs
    task_ids = load_task_ids(task_definitions_path)
    console.print(f"Loaded {len(task_ids)} task IDs from task_definitions.csv")
    
    # Collect experiments
    experiments = collect_experiments(output_dir, args.ui_mask)
    console.print(f"Found {len(experiments)} unique experiment settings with folders")
    
    # Print status
    incomplete_experiments = print_experiment_status(experiments, task_ids, args.ui_mask, console)
    
    # Prompt to delete incomplete experiments
    prompt_delete_incomplete(incomplete_experiments, output_dir, console)


if __name__ == '__main__':
    main()
