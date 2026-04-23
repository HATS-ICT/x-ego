#!/usr/bin/env python3
"""
Plot Pareto - Epoch Probe Results Visualization

Creates a grid of line plots showing how metrics change over epochs
for different tasks from epoch_probe experiments.

Grid layout:
- Rows: 3 metrics (acc, f1, auroc)
- Columns: k unique tasks, ordered by prefix (self, teammate, enemy, global)
"""

import json
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10


# Task prefix order (self first, then teammate, enemy, global)
TASK_PREFIX_ORDER = ['self', 'teammate', 'enemy', 'global']

# Metrics to plot (rows)
METRICS = ['acc', 'f1', 'auroc']

# Metric display names
METRIC_DISPLAY_NAMES = {
    'acc': 'Accuracy',
    'f1': 'F1 Score',
    'auroc': 'AUROC'
}

# Tasks to exclude from plots
# EXCLUDED_TASKS = {'self_kill_10s', 'global_anyKill_10s'}
EXCLUDED_TASKS = {}


def parse_epoch_probe_folder_name(folder_name: str) -> Optional[dict]:
    """
    Parse epoch_probe folder name to extract metadata.
    
    Format: epoch_probe-{model}-{task_id}-{ui_mask}-e{epoch:02d}-{seed_type}{seed}-{timestamp}-{hash}
    Example: epoch_probe-siglip2-global_anyKill_5s-all-e00-s1-260127-071457-k69k
    
    Args:
        folder_name: Folder name to parse
    
    Returns:
        Dictionary with parsed components, or None if parsing fails
    """
    if not folder_name.startswith('epoch_probe-'):
        return None
    
    # Pattern to match the folder name
    # epoch_probe-{model}-{task_id}-{ui_mask}-e{epoch}-{seed_type}{seed}-{date}-{time}-{hash}
    pattern = r'^epoch_probe-(\w+)-(.+)-(\w+)-e(\d+)-([rs])(\d+)-(\d+)-(\d+)-(\w+)$'
    match = re.match(pattern, folder_name)
    
    if not match:
        return None
    
    model, task_id, ui_mask, epoch, seed_type, seed, date, time, hash_val = match.groups()
    
    # Convert task_id dashes back to underscores
    task_id = task_id.replace('-', '_')
    
    return {
        'model': model,
        'task_id': task_id,
        'ui_mask': ui_mask,
        'epoch': int(epoch),
        'seed_type': seed_type,
        'seed': int(seed),
        'timestamp': f'{date}-{time}',
        'hash': hash_val
    }


def get_task_prefix(task_id: str) -> Optional[str]:
    """Extract task prefix from task_id."""
    for prefix in TASK_PREFIX_ORDER:
        if task_id.startswith(prefix):
            return prefix
    return None


def collect_epoch_probe_results(output_dir: Path, model_filter: str = 'siglip2') -> pd.DataFrame:
    """
    Collect results from epoch_probe experiment folders.
    
    Args:
        output_dir: Path to output directory
        model_filter: Model type to filter (default: 'siglip2')
    
    Returns:
        DataFrame with columns: task_id, epoch, seed, acc, f1, auroc, prefix
    """
    rows = []
    
    for exp_dir in output_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        if not exp_dir.name.startswith('epoch_probe-'):
            continue
        
        # Parse folder name
        parsed = parse_epoch_probe_folder_name(exp_dir.name)
        if parsed is None:
            continue
        
        # Filter by model
        if parsed['model'] != model_filter:
            continue
        
        # Read test results
        results_file = exp_dir / 'test_results_best.json'
        if not results_file.exists():
            continue
        
        with open(results_file) as f:
            results = json.load(f)
        
        metrics = results.get('metrics', {})
        
        row = {
            'task_id': parsed['task_id'],
            'epoch': parsed['epoch'],
            'seed': parsed['seed'],
            'seed_type': parsed['seed_type'],
            'acc': metrics.get('acc'),
            'f1': metrics.get('f1'),
            'auroc': metrics.get('auroc'),
            'prefix': get_task_prefix(parsed['task_id'])
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def collect_probe_results(
    output_dir: Path, 
    model_filter: str = 'siglip2',
    target_tasks: Optional[set] = None
) -> pd.DataFrame:
    """
    Collect results from probe-* experiment folders (baseline and final checkpoint).
    
    These folders have format: probe-{model}-{task_id}-{ui_mask}-{timestamp}-{hash}
    
    - baseline (epoch -1): stage1_checkpoint is None
    - finetuned (epoch 39): stage1_checkpoint points to last.ckpt
    
    Args:
        output_dir: Path to output directory
        model_filter: Model type to filter (default: 'siglip2')
        target_tasks: Set of task_ids to include (None = all)
    
    Returns:
        DataFrame with columns: task_id, epoch, seed, acc, f1, auroc, prefix
    """
    import yaml
    
    rows = []
    ui_masks = {'all', 'minimap_only', 'none'}
    
    # Track seen (task_id, epoch) to avoid duplicates - take most recent
    seen = {}
    
    for exp_dir in output_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        if not exp_dir.name.startswith(f'probe-{model_filter}-'):
            continue
        
        # Skip epoch_probe folders
        if exp_dir.name.startswith('epoch_probe-'):
            continue
        
        # Parse folder name: probe-{model}-{task_id}-{ui_mask}-{timestamp}-{hash}
        parts = exp_dir.name.split('-')
        if len(parts) < 6:
            continue
        
        # Find ui_mask position
        ui_mask_idx = None
        for i, part in enumerate(parts):
            if part in ui_masks:
                ui_mask_idx = i
                break
        
        if ui_mask_idx is None:
            continue
        
        task_id = '_'.join(parts[2:ui_mask_idx])
        ui_mask = parts[ui_mask_idx]
        
        # Only care about 'all' ui_mask
        if ui_mask != 'all':
            continue
        
        # Filter by target tasks if specified
        if target_tasks is not None and task_id not in target_tasks:
            continue
        
        # Check files exist
        hparam_path = exp_dir / 'hparam.yaml'
        results_path = exp_dir / 'test_results_best.json'
        
        if not hparam_path.exists() or not results_path.exists():
            continue
        
        # Read hparam to determine baseline vs finetuned
        with open(hparam_path) as f:
            hparam = yaml.safe_load(f)
        
        ckpt = hparam.get('model', {}).get('stage1_checkpoint')
        
        # baseline = epoch -1, finetuned (last.ckpt) = epoch 39
        if ckpt is None:
            epoch = -1
        else:
            epoch = 39
        
        # Read test results
        with open(results_path) as f:
            results = json.load(f)
        
        metrics = results.get('metrics', {})
        
        # Extract timestamp from folder name for deduplication
        timestamp = '-'.join(parts[ui_mask_idx + 1:-1])
        
        key = (task_id, epoch)
        if key in seen:
            # Keep the most recent one (higher timestamp)
            if timestamp <= seen[key]['timestamp']:
                continue
        
        row = {
            'task_id': task_id,
            'epoch': epoch,
            'seed': 1,  # Default seed for probe experiments
            'seed_type': 'p',  # 'p' for probe
            'acc': metrics.get('acc'),
            'f1': metrics.get('f1'),
            'auroc': metrics.get('auroc'),
            'prefix': get_task_prefix(task_id),
            'timestamp': timestamp
        }
        
        seen[key] = row
        rows.append(row)
    
    # Remove timestamp column before returning
    df = pd.DataFrame(rows)
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    
    return df


def collect_all_results(
    output_dir: Path, 
    model_filter: str = 'siglip2'
) -> pd.DataFrame:
    """
    Collect results from both epoch_probe and probe folders.
    
    Combines:
    - epoch_probe results (epochs 0-34)
    - probe baseline results (epoch -1)
    - probe finetuned results (epoch 39)
    
    Args:
        output_dir: Path to output directory
        model_filter: Model type to filter (default: 'siglip2')
    
    Returns:
        Combined DataFrame with all results
    """
    # Collect epoch_probe results
    df_epoch = collect_epoch_probe_results(output_dir, model_filter)
    
    # Get target tasks from epoch_probe
    target_tasks = set(df_epoch['task_id'].unique()) if not df_epoch.empty else None
    
    # Collect probe results (baseline and final)
    df_probe = collect_probe_results(output_dir, model_filter, target_tasks)
    
    # Combine
    if df_epoch.empty and df_probe.empty:
        return pd.DataFrame()
    elif df_epoch.empty:
        df = df_probe
    elif df_probe.empty:
        df = df_epoch
    else:
        df = pd.concat([df_epoch, df_probe], ignore_index=True)
    
    # Filter out excluded tasks
    if not df.empty and EXCLUDED_TASKS:
        df = df[~df['task_id'].isin(EXCLUDED_TASKS)]
    
    return df


def sort_tasks_by_prefix(tasks: list) -> list:
    """
    Sort tasks by prefix order (self, teammate, enemy, global).
    Within each prefix, sort alphabetically.
    """
    def sort_key(task_id):
        prefix = get_task_prefix(task_id)
        prefix_order = TASK_PREFIX_ORDER.index(prefix) if prefix in TASK_PREFIX_ORDER else len(TASK_PREFIX_ORDER)
        return (prefix_order, task_id)
    
    return sorted(tasks, key=sort_key)


def plot_epoch_metrics_grid(
    df: pd.DataFrame,
    output_dir: Path,
    filename_base: str = 'epoch_metrics_grid'
):
    """
    Create a grid of line plots showing metrics over epochs.
    
    Args:
        df: DataFrame with epoch probe results
        output_dir: Directory to save plots
        filename_base: Base filename for output
    """
    if df.empty:
        print("No data to plot")
        return
    
    # Get unique tasks and sort by prefix
    tasks = sort_tasks_by_prefix(df['task_id'].unique().tolist())
    n_tasks = len(tasks)
    n_metrics = len(METRICS)
    
    # Create figure
    fig, axes = plt.subplots(n_metrics, n_tasks, figsize=(3 * n_tasks, 3 * n_metrics))
    
    # Handle single row/column case
    if n_metrics == 1 and n_tasks == 1:
        axes = np.array([[axes]])
    elif n_metrics == 1:
        axes = axes.reshape(1, -1)
    elif n_tasks == 1:
        axes = axes.reshape(-1, 1)
    
    # Color palette for different seeds
    colors = plt.cm.tab10.colors
    
    for col_idx, task_id in enumerate(tasks):
        df_task = df[df['task_id'] == task_id]
        
        for row_idx, metric in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            
            # Get unique seeds
            seeds = sorted(df_task['seed'].unique())
            
            # Aggregate data by epoch (mean and std across seeds)
            epochs = sorted(df_task['epoch'].unique())
            means = []
            stds = []
            
            for epoch in epochs:
                df_epoch = df_task[df_task['epoch'] == epoch]
                values = df_epoch[metric].dropna()
                if len(values) > 0:
                    means.append(values.mean())
                    stds.append(values.std() if len(values) > 1 else 0)
                else:
                    means.append(np.nan)
                    stds.append(0)
            
            epochs = np.array(epochs)
            means = np.array(means)
            stds = np.array(stds)
            
            # Plot mean line with shaded std region
            ax.plot(epochs, means, 'o-', color='steelblue', linewidth=1.5, markersize=4, label='Mean')
            ax.fill_between(epochs, means - stds, means + stds, color='steelblue', alpha=0.2)
            
            # Also plot individual seeds as faint lines
            for seed_idx, seed in enumerate(seeds):
                df_seed = df_task[df_task['seed'] == seed].sort_values('epoch')
                if not df_seed.empty:
                    ax.plot(df_seed['epoch'], df_seed[metric], 
                           '--', color=colors[seed_idx % len(colors)], 
                           alpha=0.4, linewidth=0.8, label=f'Seed {seed}')
            
            # Styling
            ax.set_xlabel('Epoch' if row_idx == n_metrics - 1 else '')
            ax.set_ylabel(METRIC_DISPLAY_NAMES.get(metric, metric) if col_idx == 0 else '')
            
            # Title: task name on top row only
            if row_idx == 0:
                # Format task name for display
                task_display = task_id.replace('_', ' ').title()
                ax.set_title(task_display, fontsize=9, fontweight='bold')
            
            # Grid
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            
            # Set x-ticks to actual epoch values
            ax.set_xticks(epochs)
            ax.tick_params(axis='both', labelsize=7)
    
    # Add row labels (metric names) on the left
    for row_idx, metric in enumerate(METRICS):
        axes[row_idx, 0].annotate(
            METRIC_DISPLAY_NAMES.get(metric, metric),
            xy=(-0.25, 0.5), xycoords='axes fraction',
            fontsize=10, fontweight='bold', rotation=90,
            ha='center', va='center'
        )
    
    # Add legend for first subplot only
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Only keep unique labels
    unique_labels = []
    unique_handles = []
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)
    
    fig.legend(unique_handles, unique_labels, 
               loc='lower center', 
               bbox_to_anchor=(0.5, 0.01),
               ncol=min(len(unique_labels), 6),
               fontsize=8,
               frameon=True)
    
    # Title
    fig.suptitle('Epoch Probe Results: Metrics Over Training Epochs (siglip2)', 
                 fontsize=12, fontweight='bold', y=0.98)
    
    # Layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.08, top=0.93, right=0.98, hspace=0.25, wspace=0.2)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save in multiple formats
    for ext in ['png', 'pdf', 'svg']:
        save_path = output_dir / f'{filename_base}.{ext}'
        fig.savefig(save_path, bbox_inches='tight', dpi=300 if ext == 'png' else None)
        print(f"Saved: {save_path}")
    
    plt.close()


def get_task_category(task_id: str) -> str:
    """
    Categorize task as 'egocentric' (self) or 'allocentric' (teammate, enemy, global).
    
    Args:
        task_id: Task identifier
    
    Returns:
        'Egocentric' for self tasks, 'Allocentric' for others
    """
    if task_id.startswith('self'):
        return 'Egocentric'
    return 'Allocentric'


def plot_aggregated_epoch_metrics(
    df: pd.DataFrame,
    output_dir: Path,
    normalize: bool = False,
    epoch_filter: Optional[list] = None,
    filename_base: str = 'epoch_metrics_aggregated'
):
    """
    Create a 3x2 grid of line plots showing aggregated metrics over epochs.
    
    Groups tasks into:
    - Egocentric: self_* tasks
    - Allocentric: teammate_*, enemy_*, global_* tasks
    
    Args:
        df: DataFrame with epoch probe results
        output_dir: Directory to save plots
        normalize: If True, normalize metrics relative to baseline (epoch -1)
        epoch_filter: List of epochs to include (None = all epochs)
        filename_base: Base filename for output
    """
    if df.empty:
        print("No data to plot")
        return
    
    # Add category column
    df = df.copy()
    df['category'] = df['task_id'].apply(get_task_category)
    
    # Filter epochs if specified
    if epoch_filter is not None:
        df = df[df['epoch'].isin(epoch_filter)]
        if df.empty:
            print(f"No data after filtering to epochs: {epoch_filter}")
            return
    
    categories = ['Egocentric', 'Allocentric']
    n_categories = len(categories)
    n_metrics = len(METRICS)
    
    # Create figure
    fig, axes = plt.subplots(n_metrics, n_categories, figsize=(5 * n_categories, 4 * n_metrics))
    
    # Handle single row/column case
    if n_metrics == 1 and n_categories == 1:
        axes = np.array([[axes]])
    elif n_metrics == 1:
        axes = axes.reshape(1, -1)
    elif n_categories == 1:
        axes = axes.reshape(-1, 1)
    
    # Colors for different tasks within each category
    colors = plt.cm.tab10.colors
    
    # Track global min/max for same scale (version 1)
    if not normalize:
        global_min = {m: float('inf') for m in METRICS}
        global_max = {m: float('-inf') for m in METRICS}
        
        # First pass to find global min/max
        for metric in METRICS:
            for category in categories:
                df_cat = df[df['category'] == category]
                values = df_cat[metric].dropna()
                if len(values) > 0:
                    global_min[metric] = min(global_min[metric], values.min())
                    global_max[metric] = max(global_max[metric], values.max())
    
    for col_idx, category in enumerate(categories):
        df_cat = df[df['category'] == category]
        tasks_in_cat = sorted(df_cat['task_id'].unique())
        
        for row_idx, metric in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            
            # Get all epochs across all tasks in this category
            all_epochs = sorted(df_cat['epoch'].unique())
            
            # For normalization, compute baseline (epoch -1) mean for each task
            # Fall back to epoch 0 if epoch -1 not available
            baseline_means = {}
            if normalize:
                for task_id in tasks_in_cat:
                    # Try epoch -1 first (true baseline)
                    df_task_baseline = df_cat[(df_cat['task_id'] == task_id) & (df_cat['epoch'] == -1)]
                    values = df_task_baseline[metric].dropna()
                    if len(values) == 0:
                        # Fall back to epoch 0
                        df_task_baseline = df_cat[(df_cat['task_id'] == task_id) & (df_cat['epoch'] == 0)]
                        values = df_task_baseline[metric].dropna()
                    if len(values) > 0:
                        baseline_means[task_id] = values.mean()
            
            # Plot each task as a separate line
            for task_idx, task_id in enumerate(tasks_in_cat):
                df_task = df_cat[df_cat['task_id'] == task_id]
                
                # Aggregate by epoch
                epochs = sorted(df_task['epoch'].unique())
                means = []
                stds = []
                
                for epoch in epochs:
                    df_epoch = df_task[df_task['epoch'] == epoch]
                    values = df_epoch[metric].dropna()
                    if len(values) > 0:
                        mean_val = values.mean()
                        std_val = values.std() if len(values) > 1 else 0
                        
                        if normalize and task_id in baseline_means and baseline_means[task_id] != 0:
                            # Normalize: relative change from baseline
                            baseline = baseline_means[task_id]
                            mean_val = (mean_val - baseline) / baseline * 100  # percentage change
                            std_val = std_val / baseline * 100
                        
                        means.append(mean_val)
                        stds.append(std_val)
                    else:
                        means.append(np.nan)
                        stds.append(0)
                
                epochs = np.array(epochs)
                means = np.array(means)
                stds = np.array(stds)
                
                color = colors[task_idx % len(colors)]
                task_label = task_id.replace('_', ' ').replace('self ', '').replace('teammate ', '').replace('enemy ', '').replace('global ', '')
                
                # Plot line with shaded std
                ax.plot(epochs, means, 'o-', color=color, linewidth=1.5, markersize=4, 
                       label=task_label, alpha=0.8)
                ax.fill_between(epochs, means - stds, means + stds, color=color, alpha=0.15)
            
            # Also plot aggregated mean across all tasks
            agg_means = []
            agg_stds = []
            for epoch in all_epochs:
                df_epoch = df_cat[df_cat['epoch'] == epoch]
                
                if normalize:
                    # For normalized: compute relative change for each task, then average
                    rel_changes = []
                    for task_id in tasks_in_cat:
                        df_task_epoch = df_epoch[df_epoch['task_id'] == task_id]
                        values = df_task_epoch[metric].dropna()
                        if len(values) > 0 and task_id in baseline_means and baseline_means[task_id] != 0:
                            mean_val = values.mean()
                            rel_change = (mean_val - baseline_means[task_id]) / baseline_means[task_id] * 100
                            rel_changes.append(rel_change)
                    if rel_changes:
                        agg_means.append(np.mean(rel_changes))
                        agg_stds.append(np.std(rel_changes) if len(rel_changes) > 1 else 0)
                    else:
                        agg_means.append(np.nan)
                        agg_stds.append(0)
                else:
                    values = df_epoch[metric].dropna()
                    if len(values) > 0:
                        agg_means.append(values.mean())
                        agg_stds.append(values.std() if len(values) > 1 else 0)
                    else:
                        agg_means.append(np.nan)
                        agg_stds.append(0)
            
            all_epochs = np.array(all_epochs)
            agg_means = np.array(agg_means)
            agg_stds = np.array(agg_stds)
            
            # Plot aggregated mean as thick black line
            ax.plot(all_epochs, agg_means, 'o-', color='black', linewidth=2.5, markersize=6, 
                   label='Mean', zorder=10)
            ax.fill_between(all_epochs, agg_means - agg_stds, agg_means + agg_stds, 
                           color='black', alpha=0.1, zorder=5)
            
            # Styling
            ax.set_xlabel('Epoch' if row_idx == n_metrics - 1 else '')
            
            if normalize:
                ylabel = f'{METRIC_DISPLAY_NAMES.get(metric, metric)} (% change)'
            else:
                ylabel = METRIC_DISPLAY_NAMES.get(metric, metric)
            ax.set_ylabel(ylabel if col_idx == 0 else '')
            
            # Title: category name on top row only
            if row_idx == 0:
                ax.set_title(category, fontsize=11, fontweight='bold')
            
            # Grid
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            
            # Set x-ticks to actual epoch values
            ax.set_xticks(all_epochs)
            ax.tick_params(axis='both', labelsize=8)
            
            # Same scale for version 1 (non-normalized)
            if not normalize:
                margin = (global_max[metric] - global_min[metric]) * 0.1
                ax.set_ylim(global_min[metric] - margin, global_max[metric] + margin)
            
            # Add horizontal line at 0 for normalized plots
            if normalize:
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            
            # Legend for each subplot
            ax.legend(loc='best', fontsize=7, framealpha=0.9)
    
    # Add row labels (metric names) on the left
    for row_idx, metric in enumerate(METRICS):
        axes[row_idx, 0].annotate(
            METRIC_DISPLAY_NAMES.get(metric, metric),
            xy=(-0.22, 0.5), xycoords='axes fraction',
            fontsize=11, fontweight='bold', rotation=90,
            ha='center', va='center'
        )
    
    # Title
    if normalize:
        title = 'Epoch Probe Results: Relative Change from Baseline (siglip2)'
    else:
        title = 'Epoch Probe Results: Aggregated by Task Category (siglip2)'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    
    # Layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.10, bottom=0.05, top=0.92, right=0.98, hspace=0.25, wspace=0.15)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = '_normalized' if normalize else '_same_scale'
    
    # Save in multiple formats
    for ext in ['png', 'pdf', 'svg']:
        save_path = output_dir / f'{filename_base}{suffix}.{ext}'
        fig.savefig(save_path, bbox_inches='tight', dpi=300 if ext == 'png' else None)
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_stacked_category_means(
    df: pd.DataFrame,
    output_dir: Path,
    normalize: bool = False,
    epoch_filter: Optional[list] = None,
    filename_base: str = 'epoch_metrics_stacked'
):
    """
    Create a 3x1 grid of line plots showing Egocentric vs Allocentric means.
    
    Each subplot shows two lines: one for Egocentric (self tasks) and one for
    Allocentric (teammate, enemy, global tasks).
    
    Args:
        df: DataFrame with epoch probe results
        output_dir: Directory to save plots
        normalize: If True, normalize metrics relative to baseline (epoch -1)
        epoch_filter: List of epochs to include (None = all epochs)
        filename_base: Base filename for output
    """
    if df.empty:
        print("No data to plot")
        return
    
    # Add category column
    df = df.copy()
    df['category'] = df['task_id'].apply(get_task_category)
    
    # Filter epochs if specified
    if epoch_filter is not None:
        df = df[df['epoch'].isin(epoch_filter)]
        if df.empty:
            print(f"No data after filtering to epochs: {epoch_filter}")
            return
    
    categories = ['Egocentric', 'Allocentric']
    n_metrics = len(METRICS)
    
    # Colors for categories
    category_colors = {
        'Egocentric': '#E74C3C',  # Red
        'Allocentric': '#3498DB'  # Blue
    }
    
    # Create figure (3 rows, 1 column)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(8, 3 * n_metrics))
    
    # Handle single row case
    if n_metrics == 1:
        axes = [axes]
    
    # Get all epochs
    all_epochs = sorted(df['epoch'].unique())
    
    # Compute baseline means for normalization
    baseline_means = {}
    if normalize:
        for category in categories:
            df_cat = df[df['category'] == category]
            tasks_in_cat = df_cat['task_id'].unique()
            
            baseline_means[category] = {}
            for task_id in tasks_in_cat:
                # Try epoch -1 first (true baseline)
                df_task_baseline = df_cat[(df_cat['task_id'] == task_id) & (df_cat['epoch'] == -1)]
                for metric in METRICS:
                    values = df_task_baseline[metric].dropna()
                    if len(values) == 0:
                        # Fall back to epoch 0
                        df_task_e0 = df_cat[(df_cat['task_id'] == task_id) & (df_cat['epoch'] == 0)]
                        values = df_task_e0[metric].dropna()
                    if len(values) > 0:
                        if metric not in baseline_means[category]:
                            baseline_means[category][metric] = {}
                        baseline_means[category][metric][task_id] = values.mean()
    
    for row_idx, metric in enumerate(METRICS):
        ax = axes[row_idx]
        
        for category in categories:
            df_cat = df[df['category'] == category]
            tasks_in_cat = df_cat['task_id'].unique()
            
            # Compute aggregated mean and std for each epoch
            cat_means = []
            cat_stds = []
            
            for epoch in all_epochs:
                df_epoch = df_cat[df_cat['epoch'] == epoch]
                
                if normalize:
                    # Compute relative change for each task, then average
                    rel_changes = []
                    for task_id in tasks_in_cat:
                        df_task_epoch = df_epoch[df_epoch['task_id'] == task_id]
                        values = df_task_epoch[metric].dropna()
                        if (len(values) > 0 and 
                            category in baseline_means and 
                            metric in baseline_means[category] and
                            task_id in baseline_means[category][metric] and
                            baseline_means[category][metric][task_id] != 0):
                            mean_val = values.mean()
                            baseline = baseline_means[category][metric][task_id]
                            rel_change = (mean_val - baseline) / baseline * 100
                            rel_changes.append(rel_change)
                    
                    if rel_changes:
                        cat_means.append(np.mean(rel_changes))
                        cat_stds.append(np.std(rel_changes) if len(rel_changes) > 1 else 0)
                    else:
                        cat_means.append(np.nan)
                        cat_stds.append(0)
                else:
                    values = df_epoch[metric].dropna()
                    if len(values) > 0:
                        cat_means.append(values.mean())
                        cat_stds.append(values.std() if len(values) > 1 else 0)
                    else:
                        cat_means.append(np.nan)
                        cat_stds.append(0)
            
            epochs_arr = np.array(all_epochs)
            cat_means = np.array(cat_means)
            cat_stds = np.array(cat_stds)
            
            color = category_colors[category]
            
            # Plot line with shaded std
            ax.plot(epochs_arr, cat_means, 'o-', color=color, linewidth=2, markersize=6, 
                   label=category, alpha=0.9)
            ax.fill_between(epochs_arr, cat_means - cat_stds, cat_means + cat_stds, 
                           color=color, alpha=0.15)
        
        # Styling
        ax.set_xlabel('Epoch' if row_idx == n_metrics - 1 else '')
        
        if normalize:
            ylabel = f'{METRIC_DISPLAY_NAMES.get(metric, metric)} (% change)'
        else:
            ylabel = METRIC_DISPLAY_NAMES.get(metric, metric)
        ax.set_ylabel(ylabel)
        
        # Title for each subplot
        ax.set_title(METRIC_DISPLAY_NAMES.get(metric, metric), fontsize=11, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        # Set x-ticks to actual epoch values
        ax.set_xticks(all_epochs)
        ax.tick_params(axis='both', labelsize=9)
        
        # Add horizontal line at 0 for normalized plots
        if normalize:
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Legend
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
    
    # Title
    if normalize:
        title = 'Egocentric vs Allocentric: Relative Change from Baseline (siglip2)'
    else:
        title = 'Egocentric vs Allocentric: Mean Performance Over Epochs (siglip2)'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    
    # Layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = '_normalized' if normalize else '_same_scale'
    
    # Save in multiple formats
    for ext in ['png', 'pdf', 'svg']:
        save_path = output_dir / f'{filename_base}{suffix}.{ext}'
        fig.savefig(save_path, bbox_inches='tight', dpi=300 if ext == 'png' else None)
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_pareto_scatter(
    df: pd.DataFrame,
    output_dir: Path,
    normalize: bool = False,
    epoch_filter: Optional[list] = None,
    filename_base: str = 'pareto_scatter'
):
    """
    Create a Pareto-style scatter plot showing Egocentric vs Allocentric performance.
    
    X-axis: Allocentric mean performance
    Y-axis: Egocentric mean performance
    Color: Epoch (gradient from early to late)
    
    Args:
        df: DataFrame with epoch probe results
        output_dir: Directory to save plots
        normalize: If True, normalize metrics relative to baseline (epoch -1)
        epoch_filter: List of epochs to include (None = all epochs)
        filename_base: Base filename for output
    """
    if df.empty:
        print("No data to plot")
        return
    
    # Add category column
    df = df.copy()
    df['category'] = df['task_id'].apply(get_task_category)
    
    # Filter epochs if specified
    if epoch_filter is not None:
        df = df[df['epoch'].isin(epoch_filter)]
        if df.empty:
            print(f"No data after filtering to epochs: {epoch_filter}")
            return
    
    categories = ['Egocentric', 'Allocentric']
    n_metrics = len(METRICS)
    
    # Get all epochs
    all_epochs = sorted(df['epoch'].unique())
    
    # Compute baseline means for normalization
    baseline_means = {}
    if normalize:
        for category in categories:
            df_cat = df[df['category'] == category]
            tasks_in_cat = df_cat['task_id'].unique()
            
            baseline_means[category] = {}
            for task_id in tasks_in_cat:
                # Try epoch -1 first (true baseline)
                df_task_baseline = df_cat[(df_cat['task_id'] == task_id) & (df_cat['epoch'] == -1)]
                for metric in METRICS:
                    values = df_task_baseline[metric].dropna()
                    if len(values) == 0:
                        # Fall back to epoch 0
                        df_task_e0 = df_cat[(df_cat['task_id'] == task_id) & (df_cat['epoch'] == 0)]
                        values = df_task_e0[metric].dropna()
                    if len(values) > 0:
                        if metric not in baseline_means[category]:
                            baseline_means[category][metric] = {}
                        baseline_means[category][metric][task_id] = values.mean()
    
    # Compute mean performance for each category and epoch (aggregated)
    pareto_data_agg = {metric: [] for metric in METRICS}
    # Also collect individual seed data points
    pareto_data_seeds = {metric: [] for metric in METRICS}
    
    # Get unique seeds
    all_seeds = sorted(df['seed'].unique())
    
    for epoch in all_epochs:
        df_epoch = df[df['epoch'] == epoch]
        
        epoch_data = {'epoch': epoch}
        
        for category in categories:
            df_cat_epoch = df_epoch[df_epoch['category'] == category]
            tasks_in_cat = df[df['category'] == category]['task_id'].unique()
            
            for metric in METRICS:
                if normalize:
                    # Compute relative change for each task, then average
                    rel_changes = []
                    for task_id in tasks_in_cat:
                        df_task_epoch = df_cat_epoch[df_cat_epoch['task_id'] == task_id]
                        values = df_task_epoch[metric].dropna()
                        if (len(values) > 0 and 
                            category in baseline_means and 
                            metric in baseline_means[category] and
                            task_id in baseline_means[category][metric] and
                            baseline_means[category][metric][task_id] != 0):
                            mean_val = values.mean()
                            baseline = baseline_means[category][metric][task_id]
                            rel_change = (mean_val - baseline) / baseline * 100
                            rel_changes.append(rel_change)
                    
                    if rel_changes:
                        epoch_data[f'{category}_{metric}'] = np.mean(rel_changes)
                    else:
                        epoch_data[f'{category}_{metric}'] = np.nan
                else:
                    values = df_cat_epoch[metric].dropna()
                    if len(values) > 0:
                        epoch_data[f'{category}_{metric}'] = values.mean()
                    else:
                        epoch_data[f'{category}_{metric}'] = np.nan
        
        for metric in METRICS:
            pareto_data_agg[metric].append({
                'epoch': epoch,
                'egocentric': epoch_data.get(f'Egocentric_{metric}', np.nan),
                'allocentric': epoch_data.get(f'Allocentric_{metric}', np.nan)
            })
        
        # Collect individual seed data points
        for seed in all_seeds:
            df_epoch_seed = df_epoch[df_epoch['seed'] == seed]
            
            if df_epoch_seed.empty:
                continue
            
            seed_data = {'epoch': epoch, 'seed': seed}
            
            for category in categories:
                df_cat_epoch_seed = df_epoch_seed[df_epoch_seed['category'] == category]
                tasks_in_cat = df[df['category'] == category]['task_id'].unique()
                
                for metric in METRICS:
                    if normalize:
                        rel_changes = []
                        for task_id in tasks_in_cat:
                            df_task = df_cat_epoch_seed[df_cat_epoch_seed['task_id'] == task_id]
                            values = df_task[metric].dropna()
                            if (len(values) > 0 and 
                                category in baseline_means and 
                                metric in baseline_means[category] and
                                task_id in baseline_means[category][metric] and
                                baseline_means[category][metric][task_id] != 0):
                                mean_val = values.mean()
                                baseline = baseline_means[category][metric][task_id]
                                rel_change = (mean_val - baseline) / baseline * 100
                                rel_changes.append(rel_change)
                        
                        if rel_changes:
                            seed_data[f'{category}_{metric}'] = np.mean(rel_changes)
                        else:
                            seed_data[f'{category}_{metric}'] = np.nan
                    else:
                        values = df_cat_epoch_seed[metric].dropna()
                        if len(values) > 0:
                            seed_data[f'{category}_{metric}'] = values.mean()
                        else:
                            seed_data[f'{category}_{metric}'] = np.nan
            
            for metric in METRICS:
                ego_val = seed_data.get(f'Egocentric_{metric}', np.nan)
                allo_val = seed_data.get(f'Allocentric_{metric}', np.nan)
                if not np.isnan(ego_val) and not np.isnan(allo_val):
                    pareto_data_seeds[metric].append({
                        'epoch': epoch,
                        'seed': seed,
                        'egocentric': ego_val,
                        'allocentric': allo_val
                    })
    
    # Create figure (1 row, 3 columns for 3 metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    # Color map for epochs
    cmap = plt.cm.viridis
    epoch_min = min(all_epochs)
    epoch_max = max(all_epochs)
    epoch_range = epoch_max - epoch_min if epoch_max != epoch_min else 1
    
    for ax_idx, metric in enumerate(METRICS):
        ax = axes[ax_idx]
        
        data_agg = pareto_data_agg[metric]
        data_seeds = pareto_data_seeds[metric]
        
        # Plot individual seed points
        for point in data_seeds:
            epoch = point['epoch']
            ego = point['egocentric']
            allo = point['allocentric']
            
            # Color based on epoch
            color_val = (epoch - epoch_min) / epoch_range
            color = cmap(color_val)
            
            ax.scatter(allo, ego, c=[color], s=40, alpha=0.6, edgecolors='none', zorder=1)
        
        # Styling
        if normalize:
            ax.set_xlabel('Allocentric (% change)')
            ax.set_ylabel('Egocentric (% change)')
            # Add reference lines at 0
            ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        else:
            ax.set_xlabel('Allocentric')
            ax.set_ylabel('Egocentric')
        
        ax.set_title(METRIC_DISPLAY_NAMES.get(metric, metric), fontsize=11, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=epoch_min, vmax=epoch_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Epoch', fontsize=10)
    
    # Title
    if normalize:
        title = 'Pareto Trade-off: Egocentric vs Allocentric (% change from baseline)'
    else:
        title = 'Pareto Trade-off: Egocentric vs Allocentric Performance'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    
    # Layout
    plt.tight_layout()
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = '_normalized' if normalize else '_absolute'
    
    # Save in multiple formats
    for ext in ['png', 'pdf', 'svg']:
        save_path = output_dir / f'{filename_base}{suffix}.{ext}'
        fig.savefig(save_path, bbox_inches='tight', dpi=300 if ext == 'png' else None)
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_combined_accuracy_early(
    df: pd.DataFrame,
    output_dir: Path,
    filename_base: str = 'combined_accuracy_early'
):
    """
    Create a combined plot with two side-by-side subplots for Accuracy only:
    - Left: Stacked category means (Egocentric vs Allocentric over epochs)
    - Right: Pareto scatter plot (Egocentric vs Allocentric)
    
    Uses early epoch filter and normalized values.
    Epoch labels are shifted from -1,0,1,... to 0,1,2,...
    No title is shown.
    
    Args:
        df: DataFrame with epoch probe results
        output_dir: Directory to save plots
        filename_base: Base filename for output
    """
    if df.empty:
        print("No data to plot")
        return
    
    # Settings
    metric = 'acc'
    epoch_filter = [-1, 0, 1, 2, 3, 4]  # Early epochs
    normalize = True
    
    # Add category column
    df = df.copy()
    df['category'] = df['task_id'].apply(get_task_category)
    
    # Filter epochs
    df = df[df['epoch'].isin(epoch_filter)]
    if df.empty:
        print(f"No data after filtering to epochs: {epoch_filter}")
        return
    
    categories = ['Egocentric', 'Allocentric']
    
    # Colors for categories
    category_colors = {
        'Egocentric': '#E74C3C',  # Red
        'Allocentric': '#3498DB'  # Blue
    }
    
    # Get all epochs and create shifted mapping
    all_epochs = sorted(df['epoch'].unique())
    epoch_shift = {e: e + 1 for e in all_epochs}  # -1->0, 0->1, 1->2, etc.
    shifted_epochs = [epoch_shift[e] for e in all_epochs]
    
    # Compute baseline means for normalization
    baseline_means = {}
    for category in categories:
        df_cat = df[df['category'] == category]
        tasks_in_cat = df_cat['task_id'].unique()
        
        baseline_means[category] = {}
        for task_id in tasks_in_cat:
            # Try epoch -1 first (true baseline)
            df_task_baseline = df_cat[(df_cat['task_id'] == task_id) & (df_cat['epoch'] == -1)]
            values = df_task_baseline[metric].dropna()
            if len(values) == 0:
                # Fall back to epoch 0
                df_task_e0 = df_cat[(df_cat['task_id'] == task_id) & (df_cat['epoch'] == 0)]
                values = df_task_e0[metric].dropna()
            if len(values) > 0:
                baseline_means[category][task_id] = values.mean()
    
    # Create figure (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    ax_left, ax_right = axes
    
    # ========== LEFT PLOT: Stacked category means ==========
    for category in categories:
        df_cat = df[df['category'] == category]
        tasks_in_cat = df_cat['task_id'].unique()
        
        # Compute aggregated mean and std for each epoch
        cat_means = []
        cat_stds = []
        
        for epoch in all_epochs:
            df_epoch = df_cat[df_cat['epoch'] == epoch]
            
            # Compute relative change for each task, then average
            rel_changes = []
            for task_id in tasks_in_cat:
                df_task_epoch = df_epoch[df_epoch['task_id'] == task_id]
                values = df_task_epoch[metric].dropna()
                if (len(values) > 0 and 
                    task_id in baseline_means[category] and
                    baseline_means[category][task_id] != 0):
                    mean_val = values.mean()
                    baseline = baseline_means[category][task_id]
                    rel_change = (mean_val - baseline) / baseline * 100
                    rel_changes.append(rel_change)
            
            if rel_changes:
                cat_means.append(np.mean(rel_changes))
                cat_stds.append(np.std(rel_changes) if len(rel_changes) > 1 else 0)
            else:
                cat_means.append(np.nan)
                cat_stds.append(0)
        
        cat_means = np.array(cat_means)
        cat_stds = np.array(cat_stds)
        
        color = category_colors[category]
        
        # Plot line with shaded std using shifted epochs
        ax_left.plot(shifted_epochs, cat_means, 'o-', color=color, linewidth=2, markersize=6, 
                    label=category, alpha=0.9)
        ax_left.fill_between(shifted_epochs, cat_means - cat_stds, cat_means + cat_stds, 
                            color=color, alpha=0.15)
    
    # Left plot styling
    ax_left.set_xlabel('Epochs')
    ax_left.set_ylabel('Accuracy (% change)')
    ax_left.set_xticks(shifted_epochs)
    ax_left.tick_params(axis='both', labelsize=9)
    ax_left.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax_left.legend(loc='best', fontsize=9, framealpha=0.9)
    
    # ========== RIGHT PLOT: Pareto scatter ==========
    # Collect individual seed data points
    pareto_data_seeds = []
    all_seeds = sorted(df['seed'].unique())
    
    for epoch in all_epochs:
        df_epoch = df[df['epoch'] == epoch]
        
        for seed in all_seeds:
            df_epoch_seed = df_epoch[df_epoch['seed'] == seed]
            
            if df_epoch_seed.empty:
                continue
            
            seed_data = {'epoch': epoch, 'seed': seed}
            
            for category in categories:
                df_cat_epoch_seed = df_epoch_seed[df_epoch_seed['category'] == category]
                tasks_in_cat = df[df['category'] == category]['task_id'].unique()
                
                rel_changes = []
                for task_id in tasks_in_cat:
                    df_task = df_cat_epoch_seed[df_cat_epoch_seed['task_id'] == task_id]
                    values = df_task[metric].dropna()
                    if (len(values) > 0 and 
                        category in baseline_means and 
                        task_id in baseline_means[category] and
                        baseline_means[category][task_id] != 0):
                        mean_val = values.mean()
                        baseline = baseline_means[category][task_id]
                        rel_change = (mean_val - baseline) / baseline * 100
                        rel_changes.append(rel_change)
                
                if rel_changes:
                    seed_data[f'{category}'] = np.mean(rel_changes)
                else:
                    seed_data[f'{category}'] = np.nan
            
            ego_val = seed_data.get('Egocentric', np.nan)
            allo_val = seed_data.get('Allocentric', np.nan)
            if not np.isnan(ego_val) and not np.isnan(allo_val):
                pareto_data_seeds.append({
                    'epoch': epoch,
                    'seed': seed,
                    'egocentric': ego_val,
                    'allocentric': allo_val
                })
    
    # Color map for epochs (using shifted values for colorbar)
    cmap = plt.cm.viridis
    epoch_min = min(all_epochs)
    epoch_max = max(all_epochs)
    epoch_range = epoch_max - epoch_min if epoch_max != epoch_min else 1
    
    # Plot individual seed points
    for point in pareto_data_seeds:
        epoch = point['epoch']
        ego = point['egocentric']
        allo = point['allocentric']
        
        # Color based on epoch
        color_val = (epoch - epoch_min) / epoch_range
        color = cmap(color_val)
        
        ax_right.scatter(allo, ego, c=[color], s=40, alpha=0.6, edgecolors='none', zorder=1)
    
    # Right plot styling
    ax_right.set_xlabel('Allocentric (% change)')
    ax_right.set_ylabel('Egocentric (% change)')
    ax_right.axhline(y=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax_right.axvline(x=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    
    # Add colorbar with shifted epoch labels
    shifted_min = epoch_shift[epoch_min]
    shifted_max = epoch_shift[epoch_max]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=shifted_min, vmax=shifted_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_right, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Epochs', fontsize=10)
    
    # Layout (no suptitle)
    plt.tight_layout()
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save in multiple formats
    for ext in ['png', 'pdf', 'svg']:
        save_path = output_dir / f'{filename_base}.{ext}'
        fig.savefig(save_path, bbox_inches='tight', dpi=300 if ext == 'png' else None)
        print(f"Saved: {save_path}")
    
    plt.close()


def main():
    """Main entry point."""
    # Paths
    project_root = Path(__file__).parent.parent.parent.parent
    output_dir = project_root / 'output'
    artifacts_dir = project_root / 'artifacts' / 'pareto'
    
    print(f"Collecting results from: {output_dir}")
    
    # Collect all results (epoch_probe + probe baseline/final)
    df = collect_all_results(output_dir, model_filter='siglip2')
    
    print(f"Collected {len(df)} results")
    print(f"Unique tasks: {sorted(df['task_id'].unique())}")
    print(f"Unique epochs: {sorted(df['epoch'].unique())}")
    print(f"Unique seeds: {sorted(df['seed'].unique())}")
    
    if df.empty:
        print("No results found!")
        return
    
    # Create plots
    plot_epoch_metrics_grid(df, artifacts_dir)
    
    # Create combined accuracy plot (early epochs, normalized)
    print("\n=== Creating combined accuracy plot ===")
    plot_combined_accuracy_early(df, artifacts_dir)
    
    # Define epoch filter versions
    # Version 1: All epochs (current version)
    epoch_filters = {
        'all': None,  # All epochs: -1, 0, 1, 2, 3, 4, 9, 14, 19, 24, 29, 34, 39
        'early': [-1, 0, 1, 2, 3, 4],  # Baseline + early epochs
        'sparse': [-1, 4, 9, 14, 19, 24, 29, 34, 39],  # Baseline + sparse epochs
    }
    
    # Create aggregated plots for each epoch filter version
    for version_name, epoch_filter in epoch_filters.items():
        print(f"\n=== Creating plots for version: {version_name} ===")
        
        # Same scale version
        print(f"Creating aggregated plot (same scale, {version_name})...")
        plot_aggregated_epoch_metrics(
            df, artifacts_dir, 
            normalize=False, 
            epoch_filter=epoch_filter,
            filename_base=f'epoch_metrics_aggregated_{version_name}'
        )
        
        # Normalized version
        print(f"Creating aggregated plot (normalized, {version_name})...")
        plot_aggregated_epoch_metrics(
            df, artifacts_dir, 
            normalize=True, 
            epoch_filter=epoch_filter,
            filename_base=f'epoch_metrics_aggregated_{version_name}'
        )
        
        # Stacked category means (two lines: Egocentric vs Allocentric)
        print(f"Creating stacked plot (same scale, {version_name})...")
        plot_stacked_category_means(
            df, artifacts_dir,
            normalize=False,
            epoch_filter=epoch_filter,
            filename_base=f'epoch_metrics_stacked_{version_name}'
        )
        
        print(f"Creating stacked plot (normalized, {version_name})...")
        plot_stacked_category_means(
            df, artifacts_dir,
            normalize=True,
            epoch_filter=epoch_filter,
            filename_base=f'epoch_metrics_stacked_{version_name}'
        )
        
        # Pareto scatter plot (Egocentric vs Allocentric)
        print(f"Creating pareto scatter plot (absolute, {version_name})...")
        plot_pareto_scatter(
            df, artifacts_dir,
            normalize=False,
            epoch_filter=epoch_filter,
            filename_base=f'pareto_scatter_{version_name}'
        )
        
        print(f"Creating pareto scatter plot (normalized, {version_name})...")
        plot_pareto_scatter(
            df, artifacts_dir,
            normalize=True,
            epoch_filter=epoch_filter,
            filename_base=f'pareto_scatter_{version_name}'
        )
    
    print(f"\nPlots saved to: {artifacts_dir}")


if __name__ == '__main__':
    main()
