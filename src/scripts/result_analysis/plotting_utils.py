#!/usr/bin/env python3
"""
Plotting Utilities

Utility functions for creating various plots from results data.
"""

from pathlib import Path
from typing import Optional
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10


# Task prefix categories
TASK_PREFIXES = ['enemy', 'self', 'global', 'teammate']

# Shortest time horizon for each task base (used to filter prefix aggregation)
# Tasks not listed here have no time horizon variants
SHORTEST_HORIZON_MAP = {
    'self_location': 0,
    'teammate_location': 0,
    'enemy_location': 0,
    'self_kill': 5,
    'self_death': 5,
    'global_anyKill': 5,
}

# Metrics available for each ML form
METRICS_BY_ML_FORM = {
    'binary_cls': ['acc', 'f1', 'auroc'],
    'multi_cls': ['acc', 'f1', 'acc_top3', 'acc_top5'],
    'multi_label_cls': ['acc_exact', 'hamming_dist', 'f1', 'auroc'],  # hamming_acc = 1 - hamming_dist
    'regression': ['mae', 'mse', 'r2']
}

# Primary metric for each ML form (used when only one metric is needed)
PRIMARY_METRIC_MAP = {
    'binary_cls': 'acc',
    'multi_cls': 'acc',
    'multi_label_cls': 'hamming_acc',
    'regression': 'mae'
}


def get_metrics_for_ml_form(ml_form: str) -> list:
    """Get all available metrics for a given ML form."""
    return METRICS_BY_ML_FORM.get(ml_form, ['acc'])


def get_primary_metric_for_ml_form(ml_form: str) -> str:
    """Get the primary metric for a given ML form."""
    return PRIMARY_METRIC_MAP.get(ml_form, 'acc')


def save_plot_multi_format(fig, output_dir: Path, filename_base: str):
    """
    Save a plot in three formats (PNG, PDF, SVG) in dedicated subfolders.
    
    Args:
        fig: matplotlib figure object
        output_dir: Base output directory
        filename_base: Filename without extension (e.g., 'my_plot')
    """
    output_dir = Path(output_dir)
    
    # Create subfolders
    png_dir = output_dir / 'png'
    pdf_dir = output_dir / 'pdf'
    svg_dir = output_dir / 'svg'
    
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)
    
    # Save in all three formats
    png_path = png_dir / f'{filename_base}.png'
    pdf_path = pdf_dir / f'{filename_base}.pdf'
    svg_path = svg_dir / f'{filename_base}.svg'
    
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(svg_path, bbox_inches='tight')
    
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {svg_path}")


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess dataframe to add computed metrics.
    Adds hamming_acc = 1 - hamming_dist for multi_label_cls tasks.
    """
    df = df.copy()
    if 'hamming_dist' in df.columns:
        df['hamming_acc'] = 1 - df['hamming_dist']
    return df


def get_task_prefix(task_id: str) -> Optional[str]:
    """Extract task prefix category from task_id."""
    for prefix in TASK_PREFIXES:
        if task_id.startswith(prefix):
            return prefix
    return None


def extract_time_horizon(task_id: str) -> Optional[int]:
    """
    Extract time horizon from task_id suffix like _0s, _10s, etc.
    
    Returns:
        Time horizon in seconds, or None if not found
    """
    match = re.search(r'_(\d+)s$', task_id)
    if match:
        return int(match.group(1))
    return None


def get_task_base_name(task_id: str) -> str:
    """Get task name without time horizon suffix."""
    return re.sub(r'_\d+s$', '', task_id)


def is_shortest_horizon_task(task_id: str) -> bool:
    """
    Check if a task is the shortest horizon variant for its task base.
    
    For tasks with multiple time horizons (e.g., self_location_0s, self_location_5s),
    returns True only for the shortest horizon variant.
    For tasks without time horizons, always returns True.
    
    Args:
        task_id: Task identifier (e.g., 'self_location_0s', 'self_kill_5s')
    
    Returns:
        True if this is the shortest horizon variant or has no horizon
    """
    task_base = get_task_base_name(task_id)
    time_horizon = extract_time_horizon(task_id)
    
    # If task has no time horizon suffix, include it
    if time_horizon is None:
        return True
    
    # If task base is in the shortest horizon map, check if this is the shortest
    if task_base in SHORTEST_HORIZON_MAP:
        return time_horizon == SHORTEST_HORIZON_MAP[task_base]
    
    # For task bases not in the map, include all variants
    return True


def _get_best_experiment_value(df_subset: pd.DataFrame, metric_name: str) -> float:
    """
    Get the metric value from the best experiment when there are duplicates.
    
    Uses the most recent experiment based on exp_name timestamp.
    Experiment names contain timestamps like: probe-model-task-ui-DDMMYY-HHMMSS-hash
    
    Args:
        df_subset: DataFrame subset for a specific task/init_type combination
        metric_name: Name of the metric to extract
    
    Returns:
        Metric value from the most recent experiment, or np.nan if empty
    """
    if df_subset.empty:
        return np.nan
    
    if len(df_subset) == 1:
        return df_subset[metric_name].values[0]
    
    # Multiple experiments - take the most recent one based on exp_name
    # Sort by exp_name descending (later timestamps come later alphabetically for same date)
    # The timestamp format is DDMMYY-HHMMSS, so we need to extract and compare
    df_sorted = df_subset.sort_values('exp_name', ascending=False)
    return df_sorted[metric_name].values[0]


def _aggregate_repeat_experiments(
    df_subset: pd.DataFrame, 
    metric_name: str
) -> tuple[float, float, int]:
    """
    Aggregate metric values from repeat experiments (same model, task, init_type).
    
    Args:
        df_subset: DataFrame subset for a specific task/init_type combination
        metric_name: Name of the metric to extract
    
    Returns:
        Tuple of (mean, std, count) for the metric across repeats.
        Returns (np.nan, np.nan, 0) if empty.
    """
    if df_subset.empty:
        return np.nan, np.nan, 0
    
    values = df_subset[metric_name].dropna().values
    
    if len(values) == 0:
        return np.nan, np.nan, 0
    
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
    
    return mean_val, std_val, len(values)


def _aggregate_group_repeats(
    df_subset: pd.DataFrame,
    metric_name: str,
    group_col: str = 'task_id'
) -> tuple[float, float, int]:
    """
    Aggregate metric values across tasks within a group (e.g., task prefix),
    accounting for repeats within each task.
    
    For each unique task, first computes the mean across repeats,
    then averages across tasks in the group.
    The std is computed by pooling std across repeats for all tasks.
    
    Args:
        df_subset: DataFrame subset for a group (e.g., specific prefix, model, init_type)
        metric_name: Name of the metric to extract
        group_col: Column to group by (default: 'task_id')
    
    Returns:
        Tuple of (mean, std, count) where:
        - mean: average of per-task means
        - std: pooled std across all repeats
        - count: total number of repeat experiments
    """
    if df_subset.empty:
        return np.nan, np.nan, 0
    
    # Collect all repeat values and per-task means
    all_values = []
    task_means = []
    for task_id in df_subset[group_col].unique():
        df_task = df_subset[df_subset[group_col] == task_id]
        values = df_task[metric_name].dropna().values
        if len(values) > 0:
            task_means.append(np.mean(values))
            all_values.extend(values)
    
    if len(task_means) == 0:
        return np.nan, np.nan, 0
    
    # Mean is average of per-task means
    mean_val = np.mean(task_means)
    # Std is computed from all individual repeat values (pooled std)
    std_val = np.std(all_values, ddof=1) if len(all_values) > 1 else 0.0
    
    return mean_val, std_val, len(all_values)


def plot_baseline_vs_finetuned_per_model(
    df: pd.DataFrame,
    ml_form: str,
    ui_mask: str,
    checkpoint_type: str,
    output_dir: Path
):
    """
    Plot baseline vs finetuned comparison for each model.
    Creates one plot per model with subplots for each metric.
    Shows error bars (standard deviation) for repeat experiments.
    
    Args:
        df: DataFrame filtered by ml_form and ui_mask
        ml_form: ML form type
        ui_mask: UI mask setting
        checkpoint_type: 'best' or 'last'
        output_dir: Directory to save plots
    """
    df = preprocess_dataframe(df)
    metrics = get_metrics_for_ml_form(ml_form)
    
    # Filter to metrics that exist in the dataframe
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        print(f"No metrics found for {ml_form}")
        return
    
    models = sorted(df['model_type'].unique())
    tasks = sorted(df['task_id'].unique())
    
    if len(tasks) == 0:
        return
    
    n_metrics = len(available_metrics)
    
    # Create one plot per model
    for model in models:
        df_model = df[df['model_type'] == model].copy()
        
        if df_model.empty:
            continue
        
        # Create subplots for each metric
        fig, axes = plt.subplots(n_metrics, 1, figsize=(max(14, len(tasks) * 0.5), 5 * n_metrics))
        
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric_name in zip(axes, available_metrics):
            # Prepare data for plotting with mean and std from repeats
            baseline_means = []
            baseline_stds = []
            finetuned_means = []
            finetuned_stds = []
            task_labels = []
            
            for task in tasks:
                df_task = df_model[df_model['task_id'] == task]
                
                df_baseline = df_task[df_task['init_type'] == 'baseline']
                df_finetuned = df_task[df_task['init_type'] == 'finetuned']
                
                # Aggregate across repeats
                baseline_mean, baseline_std, _ = _aggregate_repeat_experiments(df_baseline, metric_name)
                finetuned_mean, finetuned_std, _ = _aggregate_repeat_experiments(df_finetuned, metric_name)
                
                baseline_means.append(baseline_mean)
                baseline_stds.append(baseline_std if not np.isnan(baseline_std) else 0)
                finetuned_means.append(finetuned_mean)
                finetuned_stds.append(finetuned_std if not np.isnan(finetuned_std) else 0)
                task_labels.append(task)
            
            x = np.arange(len(tasks))
            width = 0.35
            
            # Plot bars with error bars
            ax.bar(x - width/2, baseline_means, width, label='Baseline', 
                   color='steelblue', edgecolor='black', alpha=0.8,
                   yerr=baseline_stds, capsize=3, error_kw={'linewidth': 1.5, 'capthick': 1.5})
            ax.bar(x + width/2, finetuned_means, width, label='Finetuned', 
                   color='coral', edgecolor='black', alpha=0.8,
                   yerr=finetuned_stds, capsize=3, error_kw={'linewidth': 1.5, 'capthick': 1.5})
            
            ax.set_xlabel('Task', fontsize=10)
            ax.set_ylabel(f'{metric_name.upper()}', fontsize=10)
            ax.set_title(f'{metric_name.upper()}', fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(task_labels, rotation=45, ha='right', fontsize=7)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add horizontal line at mean
            all_vals = [v for v in baseline_means + finetuned_means if not np.isnan(v)]
            if all_vals:
                ax.axhline(np.mean(all_vals), color='gray', linestyle='--', alpha=0.5)
        
        fig.suptitle(f'{model.upper()} - Baseline vs Finetuned\n'
                     f'ML Form: {ml_form} | UI: {ui_mask} | Checkpoint: {checkpoint_type}',
                     fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        filename_base = f'baseline_vs_finetuned_{ml_form}_{model}'
        save_plot_multi_format(fig, output_dir, filename_base)
        plt.close()


def plot_by_task_prefix(
    df: pd.DataFrame,
    ml_form: str,
    ui_mask: str,
    checkpoint_type: str,
    output_dir: Path
):
    """
    Plot results grouped by task prefix (enemy, self, global, teammate).
    Uses two colors for baseline/finetuned and hatch patterns for models.
    Creates subplots for each metric.
    Shows error bars (std across tasks within each prefix).
    
    For tasks with multiple time horizons, only includes the shortest horizon
    variant in the aggregation (longer horizons are shown in time horizon plots).
    
    Args:
        df: DataFrame filtered by ml_form and ui_mask
        ml_form: ML form type
        ui_mask: UI mask setting
        checkpoint_type: 'best' or 'last'
        output_dir: Directory to save plots
    """
    df = preprocess_dataframe(df)
    metrics = get_metrics_for_ml_form(ml_form)
    
    # Filter to metrics that exist in the dataframe
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        print(f"No metrics found for {ml_form}")
        return
    
    # Add prefix column
    df = df.copy()
    df['task_prefix'] = df['task_id'].apply(get_task_prefix)
    
    # Filter to only include shortest horizon tasks for prefix aggregation
    df = df[df['task_id'].apply(is_shortest_horizon_task)]
    
    # Filter to known prefixes
    df = df[df['task_prefix'].notna()]
    
    if df.empty:
        print(f"No tasks with known prefixes for {ml_form}, {ui_mask}")
        return
    
    # Get unique prefixes present
    prefixes = [p for p in TASK_PREFIXES if p in df['task_prefix'].values]
    
    if len(prefixes) == 0:
        return
    
    models = sorted(df['model_type'].unique())
    init_types = [it for it in ['baseline', 'finetuned'] if it in df['init_type'].unique()]
    
    # Two colors for baseline vs finetuned
    colors = {'baseline': 'steelblue', 'finetuned': 'coral'}
    
    # Different hatch patterns for models
    hatches = {'clip': '', 'dinov2': '///', 'siglip2': '...', 'vjepa2': 'xxx'}
    
    n_metrics = len(available_metrics)
    
    # Create subplots for each metric
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric_name in zip(axes, available_metrics):
        x = np.arange(len(prefixes))
        n_groups = len(models) * len(init_types)
        width = 0.8 / n_groups
        
        bar_idx = 0
        for model in models:
            for init_type in init_types:
                means = []
                stds = []
                for prefix in prefixes:
                    df_subset = df[(df['task_prefix'] == prefix) & 
                                  (df['model_type'] == model) & 
                                  (df['init_type'] == init_type)]
                    # Aggregate: for each task_id, compute mean across repeats,
                    # then compute mean and std across tasks in prefix
                    mean_val, std_val, _ = _aggregate_group_repeats(df_subset, metric_name, 'task_id')
                    means.append(mean_val)
                    stds.append(std_val if not np.isnan(std_val) else 0)
                
                offset = (bar_idx - n_groups/2 + 0.5) * width
                ax.bar(x + offset, means, width, 
                       label=f'{model}-{init_type}', 
                       color=colors[init_type], 
                       edgecolor='black', 
                       hatch=hatches.get(model, ''),
                       alpha=0.85,
                       yerr=stds, capsize=3, error_kw={'linewidth': 1.5, 'capthick': 1.5})
                bar_idx += 1
        
        ax.set_xlabel('Task Category', fontsize=10)
        ax.set_ylabel(f'Average {metric_name.upper()}', fontsize=10)
        ax.set_title(f'{metric_name.upper()}', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([p.capitalize() for p in prefixes])
        ax.grid(True, alpha=0.3, axis='y')
    
    # Add single legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=8)
    
    fig.suptitle(f'Performance by Task Category\n'
                 f'ML Form: {ml_form} | UI: {ui_mask} | Checkpoint: {checkpoint_type}',
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    filename_base = f'by_prefix_{ml_form}'
    save_plot_multi_format(fig, output_dir, filename_base)
    plt.close()


def plot_time_horizon_lines(
    df: pd.DataFrame,
    ml_form: str,
    ui_mask: str,
    checkpoint_type: str,
    output_dir: Path
):
    """
    Plot line plots for tasks with time horizon suffix (_0s, _10s, etc.).
    Creates a grid with rows for each task base and columns for each metric.
    Shows shaded std region for repeat experiments.
    
    Args:
        df: DataFrame filtered by ml_form and ui_mask
        ml_form: ML form type
        ui_mask: UI mask setting
        checkpoint_type: 'best' or 'last'
        output_dir: Directory to save plots
    """
    df = preprocess_dataframe(df)
    metrics = get_metrics_for_ml_form(ml_form)
    
    # Filter to metrics that exist in the dataframe
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        print(f"No metrics found for {ml_form}")
        return
    
    # Add time horizon column
    df = df.copy()
    df['time_horizon'] = df['task_id'].apply(extract_time_horizon)
    df['task_base'] = df['task_id'].apply(get_task_base_name)
    
    # Filter to tasks with time horizon
    df_temporal = df[df['time_horizon'].notna()].copy()
    
    if df_temporal.empty:
        print(f"No temporal tasks for {ml_form}, {ui_mask}")
        return
    
    # Get unique task bases
    task_bases = sorted(df_temporal['task_base'].unique())
    
    if len(task_bases) == 0:
        return
    
    models = sorted(df_temporal['model_type'].unique())
    init_types = [it for it in ['baseline', 'finetuned'] if it in df_temporal['init_type'].values]
    
    n_tasks = len(task_bases)
    n_metrics = len(available_metrics)
    
    # Create grid: rows = task bases, columns = metrics
    fig, axes = plt.subplots(n_tasks, n_metrics, figsize=(5 * n_metrics, 4 * n_tasks))
    
    if n_tasks == 1 and n_metrics == 1:
        axes = np.array([[axes]])
    elif n_tasks == 1:
        axes = axes.reshape(1, -1)
    elif n_metrics == 1:
        axes = axes.reshape(-1, 1)
    
    line_styles = {'baseline': '--', 'finetuned': '-'}
    markers = {'clip': 'o', 'dinov2': 's', 'siglip2': '^', 'vjepa2': 'D'}
    colors = {'clip': '#1f77b4', 'dinov2': '#2ca02c', 'siglip2': '#9467bd', 'vjepa2': '#8c564b'}
    
    for row_idx, task_base in enumerate(task_bases):
        df_task = df_temporal[df_temporal['task_base'] == task_base]
        
        for col_idx, metric_name in enumerate(available_metrics):
            ax = axes[row_idx, col_idx]
            
            for model in models:
                for init_type in init_types:
                    df_subset = df_task[(df_task['model_type'] == model) & 
                                       (df_task['init_type'] == init_type)]
                    
                    if df_subset.empty:
                        continue
                    
                    # Aggregate across repeats for each time horizon
                    horizons = []
                    means = []
                    stds = []
                    for horizon in sorted(df_subset['time_horizon'].unique()):
                        df_horizon = df_subset[df_subset['time_horizon'] == horizon]
                        mean_val, std_val, _ = _aggregate_repeat_experiments(df_horizon, metric_name)
                        horizons.append(horizon)
                        means.append(mean_val)
                        stds.append(std_val if not np.isnan(std_val) else 0)
                    
                    horizons = np.array(horizons)
                    means = np.array(means)
                    stds = np.array(stds)
                    
                    color = colors.get(model, 'gray')
                    
                    # Plot line with markers
                    ax.plot(horizons, means, 
                           linestyle=line_styles.get(init_type, '-'),
                           marker=markers.get(model, 'o'),
                           color=color,
                           label=f'{model}-{init_type}',
                           markersize=5,
                           linewidth=1.5,
                           alpha=0.9)
                    
                    # Add shaded std region
                    ax.fill_between(horizons, means - stds, means + stds,
                                   color=color, alpha=0.15)
            
            ax.set_xlabel('Time Horizon (s)', fontsize=9)
            ax.set_ylabel(f'{metric_name.upper()}', fontsize=9)
            
            # Set title: task name on first row, metric on all
            if row_idx == 0:
                ax.set_title(f'{metric_name.upper()}', fontsize=10, fontweight='bold')
            
            # Add task name as y-label on first column
            if col_idx == 0:
                ax.annotate(task_base, xy=(-0.3, 0.5), xycoords='axes fraction',
                           fontsize=10, fontweight='bold', rotation=90,
                           ha='center', va='center')
            
            ax.grid(True, alpha=0.3)
            
            # Set x-ticks to actual horizon values
            all_horizons = sorted(df_task['time_horizon'].unique())
            ax.set_xticks(all_horizons)
    
    # Add single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=8)
    
    fig.suptitle(f'Performance vs Time Horizon\n'
                 f'ML Form: {ml_form} | UI: {ui_mask} | Checkpoint: {checkpoint_type}',
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    filename_base = f'time_horizon_{ml_form}'
    save_plot_multi_format(fig, output_dir, filename_base)
    plt.close()
