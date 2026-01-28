#!/usr/bin/env python3
"""
Additional Plotting Utilities for Multi-Task Learning Narrative

Specialized plots for visualizing the Global↑ vs Self↓ trade-off in 33 downstream tasks.
These plots handle metric heterogeneity (Accuracy, MSE, F1) through normalization.

Plots included:
1. Normalized Improvement by Perspective (Box/Violin Plot)
2. Baseline vs Finetuned Scatter Plot with Identity Line
3. Heatmap Strip: Abstract Summary of Relative Improvement
"""

from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
import numpy as np

from plotting_utils import (
    preprocess_dataframe,
    get_task_prefix,
    get_primary_metric_for_ml_form,
    _aggregate_repeat_experiments,
    save_plot_multi_format,
)


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

# Perspective display names and colors
PERSPECTIVE_DISPLAY = {
    'self': 'Self',
    'teammate': 'Teammate', 
    'enemy': 'Enemy',
    'global': 'Global'
}

PERSPECTIVE_COLORS = {
    'self': '#e74c3c',      # Red
    'teammate': '#f39c12',  # Orange
    'enemy': '#9b59b6',     # Purple
    'global': '#3498db',    # Blue
}

# ML form markers for scatter plot
ML_FORM_MARKERS = {
    'binary_cls': 'o',
    'multi_cls': 's',
    'multi_label_cls': '^',
    'regression': 'D',
}

# Metrics where lower is better (need sign inversion for improvement calculation)
LOWER_IS_BETTER_METRICS = {'mse', 'mae', 'hamming_dist'}


def _get_primary_metric_value(
    df: pd.DataFrame,
    task_id: str,
    model_type: str,
    init_type: str,
    ml_form: str
) -> Tuple[float, float]:
    """
    Get the primary metric value for a specific task/model/init combination.
    
    Args:
        df: DataFrame with results
        task_id: Task identifier
        model_type: Model type (clip, dinov2, etc.)
        init_type: 'baseline' or 'finetuned'
        ml_form: ML form type
    
    Returns:
        Tuple of (mean, std) for the primary metric
    """
    df_subset = df[
        (df['task_id'] == task_id) &
        (df['model_type'] == model_type) &
        (df['init_type'] == init_type)
    ]
    
    primary_metric = get_primary_metric_for_ml_form(ml_form)
    
    if df_subset.empty or primary_metric not in df_subset.columns:
        return np.nan, np.nan
    
    mean_val, std_val, _ = _aggregate_repeat_experiments(df_subset, primary_metric)
    return mean_val, std_val


def _get_primary_metric_values_individual(
    df: pd.DataFrame,
    task_id: str,
    model_type: str,
    init_type: str,
    ml_form: str
) -> List[float]:
    """
    Get all individual primary metric values for a specific task/model/init combination.
    
    Args:
        df: DataFrame with results
        task_id: Task identifier
        model_type: Model type (clip, dinov2, etc.)
        init_type: 'baseline' or 'finetuned'
        ml_form: ML form type
    
    Returns:
        List of individual metric values (one per repeat)
    """
    df_subset = df[
        (df['task_id'] == task_id) &
        (df['model_type'] == model_type) &
        (df['init_type'] == init_type)
    ]
    
    primary_metric = get_primary_metric_for_ml_form(ml_form)
    
    if df_subset.empty or primary_metric not in df_subset.columns:
        return []
    
    return df_subset[primary_metric].dropna().tolist()


def compute_relative_improvement(
    baseline_val: float,
    finetuned_val: float,
    metric_name: str
) -> float:
    """
    Compute relative percentage improvement.
    
    For metrics where lower is better (MSE, MAE), invert the sign so
    positive always means "better".
    
    Formula: ((Finetuned - Baseline) / |Baseline|) * 100
    For lower-is-better: ((Baseline - Finetuned) / |Baseline|) * 100
    
    Args:
        baseline_val: Baseline metric value
        finetuned_val: Finetuned metric value
        metric_name: Name of the metric
    
    Returns:
        Relative improvement percentage (positive = better)
    """
    if np.isnan(baseline_val) or np.isnan(finetuned_val):
        return np.nan
    
    if abs(baseline_val) < 1e-10:
        return np.nan
    
    if metric_name in LOWER_IS_BETTER_METRICS:
        # Lower is better: positive improvement when finetuned < baseline
        improvement = ((baseline_val - finetuned_val) / abs(baseline_val)) * 100
    else:
        # Higher is better: positive improvement when finetuned > baseline
        improvement = ((finetuned_val - baseline_val) / abs(baseline_val)) * 100
    
    return improvement


def compute_improvement_dataframe(
    df: pd.DataFrame,
    model_type: Optional[str] = None,
    aggregate_repeats: bool = True
) -> pd.DataFrame:
    """
    Compute relative improvement for all tasks.
    
    Args:
        df: DataFrame with results (from ResultsCollector.create_results_dataframe)
        model_type: If specified, filter to this model only
        aggregate_repeats: If True, aggregate repeats to mean. If False, return
                          individual rows for each repeat experiment.
    
    Returns:
        DataFrame with columns: task_id, perspective, ml_form, improvement_pct, 
                               baseline_val, finetuned_val, metric_name
                               (plus repeat_idx if aggregate_repeats=False)
    """
    df = preprocess_dataframe(df)
    
    if model_type:
        df = df[df['model_type'] == model_type]
    
    tasks = df['task_id'].unique()
    models = df['model_type'].unique() if model_type is None else [model_type]
    
    rows = []
    
    for task_id in tasks:
        df_task = df[df['task_id'] == task_id]
        
        if df_task.empty:
            continue
        
        ml_form = df_task['ml_form'].iloc[0]
        primary_metric = get_primary_metric_for_ml_form(ml_form)
        perspective = get_task_prefix(task_id)
        
        if perspective is None:
            continue
        
        for model in models:
            if aggregate_repeats:
                # Aggregated mode: one row per task/model with mean values
                baseline_val, baseline_std = _get_primary_metric_value(
                    df, task_id, model, 'baseline', ml_form
                )
                finetuned_val, finetuned_std = _get_primary_metric_value(
                    df, task_id, model, 'finetuned', ml_form
                )
                
                improvement = compute_relative_improvement(
                    baseline_val, finetuned_val, primary_metric
                )
                
                rows.append({
                    'task_id': task_id,
                    'model_type': model,
                    'perspective': perspective,
                    'ml_form': ml_form,
                    'metric_name': primary_metric,
                    'baseline_val': baseline_val,
                    'finetuned_val': finetuned_val,
                    'improvement_pct': improvement,
                })
            else:
                # Individual mode: one row per repeat experiment
                baseline_vals = _get_primary_metric_values_individual(
                    df, task_id, model, 'baseline', ml_form
                )
                finetuned_vals = _get_primary_metric_values_individual(
                    df, task_id, model, 'finetuned', ml_form
                )
                
                # Pair up baseline and finetuned values by index
                # If counts differ, use min count (paired comparison)
                n_pairs = min(len(baseline_vals), len(finetuned_vals))
                
                for repeat_idx in range(n_pairs):
                    baseline_val = baseline_vals[repeat_idx]
                    finetuned_val = finetuned_vals[repeat_idx]
                    
                    improvement = compute_relative_improvement(
                        baseline_val, finetuned_val, primary_metric
                    )
                    
                    rows.append({
                        'task_id': task_id,
                        'model_type': model,
                        'perspective': perspective,
                        'ml_form': ml_form,
                        'metric_name': primary_metric,
                        'baseline_val': baseline_val,
                        'finetuned_val': finetuned_val,
                        'improvement_pct': improvement,
                        'repeat_idx': repeat_idx,
                    })
    
    return pd.DataFrame(rows)


def plot_improvement_by_perspective_boxplot(
    df: pd.DataFrame,
    ui_mask: str,
    checkpoint_type: str,
    output_dir: Path,
    model_type: Optional[str] = None,
    use_violin: bool = False,
    aggregate_repeats: bool = True
):
    """
    Plot 1: Normalized Improvement by Perspective (Box/Violin Plot)
    
    Directly proves the hypothesis: Global gains, Self losses.
    X-axis: 4 Perspectives (Self, Teammate, Enemy, Global)
    Y-axis: % Relative Improvement over Baseline
    
    Args:
        df: DataFrame with results
        ui_mask: UI mask setting
        checkpoint_type: 'best' or 'last'
        output_dir: Directory to save plots
        model_type: If specified, filter to this model only
        use_violin: If True, use violin plot instead of box plot
        aggregate_repeats: If True, each point is mean across repeats.
                          If False, each point is an individual run.
    """
    improvement_df = compute_improvement_dataframe(df, model_type, aggregate_repeats=aggregate_repeats)
    
    if improvement_df.empty:
        print("No data for improvement by perspective plot")
        return
    
    # Filter out NaN improvements
    improvement_df = improvement_df.dropna(subset=['improvement_pct'])
    
    if improvement_df.empty:
        print("No valid improvement data")
        return
    
    # Order perspectives
    perspective_order = ['self', 'teammate', 'enemy', 'global']
    improvement_df['perspective'] = pd.Categorical(
        improvement_df['perspective'],
        categories=perspective_order,
        ordered=True
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color palette
    palette = [PERSPECTIVE_COLORS[p] for p in perspective_order]
    
    if use_violin:
        sns.violinplot(
            data=improvement_df,
            x='perspective',
            y='improvement_pct',
            palette=palette,
            ax=ax,
            inner='box',
            cut=0
        )
    else:
        sns.boxplot(
            data=improvement_df,
            x='perspective',
            y='improvement_pct',
            palette=palette,
            ax=ax,
            width=0.6,
            flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.5}
        )
        
        # Add individual points (strip plot)
        sns.stripplot(
            data=improvement_df,
            x='perspective',
            y='improvement_pct',
            color='black',
            alpha=0.4,
            size=5,
            ax=ax,
            jitter=True
        )
    
    # Add horizontal line at y=0 (no change)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.8, zorder=0)
    
    # Labels and title
    ax.set_xlabel('Perspective', fontsize=12, fontweight='bold')
    ax.set_ylabel('Relative Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_xticklabels([PERSPECTIVE_DISPLAY[p] for p in perspective_order], fontsize=11)
    
    model_str = f' ({model_type.upper()})' if model_type else ' (All Models)'
    agg_str = ' [Aggregated]' if aggregate_repeats else ' [Individual Runs]'
    title = f'Contrastive Pre-training: Improvement by Perspective{model_str}{agg_str}'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    # Add annotation
    n_points = len(improvement_df)
    annotation_text = f'Above line = Improvement\nBelow line = Degradation\nn = {n_points} {"(mean per task)" if aggregate_repeats else "(individual runs)"}'
    ax.annotate(
        annotation_text,
        xy=(0.02, 0.98), xycoords='axes fraction',
        fontsize=9, ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save
    plot_type = 'violin' if use_violin else 'boxplot'
    model_suffix = f'_{model_type}' if model_type else '_all'
    agg_suffix = '_aggregated' if aggregate_repeats else '_individual'
    filename_base = f'improvement_by_perspective_{plot_type}{model_suffix}{agg_suffix}'
    save_plot_multi_format(fig, output_dir, filename_base)
    plt.close()


def plot_baseline_vs_finetuned_scatter(
    df: pd.DataFrame,
    ui_mask: str,
    checkpoint_type: str,
    output_dir: Path,
    model_type: Optional[str] = None,
    aggregate_repeats: bool = True
):
    """
    Plot 2: Baseline vs Finetuned Scatter Plot with Identity Line
    
    Shows which tasks improve vs degrade without hiding complexity.
    X-axis: Baseline Performance (normalized 0-1)
    Y-axis: Finetuned Performance (normalized 0-1)
    Points above y=x line = improvement, below = degradation
    
    Color by perspective, shape by ML form.
    
    Args:
        df: DataFrame with results
        ui_mask: UI mask setting
        checkpoint_type: 'best' or 'last'
        output_dir: Directory to save plots
        model_type: If specified, filter to this model only
        aggregate_repeats: If True, each point is mean across repeats.
                          If False, each point is an individual run.
    """
    improvement_df = compute_improvement_dataframe(df, model_type, aggregate_repeats=aggregate_repeats)
    
    if improvement_df.empty:
        print("No data for scatter plot")
        return
    
    # Filter out NaN values
    improvement_df = improvement_df.dropna(subset=['baseline_val', 'finetuned_val'])
    
    if improvement_df.empty:
        print("No valid data for scatter plot")
        return
    
    # Normalize values to 0-1 range per ML form
    # For regression metrics (lower is better), we need to handle differently
    improvement_df['baseline_norm'] = np.nan
    improvement_df['finetuned_norm'] = np.nan
    
    for ml_form in improvement_df['ml_form'].unique():
        mask = improvement_df['ml_form'] == ml_form
        df_ml = improvement_df[mask]
        
        metric_name = df_ml['metric_name'].iloc[0]
        
        # Get all values for normalization
        all_vals = pd.concat([df_ml['baseline_val'], df_ml['finetuned_val']]).dropna()
        
        if len(all_vals) == 0:
            continue
        
        min_val = all_vals.min()
        max_val = all_vals.max()
        
        if max_val - min_val < 1e-10:
            # All values are the same
            improvement_df.loc[mask, 'baseline_norm'] = 0.5
            improvement_df.loc[mask, 'finetuned_norm'] = 0.5
        else:
            # Normalize
            baseline_norm = (df_ml['baseline_val'] - min_val) / (max_val - min_val)
            finetuned_norm = (df_ml['finetuned_val'] - min_val) / (max_val - min_val)
            
            # For lower-is-better metrics, invert so higher is still better visually
            if metric_name in LOWER_IS_BETTER_METRICS:
                baseline_norm = 1 - baseline_norm
                finetuned_norm = 1 - finetuned_norm
            
            improvement_df.loc[mask, 'baseline_norm'] = baseline_norm.values
            improvement_df.loc[mask, 'finetuned_norm'] = finetuned_norm.values
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot identity line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='No Change (y=x)', zorder=1)
    
    # Fill regions
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.1, color='green', label='Improvement Region')
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.1, color='red', label='Degradation Region')
    
    # Adjust marker size based on aggregation mode
    marker_size = 100 if aggregate_repeats else 60
    marker_alpha = 0.7 if aggregate_repeats else 0.5
    
    # Plot points by perspective and ML form
    for perspective in ['self', 'teammate', 'enemy', 'global']:
        for ml_form in improvement_df['ml_form'].unique():
            mask = (improvement_df['perspective'] == perspective) & (improvement_df['ml_form'] == ml_form)
            df_subset = improvement_df[mask]
            
            if df_subset.empty:
                continue
            
            ax.scatter(
                df_subset['baseline_norm'],
                df_subset['finetuned_norm'],
                c=PERSPECTIVE_COLORS.get(perspective, 'gray'),
                marker=ML_FORM_MARKERS.get(ml_form, 'o'),
                s=marker_size,
                alpha=marker_alpha,
                edgecolors='black',
                linewidths=0.5,
                label=f'{PERSPECTIVE_DISPLAY.get(perspective, perspective)} ({ml_form})',
                zorder=2
            )
    
    # Labels and title
    ax.set_xlabel('Baseline Performance (Normalized)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Finetuned Performance (Normalized)', fontsize=12, fontweight='bold')
    
    model_str = f' ({model_type.upper()})' if model_type else ' (All Models)'
    agg_str = ' [Aggregated]' if aggregate_repeats else ' [Individual Runs]'
    ax.set_title(f'Baseline vs Finetuned Performance{model_str}{agg_str}', fontsize=13, fontweight='bold', pad=15)
    
    # Set axis limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    
    # Create custom legend
    # Perspective colors
    perspective_handles = [
        mpatches.Patch(color=PERSPECTIVE_COLORS[p], label=PERSPECTIVE_DISPLAY[p])
        for p in ['self', 'teammate', 'enemy', 'global']
    ]
    
    # ML form markers
    ml_form_handles = [
        plt.Line2D([0], [0], marker=ML_FORM_MARKERS[ml], color='gray', 
                   linestyle='', markersize=10, label=ml.replace('_', ' ').title())
        for ml in ML_FORM_MARKERS.keys()
    ]
    
    # Region legend
    region_handles = [
        mpatches.Patch(color='green', alpha=0.3, label='Improvement'),
        mpatches.Patch(color='red', alpha=0.3, label='Degradation'),
    ]
    
    # Add count annotation
    n_points = len(improvement_df)
    count_text = f'n = {n_points} {"(mean per task)" if aggregate_repeats else "(individual runs)"}'
    ax.annotate(
        count_text,
        xy=(0.02, 0.02), xycoords='axes fraction',
        fontsize=9, ha='left', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )
    
    # Combine legends
    first_legend = ax.legend(
        handles=perspective_handles,
        title='Perspective',
        loc='upper left',
        fontsize=9,
        title_fontsize=10
    )
    ax.add_artist(first_legend)
    
    second_legend = ax.legend(
        handles=ml_form_handles,
        title='ML Form',
        loc='lower right',
        fontsize=9,
        title_fontsize=10
    )
    ax.add_artist(second_legend)
    
    ax.legend(
        handles=region_handles,
        title='Region',
        loc='center right',
        fontsize=9,
        title_fontsize=10
    )
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    model_suffix = f'_{model_type}' if model_type else '_all'
    agg_suffix = '_aggregated' if aggregate_repeats else '_individual'
    filename_base = f'baseline_vs_finetuned_scatter{model_suffix}{agg_suffix}'
    save_plot_multi_format(fig, output_dir, filename_base)
    plt.close()


def plot_improvement_heatmap_strip(
    df: pd.DataFrame,
    ui_mask: str,
    checkpoint_type: str,
    output_dir: Path,
    model_type: Optional[str] = None
):
    """
    Plot 3: Heatmap Strip - Abstract Summary of Relative Improvement
    
    A condensed visual replacement for a table.
    Rows: Task names (grouped by perspective)
    Column: Relative Improvement (single column)
    Color: Diverging colormap (Red = degradation, White = parity, Blue = improvement)
    
    Args:
        df: DataFrame with results
        ui_mask: UI mask setting
        checkpoint_type: 'best' or 'last'
        output_dir: Directory to save plots
        model_type: If specified, filter to this model only
    """
    improvement_df = compute_improvement_dataframe(df, model_type)
    
    if improvement_df.empty:
        print("No data for heatmap strip")
        return
    
    # If multiple models, aggregate by taking mean across models
    if model_type is None:
        improvement_df = improvement_df.groupby(
            ['task_id', 'perspective', 'ml_form', 'metric_name']
        ).agg({
            'improvement_pct': 'mean',
            'baseline_val': 'mean',
            'finetuned_val': 'mean'
        }).reset_index()
    
    # Filter out NaN
    improvement_df = improvement_df.dropna(subset=['improvement_pct'])
    
    if improvement_df.empty:
        print("No valid data for heatmap strip")
        return
    
    # Sort by perspective order, then by task_id
    perspective_order = ['global', 'enemy', 'teammate', 'self']  # Reversed for visual
    improvement_df['perspective_order'] = improvement_df['perspective'].map(
        {p: i for i, p in enumerate(perspective_order)}
    )
    improvement_df = improvement_df.sort_values(
        ['perspective_order', 'task_id'],
        ascending=[True, True]
    ).reset_index(drop=True)
    
    # Create heatmap data
    tasks = improvement_df['task_id'].tolist()
    improvements = improvement_df['improvement_pct'].values
    perspectives = improvement_df['perspective'].tolist()
    
    # Create figure with appropriate height
    n_tasks = len(tasks)
    fig_height = max(8, n_tasks * 0.35)
    fig, ax = plt.subplots(figsize=(6, fig_height))
    
    # Create heatmap matrix (n_tasks x 1)
    heatmap_data = improvements.reshape(-1, 1)
    
    # Determine color scale (symmetric around 0)
    max_abs = max(abs(np.nanmin(improvements)), abs(np.nanmax(improvements)))
    max_abs = max(max_abs, 10)  # At least ±10%
    
    # Create diverging colormap centered at 0
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    
    # Plot heatmap
    im = ax.imshow(
        heatmap_data,
        cmap='RdBu',
        norm=norm,
        aspect='auto'
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Relative Improvement (%)', fontsize=10, fontweight='bold')
    
    # Set y-axis labels (task names)
    ax.set_yticks(range(n_tasks))
    
    # Create labels with perspective prefix
    task_labels = []
    current_perspective = None
    for i, (task, persp) in enumerate(zip(tasks, perspectives)):
        # Add perspective header
        if persp != current_perspective:
            current_perspective = persp
        task_labels.append(task)
    
    ax.set_yticklabels(task_labels, fontsize=8)
    
    # Add perspective separators and labels
    current_perspective = None
    separator_positions = []
    perspective_positions = {}
    
    for i, persp in enumerate(perspectives):
        if persp != current_perspective:
            if current_perspective is not None:
                separator_positions.append(i - 0.5)
            perspective_positions[persp] = i
            current_perspective = persp
    
    # Draw separators
    for pos in separator_positions:
        ax.axhline(y=pos, color='black', linewidth=2)
    
    # Add perspective labels on the left
    for persp, start_idx in perspective_positions.items():
        # Find end of this perspective
        end_idx = start_idx
        for i in range(start_idx, n_tasks):
            if perspectives[i] == persp:
                end_idx = i
            else:
                break
        
        mid_idx = (start_idx + end_idx) / 2
        ax.annotate(
            PERSPECTIVE_DISPLAY.get(persp, persp).upper(),
            xy=(-0.15, mid_idx),
            xycoords=('axes fraction', 'data'),
            fontsize=10,
            fontweight='bold',
            color=PERSPECTIVE_COLORS.get(persp, 'black'),
            ha='right',
            va='center',
            rotation=0
        )
    
    # Remove x-axis
    ax.set_xticks([])
    ax.set_xlabel('')
    
    # Title
    model_str = f' ({model_type.upper()})' if model_type else ' (All Models Avg)'
    ax.set_title(
        f'Task-Level Improvement Summary{model_str}\n'
        f'UI: {ui_mask} | Checkpoint: {checkpoint_type}',
        fontsize=12,
        fontweight='bold',
        pad=15
    )
    
    # Add value annotations
    for i, (task, imp) in enumerate(zip(tasks, improvements)):
        if not np.isnan(imp):
            text_color = 'white' if abs(imp) > max_abs * 0.6 else 'black'
            ax.annotate(
                f'{imp:+.1f}%',
                xy=(0, i),
                ha='center',
                va='center',
                fontsize=8,
                fontweight='bold',
                color=text_color
            )
    
    plt.tight_layout()
    
    # Save
    model_suffix = f'_{model_type}' if model_type else '_all'
    filename_base = f'improvement_heatmap_strip{model_suffix}'
    save_plot_multi_format(fig, output_dir, filename_base)
    plt.close()


def plot_improvement_by_perspective_grouped_bar(
    df: pd.DataFrame,
    ui_mask: str,
    checkpoint_type: str,
    output_dir: Path
):
    """
    Plot 4: Grouped Bar Chart - Improvement by Perspective per Model
    
    Shows improvement breakdown by model type for each perspective.
    Useful for comparing which models benefit most from contrastive pre-training.
    
    Args:
        df: DataFrame with results
        ui_mask: UI mask setting
        checkpoint_type: 'best' or 'last'
        output_dir: Directory to save plots
    """
    improvement_df = compute_improvement_dataframe(df, model_type=None)
    
    if improvement_df.empty:
        print("No data for grouped bar chart")
        return
    
    # Filter out NaN
    improvement_df = improvement_df.dropna(subset=['improvement_pct'])
    
    if improvement_df.empty:
        print("No valid data for grouped bar chart")
        return
    
    # Aggregate by perspective and model
    agg_df = improvement_df.groupby(['perspective', 'model_type']).agg({
        'improvement_pct': ['mean', 'std', 'count']
    }).reset_index()
    agg_df.columns = ['perspective', 'model_type', 'mean', 'std', 'count']
    
    # Order perspectives
    perspective_order = ['self', 'teammate', 'enemy', 'global']
    agg_df['perspective'] = pd.Categorical(
        agg_df['perspective'],
        categories=perspective_order,
        ordered=True
    )
    agg_df = agg_df.sort_values(['perspective', 'model_type'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = sorted(agg_df['model_type'].unique())
    n_models = len(models)
    x = np.arange(len(perspective_order))
    width = 0.8 / n_models
    
    # Model colors
    model_colors = {
        'clip': '#1f77b4',
        'dinov2': '#2ca02c',
        'siglip2': '#9467bd',
        'vjepa2': '#8c564b'
    }
    
    for i, model in enumerate(models):
        df_model = agg_df[agg_df['model_type'] == model]
        
        means = []
        stds = []
        for persp in perspective_order:
            row = df_model[df_model['perspective'] == persp]
            if not row.empty:
                means.append(row['mean'].values[0])
                stds.append(row['std'].values[0] if not np.isnan(row['std'].values[0]) else 0)
            else:
                means.append(0)
                stds.append(0)
        
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            means,
            width,
            label=model.upper(),
            color=model_colors.get(model, 'gray'),
            edgecolor='black',
            alpha=0.85,
            yerr=stds,
            capsize=3,
            error_kw={'linewidth': 1.5, 'capthick': 1.5}
        )
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.8)
    
    # Labels
    ax.set_xlabel('Perspective', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Relative Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([PERSPECTIVE_DISPLAY[p] for p in perspective_order], fontsize=11)
    ax.set_title(
        f'Improvement by Perspective and Model\n'
        f'UI: {ui_mask} | Checkpoint: {checkpoint_type}',
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    
    ax.legend(title='Model', loc='upper left', fontsize=9, title_fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    filename_base = 'improvement_by_perspective_grouped_bar'
    save_plot_multi_format(fig, output_dir, filename_base)
    plt.close()


def generate_all_narrative_plots(
    df: pd.DataFrame,
    ui_mask: str,
    checkpoint_type: str,
    output_dir: Path,
    models: Optional[List[str]] = None
):
    """
    Generate all narrative plots for the multi-task learning paper.
    
    Args:
        df: DataFrame with results
        ui_mask: UI mask setting
        checkpoint_type: 'best' or 'last'
        output_dir: Directory to save plots
        models: List of models to generate individual plots for (default: all + aggregate)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Generating Narrative Plots")
    print(f"UI: {ui_mask} | Checkpoint: {checkpoint_type}")
    print(f"{'='*60}\n")
    
    # Default models
    if models is None:
        models = df['model_type'].unique().tolist()
    
    # 1. Box/Violin plots - aggregate and per-model
    # Generate BOTH aggregated and individual versions
    print("\n--- Plot 1: Improvement by Perspective (Box/Violin) ---")
    for aggregate in [True, False]:
        agg_label = "aggregated" if aggregate else "individual"
        print(f"  Generating {agg_label} versions...")
        
        plot_improvement_by_perspective_boxplot(
            df, ui_mask, checkpoint_type, output_dir, model_type=None, 
            use_violin=False, aggregate_repeats=aggregate
        )
        plot_improvement_by_perspective_boxplot(
            df, ui_mask, checkpoint_type, output_dir, model_type=None, 
            use_violin=True, aggregate_repeats=aggregate
        )
        
        for model in models:
            plot_improvement_by_perspective_boxplot(
                df, ui_mask, checkpoint_type, output_dir, model_type=model, 
                use_violin=False, aggregate_repeats=aggregate
            )
    
    # 2. Scatter plots - BOTH aggregated and individual versions
    print("\n--- Plot 2: Baseline vs Finetuned Scatter ---")
    for aggregate in [True, False]:
        agg_label = "aggregated" if aggregate else "individual"
        print(f"  Generating {agg_label} versions...")
        
        plot_baseline_vs_finetuned_scatter(
            df, ui_mask, checkpoint_type, output_dir, model_type=None,
            aggregate_repeats=aggregate
        )
        
        for model in models:
            plot_baseline_vs_finetuned_scatter(
                df, ui_mask, checkpoint_type, output_dir, model_type=model,
                aggregate_repeats=aggregate
            )
    
    # 3. Heatmap strips (only aggregated makes sense for this visualization)
    print("\n--- Plot 3: Improvement Heatmap Strip ---")
    plot_improvement_heatmap_strip(
        df, ui_mask, checkpoint_type, output_dir, model_type=None
    )
    
    for model in models:
        plot_improvement_heatmap_strip(
            df, ui_mask, checkpoint_type, output_dir, model_type=model
        )
    
    # 4. Grouped bar chart (only aggregated makes sense)
    print("\n--- Plot 4: Grouped Bar Chart by Model ---")
    plot_improvement_by_perspective_grouped_bar(
        df, ui_mask, checkpoint_type, output_dir
    )
    
    print(f"\n{'='*60}")
    print(f"All narrative plots saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Example usage
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from results_collector import ResultsCollector
    
    project_root = Path(__file__).parent.parent.parent.parent
    output_dir = project_root / 'output'
    task_defs = project_root / 'data' / 'labels' / 'task_definitions.csv'
    artifacts_dir = project_root / 'artifacts' / 'results' / 'narrative'
    
    if not task_defs.exists():
        print(f"Error: Task definitions not found at {task_defs}")
        sys.exit(1)
    
    if not output_dir.exists():
        print(f"Error: Output directory not found at {output_dir}")
        sys.exit(1)
    
    # Collect results
    print("Collecting results...")
    collector = ResultsCollector(output_dir, task_defs)
    all_results = collector.collect_all_results()
    
    if not all_results:
        print("No results found!")
        sys.exit(1)
    
    print(f"Found {len(all_results)} experiments")
    
    # Create dataframe for best checkpoint
    df = collector.create_results_dataframe(all_results, 'best')
    
    # Filter to a specific UI mask if needed
    ui_masks = df['ui_mask'].unique()
    print(f"Available UI masks: {list(ui_masks)}")
    
    for ui_mask in ui_masks:
        df_filtered = df[df['ui_mask'] == ui_mask]
        
        if df_filtered.empty:
            continue
        
        subfolder = artifacts_dir / ui_mask
        generate_all_narrative_plots(
            df_filtered,
            ui_mask=ui_mask,
            checkpoint_type='best',
            output_dir=subfolder
        )
    
    print("\nDone!")
