#!/usr/bin/env python3
"""
Plotting Utilities

Utility functions for creating various plots from results data.
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10


def plot_performance_by_category(
    df: pd.DataFrame,
    checkpoint_type: str,
    output_path: Path,
    metric_name: str = 'acc'
):
    """
    Plot performance grouped by task category.
    
    Args:
        df: DataFrame with results
        checkpoint_type: 'best' or 'last'
        output_path: Path to save the plot
        metric_name: Metric to plot (default: 'acc')
    """
    if df.empty or metric_name not in df.columns:
        print(f"Cannot plot {metric_name} by category - data not available")
        return
    
    # Filter rows that have the metric
    df_plot = df[df[metric_name].notna()].copy()
    
    if df_plot.empty:
        print(f"No data for metric {metric_name}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by category
    categories = df_plot['category'].unique()
    
    # Create bar plot
    sns.barplot(
        data=df_plot,
        x='category',
        y=metric_name,
        hue='model_type',
        ax=ax,
        palette='Set2'
    )
    
    ax.set_title(f'Performance by Category - {checkpoint_type.upper()} ({metric_name.upper()})', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel(metric_name.upper(), fontsize=12)
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_temporal_horizon_performance(
    df: pd.DataFrame,
    checkpoint_type: str,
    output_path: Path,
    metric_name: str = 'acc'
):
    """
    Plot performance vs temporal horizon for forecast tasks.
    
    Args:
        df: DataFrame with results
        checkpoint_type: 'best' or 'last'
        output_path: Path to save the plot
        metric_name: Metric to plot
    """
    if df.empty or metric_name not in df.columns:
        print(f"Cannot plot temporal horizon - data not available")
        return
    
    # Filter forecast tasks only
    df_forecast = df[df['temporal_type'] == 'forecast'].copy()
    
    if df_forecast.empty:
        print("No forecast tasks found")
        return
    
    # Filter rows with valid metric
    df_forecast = df_forecast[df_forecast[metric_name].notna()]
    
    if df_forecast.empty:
        print(f"No forecast tasks with {metric_name}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by category and plot
    for category in df_forecast['category'].unique():
        df_cat = df_forecast[df_forecast['category'] == category]
        
        # Sort by horizon
        df_cat = df_cat.sort_values('horizon_sec')
        
        ax.plot(
            df_cat['horizon_sec'],
            df_cat[metric_name],
            marker='o',
            label=category,
            linewidth=2,
            markersize=8
        )
    
    ax.set_title(f'Performance vs Temporal Horizon - {checkpoint_type.upper()} ({metric_name.upper()})',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Horizon (seconds)', fontsize=12)
    ax.set_ylabel(metric_name.upper(), fontsize=12)
    ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_metric_distributions(
    df: pd.DataFrame,
    checkpoint_type: str,
    output_path: Path
):
    """
    Plot distributions of all metrics across tasks.
    
    Args:
        df: DataFrame with results
        checkpoint_type: 'best' or 'last'
        output_path: Path to save the plot
    """
    if df.empty:
        print("No data to plot")
        return
    
    # Get all metric columns (numeric columns excluding metadata)
    exclude_cols = {'num_classes', 'output_dim', 'horizon_sec'}
    metric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in exclude_cols
    ]
    
    if not metric_cols:
        print("No metrics found")
        return
    
    # Create subplots
    n_metrics = len(metric_cols)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, metric in enumerate(metric_cols):
        ax = axes[idx]
        
        # Filter valid values
        values = df[metric].dropna()
        
        if len(values) == 0:
            ax.text(0.5, 0.5, f'No data for {metric}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Histogram
        ax.hist(values, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {values.mean():.3f}')
        ax.axvline(values.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {values.median():.3f}')
        
        ax.set_title(f'{metric.upper()} Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel(metric.upper(), fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'Metric Distributions - {checkpoint_type.upper()}', 
                 fontsize=16, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_category_heatmap(
    df: pd.DataFrame,
    checkpoint_type: str,
    output_path: Path,
    metric_name: str = 'acc'
):
    """
    Plot heatmap of performance across categories and tasks.
    
    Args:
        df: DataFrame with results
        checkpoint_type: 'best' or 'last'
        output_path: Path to save the plot
        metric_name: Metric to plot
    """
    if df.empty or metric_name not in df.columns:
        print(f"Cannot plot heatmap - data not available")
        return
    
    # Filter valid data
    df_plot = df[df[metric_name].notna()].copy()
    
    if df_plot.empty:
        print(f"No data for {metric_name}")
        return
    
    # Create pivot table
    pivot_data = df_plot.pivot_table(
        values=metric_name,
        index='category',
        columns='task_id',
        aggfunc='mean'
    )
    
    if pivot_data.empty:
        print("Cannot create pivot table")
        return
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(12, len(pivot_data.columns) * 0.5), 
                                     max(6, len(pivot_data.index) * 0.5)))
    
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=pivot_data.values.mean(),
        cbar_kws={'label': metric_name.upper()},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_title(f'Performance Heatmap by Category and Task - {checkpoint_type.upper()} ({metric_name.upper()})',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Task ID', fontsize=12)
    ax.set_ylabel('Category', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_ml_form_comparison(
    df: pd.DataFrame,
    checkpoint_type: str,
    output_path: Path
):
    """
    Plot comparison of task counts and average performance by ML form.
    
    Args:
        df: DataFrame with results
        checkpoint_type: 'best' or 'last'
        output_path: Path to save the plot
    """
    if df.empty:
        print("No data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Task counts by ML form
    ml_form_counts = df['ml_form'].value_counts()
    
    ax1.bar(range(len(ml_form_counts)), ml_form_counts.values, color='steelblue', edgecolor='black')
    ax1.set_xticks(range(len(ml_form_counts)))
    ax1.set_xticklabels(ml_form_counts.index, rotation=45, ha='right')
    ax1.set_ylabel('Number of Tasks', fontsize=12)
    ax1.set_title('Task Count by ML Form', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for i, v in enumerate(ml_form_counts.values):
        ax1.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Average primary metric by ML form
    # Determine primary metric for each ML form
    metric_map = {
        'binary_cls': 'acc',
        'multi_cls': 'acc',
        'multi_label_cls': 'acc_subset',
        'regression': 'mae'
    }
    
    ml_forms = []
    avg_metrics = []
    
    for ml_form in df['ml_form'].unique():
        df_ml = df[df['ml_form'] == ml_form]
        primary_metric = metric_map.get(ml_form, 'acc')
        
        if primary_metric in df_ml.columns:
            avg_val = df_ml[primary_metric].mean()
            ml_forms.append(ml_form)
            avg_metrics.append(avg_val)
    
    if ml_forms:
        ax2.bar(range(len(ml_forms)), avg_metrics, color='coral', edgecolor='black')
        ax2.set_xticks(range(len(ml_forms)))
        ax2.set_xticklabels(ml_forms, rotation=45, ha='right')
        ax2.set_ylabel('Average Primary Metric', fontsize=12)
        ax2.set_title('Average Performance by ML Form', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(avg_metrics):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    fig.suptitle(f'ML Form Analysis - {checkpoint_type.upper()}',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_model_comparison(
    df: pd.DataFrame,
    checkpoint_type: str,
    output_path: Path
):
    """
    Plot comparison across different models (if multiple exist).
    
    Args:
        df: DataFrame with results
        checkpoint_type: 'best' or 'last'
        output_path: Path to save the plot
    """
    if df.empty:
        print("No data to plot")
        return
    
    model_types = df['model_type'].unique()
    
    if len(model_types) <= 1:
        print("Only one model type, skipping model comparison plot")
        return
    
    # Get common metrics across all ML forms
    common_metrics = ['acc', 'f1']
    available_metrics = [m for m in common_metrics if m in df.columns]
    
    if not available_metrics:
        print("No common metrics found for comparison")
        return
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        # Get data for this metric
        df_metric = df[df[metric].notna()]
        
        if df_metric.empty:
            continue
        
        # Group by model and calculate mean
        model_means = df_metric.groupby('model_type')[metric].mean().sort_values(ascending=False)
        
        ax.bar(range(len(model_means)), model_means.values, color='teal', edgecolor='black')
        ax.set_xticks(range(len(model_means)))
        ax.set_xticklabels(model_means.index, rotation=45, ha='right')
        ax.set_ylabel(f'Average {metric.upper()}', fontsize=12)
        ax.set_title(f'{metric.upper()} by Model', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(model_means.values):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    fig.suptitle(f'Model Comparison - {checkpoint_type.upper()}',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
