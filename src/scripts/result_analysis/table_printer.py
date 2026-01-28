#!/usr/bin/env python3
"""
Table Printer

Print results tables using rich library, grouped by ML form.
Supports displaying aggregated results with mean ± std for repeated experiments.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

from results_collector import ResultsCollector


class TablePrinter:
    """Print formatted results tables using rich library."""
    
    # Metrics that should be displayed as percentages (multiply by 100)
    # Only accuracy-related metrics; F1, AUROC, R² are conventionally shown as decimals
    PERCENTAGE_METRICS = {
        'acc', 'acc_top3', 'acc_top5', 'acc_exact', 'hamming_acc'
    }
    
    def __init__(self):
        """Initialize table printer with console."""
        self.console = Console()
    
    def _is_aggregated_df(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has aggregated results (mean/std columns)."""
        return 'n_repeats' in df.columns
    
    def _is_percentage_metric(self, metric: str) -> bool:
        """Check if metric should be displayed as percentage."""
        return metric in self.PERCENTAGE_METRICS
    
    def _format_metric_value(
        self, 
        mean_val: Optional[float], 
        std_val: Optional[float] = None,
        precision: int = 3,
        show_std: bool = True,
        as_percentage: bool = False
    ) -> str:
        """
        Format a metric value with optional std.
        
        Args:
            mean_val: Mean value (or single value if not aggregated)
            std_val: Standard deviation (None if not aggregated or single repeat)
            precision: Number of decimal places
            show_std: Whether to show std (if available)
            as_percentage: If True, multiply by 100 (e.g., 0.442 -> 44.2)
        
        Returns:
            Formatted string like "44.2" or "44.2±1.2"
        """
        if mean_val is None or (isinstance(mean_val, float) and np.isnan(mean_val)):
            return "N/A"
        
        if as_percentage:
            mean_val = mean_val * 100
            if std_val is not None:
                std_val = std_val * 100
        
        if show_std and std_val is not None and not np.isnan(std_val) and std_val > 0:
            return f"{mean_val:.{precision}f}±{std_val:.{precision}f}"
        else:
            return f"{mean_val:.{precision}f}"
    
    def _get_metric_value(
        self, 
        row: pd.Series, 
        metric: str, 
        is_aggregated: bool
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Get metric value (mean and std) from a row.
        
        Args:
            row: DataFrame row
            metric: Metric name (base name without _mean/_std suffix)
            is_aggregated: Whether the DataFrame is aggregated
        
        Returns:
            Tuple of (mean_value, std_value)
        """
        if is_aggregated:
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            mean_val = row.get(mean_col)
            std_val = row.get(std_col)
        else:
            mean_val = row.get(metric)
            std_val = None
        
        # Handle NaN
        if pd.isna(mean_val):
            mean_val = None
        if std_val is not None and pd.isna(std_val):
            std_val = None
            
        return mean_val, std_val
    
    def print_baseline_finetuned_table(
        self, 
        df: pd.DataFrame, 
        ml_form: str,
        ui_mask: str,
        checkpoint_type: str = 'best'
    ):
        """
        Print a table comparing baseline vs finetuned for each task.
        
        Args:
            df: DataFrame with results for this ML form (can be aggregated or not)
            ml_form: ML form name (binary_cls, multi_cls, etc.)
            ui_mask: UI mask setting to filter
            checkpoint_type: 'best' or 'last'
        """
        if df.empty:
            self.console.print(f"[yellow]No results for {ml_form}[/yellow]")
            return
        
        is_aggregated = self._is_aggregated_df(df)
        
        # Determine metric columns based on ml_form
        metric_cols = self._get_metric_columns(ml_form, df)
        
        if not metric_cols:
            self.console.print(f"[yellow]No metrics found for {ml_form}[/yellow]")
            return
        
        # Create rich table
        title = f"{ml_form.upper()} | UI Mask: {ui_mask} | {checkpoint_type.upper()} Checkpoint"
        if is_aggregated:
            title += " (mean±std)"
        
        table = Table(
            title=title,
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        # Add columns: Task ID, Model, then for each metric: Baseline, CECL, Delta
        table.add_column("Task ID", style="cyan", no_wrap=True)
        table.add_column("Model", style="green")
        
        if is_aggregated:
            table.add_column("N", justify="center", style="dim")  # Number of repeats
        
        for metric in metric_cols:
            table.add_column(f"{metric.upper()} (B)", justify="right", style="dim")
            table.add_column(f"{metric.upper()} (C)", justify="right", style="white")  # C for CECL
            table.add_column("Δ", justify="right")
        
        # Group by task and model
        for task_id in sorted(df['task_id'].unique()):
            df_task = df[df['task_id'] == task_id]
            
            for model in sorted(df_task['model_type'].unique()):
                df_model = df_task[df_task['model_type'] == model]
                
                baseline_rows = df_model[df_model['init_type'] == 'baseline']
                finetuned_rows = df_model[df_model['init_type'] == 'finetuned']
                
                row_values = [task_id, model]
                
                # Add repeat count if aggregated
                if is_aggregated:
                    b_n = baseline_rows['n_repeats'].values[0] if len(baseline_rows) > 0 else 0
                    f_n = finetuned_rows['n_repeats'].values[0] if len(finetuned_rows) > 0 else 0
                    row_values.append(f"{b_n}/{f_n}")
                
                for metric in metric_cols:
                    # Get baseline values
                    if len(baseline_rows) > 0:
                        b_mean, b_std = self._get_metric_value(
                            baseline_rows.iloc[0], metric, is_aggregated
                        )
                    else:
                        b_mean, b_std = None, None
                    
                    # Get finetuned values
                    if len(finetuned_rows) > 0:
                        f_mean, f_std = self._get_metric_value(
                            finetuned_rows.iloc[0], metric, is_aggregated
                        )
                    else:
                        f_mean, f_std = None, None
                    
                    as_pct = self._is_percentage_metric(metric)
                    baseline_str = self._format_metric_value(
                        b_mean, b_std, show_std=is_aggregated, as_percentage=as_pct
                    )
                    finetuned_str = self._format_metric_value(
                        f_mean, f_std, show_std=is_aggregated, as_percentage=as_pct
                    )
                    
                    # Compute delta (based on means)
                    if b_mean is not None and f_mean is not None:
                        delta = f_mean - b_mean
                        if as_pct:
                            delta = delta * 100
                        delta_style = "green" if delta > 0 else ("red" if delta < 0 else "white")
                        delta_str = f"[{delta_style}]{delta:+.3f}[/{delta_style}]"
                    else:
                        delta_str = "N/A"
                    
                    row_values.extend([baseline_str, finetuned_str, delta_str])
                
                table.add_row(*row_values)
        
        self.console.print(table)
        self.console.print()
    
    def print_ml_form_table(
        self, 
        df: pd.DataFrame, 
        ml_form: str,
        checkpoint_type: str = 'best'
    ):
        """
        Print a table for a specific ML form.
        
        Args:
            df: DataFrame with results for this ML form (can be aggregated or not)
            ml_form: ML form name (binary_cls, multi_cls, etc.)
            checkpoint_type: 'best' or 'last'
        """
        if df.empty:
            self.console.print(f"[yellow]No results for {ml_form}[/yellow]")
            return
        
        is_aggregated = self._is_aggregated_df(df)
        
        # Determine metric columns based on ml_form
        metric_cols = self._get_metric_columns(ml_form, df)
        
        # Create rich table
        title = f"{ml_form.upper()} - {checkpoint_type.upper()} Checkpoint"
        if is_aggregated:
            title += " (mean±std)"
        
        table = Table(
            title=title,
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        # Add columns
        table.add_column("Task ID", style="cyan", no_wrap=True)
        table.add_column("Model", style="green")
        table.add_column("UI Mask", style="blue")
        table.add_column("Init", style="yellow")
        if is_aggregated:
            table.add_column("N", justify="center", style="dim")
        table.add_column("Category", style="dim")
        
        # Add metric columns
        for metric in metric_cols:
            table.add_column(metric.upper(), justify="right", style="white")
        
        # Add rows
        for _, row in df.iterrows():
            metric_values = []
            for metric in metric_cols:
                mean_val, std_val = self._get_metric_value(row, metric, is_aggregated)
                as_pct = self._is_percentage_metric(metric)
                metric_values.append(
                    self._format_metric_value(
                        mean_val, std_val, show_std=is_aggregated, as_percentage=as_pct
                    )
                )
            
            init_type = row.get('init_type', 'unknown')
            init_display = 'CECL' if init_type == 'finetuned' else init_type.capitalize()
            init_style = "green" if init_type == 'finetuned' else "dim"
            
            row_data = [
                row['task_id'],
                row['model_type'],
                row.get('ui_mask', 'unknown'),
                f"[{init_style}]{init_display}[/{init_style}]",
            ]
            
            if is_aggregated:
                row_data.append(str(row.get('n_repeats', 1)))
            
            row_data.append(row['category'])
            row_data.extend(metric_values)
            
            table.add_row(*row_data)
        
        self.console.print(table)
        self.console.print()
    
    def _get_metric_columns(self, ml_form: str, df: pd.DataFrame) -> List[str]:
        """
        Get metric column names for a specific ML form.
        
        Args:
            ml_form: ML form name
            df: DataFrame to check for available columns
        
        Returns:
            List of metric column names (base names without _mean/_std suffix)
        """
        # Define expected metrics for each ML form
        # multi_cls: Top-1, Top-3, F1 (removed acc_top5)
        # multi_label_cls: Exact Match, Hamming Acc (1-hamming_dist), F1 (removed auroc)
        metric_map = {
            'binary_cls': ['acc', 'f1', 'auroc'],
            'multi_cls': ['acc', 'acc_top3', 'f1'],
            'multi_label_cls': ['acc_exact', 'hamming_acc', 'f1'],
            'regression': ['mse', 'mae', 'r2']
        }
        
        expected_metrics = metric_map.get(ml_form, [])
        
        # Check if this is an aggregated DataFrame (has _mean suffix columns)
        is_aggregated = self._is_aggregated_df(df)
        
        # Only return metrics that exist in the dataframe
        available_metrics = []
        for m in expected_metrics:
            if is_aggregated:
                # Check for _mean column
                if f'{m}_mean' in df.columns:
                    available_metrics.append(m)
            else:
                if m in df.columns:
                    available_metrics.append(m)
        
        return available_metrics
    
    def print_all_tables(
        self, 
        results_by_ml_form: Dict[str, pd.DataFrame],
        checkpoint_type: str = 'best'
    ):
        """
        Print tables for all ML forms.
        
        Args:
            results_by_ml_form: Dict mapping ml_form to DataFrame
            checkpoint_type: 'best' or 'last'
        """
        self.console.rule(f"[bold blue]Results Summary - {checkpoint_type.upper()} Checkpoint[/bold blue]")
        self.console.print()
        
        # Print in a consistent order
        ml_form_order = ['binary_cls', 'multi_cls', 'multi_label_cls', 'regression']
        
        for ml_form in ml_form_order:
            if ml_form in results_by_ml_form:
                self.print_ml_form_table(
                    results_by_ml_form[ml_form], 
                    ml_form, 
                    checkpoint_type
                )
    
    def print_summary_statistics(self, df: pd.DataFrame):
        """
        Print summary statistics across all tasks.
        
        Args:
            df: DataFrame with all results
        """
        if df.empty:
            return
        
        table = Table(
            title="Summary Statistics",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right")
        
        # Count by ML form
        ml_form_counts = df['ml_form'].value_counts()
        
        table.add_row("Total Tasks", str(len(df)))
        table.add_row("", "")  # Empty row
        
        for ml_form, count in ml_form_counts.items():
            table.add_row(f"  {ml_form}", str(count))
        
        self.console.print(table)
        self.console.print()
    
    def print_model_comparison(self, df: pd.DataFrame, checkpoint_type: str = 'best'):
        """
        Print comparison across different models (if multiple exist).
        
        Args:
            df: DataFrame with all results (can be aggregated or not)
            checkpoint_type: 'best' or 'last'
        """
        if df.empty:
            return
        
        model_types = df['model_type'].unique()
        
        if len(model_types) <= 1:
            self.console.print("[yellow]Only one model type found, skipping comparison[/yellow]")
            return
        
        is_aggregated = self._is_aggregated_df(df)
        
        self.console.rule(f"[bold blue]Model Comparison - {checkpoint_type.upper()}[/bold blue]")
        self.console.print()
        
        # For each ML form, show average metrics per model
        for ml_form in df['ml_form'].unique():
            df_ml = df[df['ml_form'] == ml_form]
            metric_cols = self._get_metric_columns(ml_form, df_ml)
            
            if not metric_cols:
                continue
            
            table = Table(
                title=f"{ml_form.upper()} - Average Metrics by Model",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold magenta"
            )
            
            table.add_column("Model", style="green")
            table.add_column("# Tasks", justify="right")
            
            for metric in metric_cols:
                table.add_column(f"Avg {metric.upper()}", justify="right")
            
            for model in sorted(df_ml['model_type'].unique()):
                df_model = df_ml[df_ml['model_type'] == model]
                
                metric_avgs = []
                for metric in metric_cols:
                    if is_aggregated:
                        col = f'{metric}_mean'
                    else:
                        col = metric
                    if col in df_model.columns:
                        val = df_model[col].mean()
                        if self._is_percentage_metric(metric):
                            val = val * 100
                        metric_avgs.append(f"{val:.3f}")
                    else:
                        metric_avgs.append("N/A")
                
                table.add_row(
                    model,
                    str(len(df_model)),
                    *metric_avgs
                )
            
            self.console.print(table)
            self.console.print()
    
    def print_baseline_vs_finetuned_comparison(self, df: pd.DataFrame, checkpoint_type: str = 'best'):
        """
        Print comparison between baseline and finetuned models.
        
        Args:
            df: DataFrame with all results (can be aggregated or not)
            checkpoint_type: 'best' or 'last'
        """
        if df.empty:
            return
        
        if 'init_type' not in df.columns:
            self.console.print("[yellow]No init_type column found, skipping baseline vs finetuned comparison[/yellow]")
            return
        
        init_types = df['init_type'].unique()
        if len(init_types) <= 1:
            self.console.print(f"[yellow]Only {init_types[0]} results found, skipping comparison[/yellow]")
            return
        
        is_aggregated = self._is_aggregated_df(df)
        
        self.console.rule(f"[bold blue]Baseline vs CECL Comparison - {checkpoint_type.upper()}[/bold blue]")
        self.console.print()
        
        # For each ML form, show comparison
        for ml_form in df['ml_form'].unique():
            df_ml = df[df['ml_form'] == ml_form]
            metric_cols = self._get_metric_columns(ml_form, df_ml)
            
            if not metric_cols:
                continue
            
            table = Table(
                title=f"{ml_form.upper()} - Baseline vs CECL by Model & UI Mask",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold magenta"
            )
            
            table.add_column("Model", style="green")
            table.add_column("UI Mask", style="blue")
            table.add_column("Init Type", style="yellow")
            table.add_column("# Tasks", justify="right")
            
            for metric in metric_cols:
                table.add_column(f"Avg {metric.upper()}", justify="right")
            
            # Group by model, ui_mask, and init_type
            for model in sorted(df_ml['model_type'].unique()):
                df_model = df_ml[df_ml['model_type'] == model]
                
                for ui_mask in sorted(df_model['ui_mask'].unique()):
                    df_ui = df_model[df_model['ui_mask'] == ui_mask]
                    
                    for init_type in ['baseline', 'finetuned']:
                        df_init = df_ui[df_ui['init_type'] == init_type]
                        
                        if df_init.empty:
                            continue
                        
                        metric_avgs = []
                        for metric in metric_cols:
                            if is_aggregated:
                                col = f'{metric}_mean'
                            else:
                                col = metric
                            if col in df_init.columns:
                                val = df_init[col].mean()
                                if self._is_percentage_metric(metric):
                                    val = val * 100
                                metric_avgs.append(f"{val:.3f}")
                            else:
                                metric_avgs.append("N/A")
                        
                        init_display = 'CECL' if init_type == 'finetuned' else init_type.capitalize()
                        init_style = "green" if init_type == 'finetuned' else "dim"
                        
                        table.add_row(
                            model,
                            ui_mask,
                            f"[{init_style}]{init_display}[/{init_style}]",
                            str(len(df_init)),
                            *metric_avgs
                        )
            
            self.console.print(table)
            self.console.print()
    
    def print_detailed_task_comparison(self, df: pd.DataFrame, checkpoint_type: str = 'best'):
        """
        Print detailed task-level comparison between baseline and finetuned.
        
        Args:
            df: DataFrame with all results (can be aggregated or not)
            checkpoint_type: 'best' or 'last'
        """
        if df.empty:
            return
        
        if 'init_type' not in df.columns:
            return
        
        is_aggregated = self._is_aggregated_df(df)
        
        title_suffix = " (mean±std)" if is_aggregated else ""
        self.console.rule(f"[bold blue]Task-Level Baseline vs CECL - {checkpoint_type.upper()}{title_suffix}[/bold blue]")
        self.console.print()
        
        # For each ML form
        for ml_form in df['ml_form'].unique():
            df_ml = df[df['ml_form'] == ml_form]
            metric_cols = self._get_metric_columns(ml_form, df_ml)
            
            if not metric_cols:
                continue
            
            # Use first metric for comparison
            primary_metric = metric_cols[0]
            
            table = Table(
                title=f"{ml_form.upper()} - Task Comparison ({primary_metric.upper()})",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold magenta"
            )
            
            table.add_column("Task ID", style="cyan", no_wrap=True)
            table.add_column("Model", style="green")
            table.add_column("UI Mask", style="blue")
            if is_aggregated:
                table.add_column("N", justify="center", style="dim")
            table.add_column("Baseline", justify="right")
            table.add_column("CECL", justify="right")
            table.add_column("Delta", justify="right")
            
            # Group by task, model, ui_mask
            for task_id in sorted(df_ml['task_id'].unique()):
                df_task = df_ml[df_ml['task_id'] == task_id]
                
                for model in sorted(df_task['model_type'].unique()):
                    df_model = df_task[df_task['model_type'] == model]
                    
                    for ui_mask in sorted(df_model['ui_mask'].unique()):
                        df_ui = df_model[df_model['ui_mask'] == ui_mask]
                        
                        baseline_rows = df_ui[df_ui['init_type'] == 'baseline']
                        finetuned_rows = df_ui[df_ui['init_type'] == 'finetuned']
                        
                        # Get values
                        if len(baseline_rows) > 0:
                            b_mean, b_std = self._get_metric_value(
                                baseline_rows.iloc[0], primary_metric, is_aggregated
                            )
                        else:
                            b_mean, b_std = None, None
                        
                        if len(finetuned_rows) > 0:
                            f_mean, f_std = self._get_metric_value(
                                finetuned_rows.iloc[0], primary_metric, is_aggregated
                            )
                        else:
                            f_mean, f_std = None, None
                        
                        as_pct = self._is_percentage_metric(primary_metric)
                        baseline_str = self._format_metric_value(
                            b_mean, b_std, show_std=is_aggregated, as_percentage=as_pct
                        )
                        finetuned_str = self._format_metric_value(
                            f_mean, f_std, show_std=is_aggregated, as_percentage=as_pct
                        )
                        
                        if b_mean is not None and f_mean is not None:
                            delta = f_mean - b_mean
                            if as_pct:
                                delta = delta * 100
                            delta_style = "green" if delta > 0 else ("red" if delta < 0 else "white")
                            delta_str = f"[{delta_style}]{delta:+.3f}[/{delta_style}]"
                        else:
                            delta_str = "N/A"
                        
                        row_data = [task_id, model, ui_mask]
                        if is_aggregated:
                            b_n = baseline_rows['n_repeats'].values[0] if len(baseline_rows) > 0 else 0
                            f_n = finetuned_rows['n_repeats'].values[0] if len(finetuned_rows) > 0 else 0
                            row_data.append(f"{b_n}/{f_n}")
                        row_data.extend([baseline_str, finetuned_str, delta_str])
                        
                        table.add_row(*row_data)
            
            self.console.print(table)
            self.console.print()
    
    def _get_task_perspective(self, task_id: str) -> Optional[str]:
        """Extract perspective (self, teammate, enemy, global) from task_id."""
        PERSPECTIVES = ['self', 'teammate', 'enemy', 'global']
        for perspective in PERSPECTIVES:
            if task_id.startswith(perspective):
                return perspective
        return None
    
    def _compute_relative_improvement(
        self, 
        baseline_val: float, 
        finetuned_val: float, 
        metric: str
    ) -> float:
        """
        Compute relative percentage improvement.
        
        For metrics where lower is better (MSE, MAE), invert the sign so
        positive always means "better".
        
        Formula: ((Finetuned - Baseline) / |Baseline|) * 100
        """
        LOWER_IS_BETTER = {'mse', 'mae', 'hamming_dist'}
        
        if np.isnan(baseline_val) or np.isnan(finetuned_val):
            return np.nan
        
        if abs(baseline_val) < 1e-10:
            return np.nan
        
        if metric in LOWER_IS_BETTER:
            # For lower-is-better: reduction is improvement
            improvement = ((baseline_val - finetuned_val) / abs(baseline_val)) * 100
        else:
            # For higher-is-better: increase is improvement
            improvement = ((finetuned_val - baseline_val) / abs(baseline_val)) * 100
        
        return improvement
    
    def _get_primary_metric(self, ml_form: str, metric_type: str = 'acc') -> str:
        """
        Get the primary metric for a given ML form.
        
        Args:
            ml_form: ML form type
            metric_type: 'acc' for accuracy-based or 'f1' for F1-based
        
        Returns:
            Metric name to use
        """
        if metric_type == 'f1':
            # F1-based metrics
            PRIMARY_METRIC_MAP = {
                'binary_cls': 'f1',
                'multi_cls': 'f1',
                'multi_label_cls': 'f1',
                'regression': 'mae'  # regression doesn't have F1
            }
        else:
            # Accuracy-based metrics (default)
            PRIMARY_METRIC_MAP = {
                'binary_cls': 'acc',
                'multi_cls': 'acc',
                'multi_label_cls': 'hamming_acc',
                'regression': 'mae'
            }
        return PRIMARY_METRIC_MAP.get(ml_form, 'acc' if metric_type == 'acc' else 'f1')
    
    def print_categorical_summary_table(
        self,
        df: pd.DataFrame,
        model_type: Optional[str] = None,
        checkpoint_type: str = 'best',
        metric_type: str = 'acc'
    ):
        """
        Print a summary table grouped by perspective (Global, Self, Teammate, Enemy).
        
        Shows:
        - Number of tasks per perspective
        - Average relative improvement (%)
        - Win rate vs baseline
        - Top task name and its delta
        
        This table supports the "Global vs Self trade-off" narrative.
        
        Args:
            df: DataFrame with results (should be aggregated if from repeated experiments)
            model_type: Filter to specific model, or None for all models
            checkpoint_type: 'best' or 'last'
            metric_type: 'acc' for accuracy-based or 'f1' for F1-based metrics
        """
        if df.empty:
            self.console.print("[yellow]No results to summarize[/yellow]")
            return
        
        is_aggregated = self._is_aggregated_df(df)
        
        # Filter by model if specified
        if model_type:
            df = df[df['model_type'] == model_type]
        
        if df.empty:
            self.console.print(f"[yellow]No results for model: {model_type}[/yellow]")
            return
        
        # Add perspective column
        df = df.copy()
        df['perspective'] = df['task_id'].apply(self._get_task_perspective)
        df = df[df['perspective'].notna()]
        
        if df.empty:
            self.console.print("[yellow]No tasks with recognized perspectives[/yellow]")
            return
        
        # Build title
        model_str = model_type.upper() if model_type else "All Models"
        metric_str = "F1" if metric_type == 'f1' else "ACC"
        title = f"Categorical Summary ({metric_str}) | {model_str} | {checkpoint_type.upper()}"
        
        table = Table(
            title=title,
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        table.add_column("Perspective", style="cyan", no_wrap=True)
        table.add_column("# Tasks", justify="center")
        table.add_column("Avg Improv. (%)", justify="right")
        table.add_column("Win Rate", justify="center")
        table.add_column("Best Task", style="dim")
        table.add_column("Best Δ (%)", justify="right")
        table.add_column("Worst Task", style="dim")
        table.add_column("Worst Δ (%)", justify="right")
        
        # Process each perspective
        PERSPECTIVE_ORDER = ['global', 'enemy', 'teammate', 'self']
        PERSPECTIVE_DISPLAY = {
            'global': 'Global',
            'enemy': 'Enemy',
            'teammate': 'Teammate',
            'self': 'Self'
        }
        
        for perspective in PERSPECTIVE_ORDER:
            df_persp = df[df['perspective'] == perspective]
            
            if df_persp.empty:
                continue
            
            # Get unique tasks in this perspective
            task_ids = df_persp['task_id'].unique()
            n_tasks = len(task_ids)
            
            task_improvements = []  # List of (task_id, improvement)
            wins = 0
            
            for task_id in task_ids:
                df_task = df_persp[df_persp['task_id'] == task_id]
                
                if df_task.empty:
                    continue
                
                ml_form = df_task['ml_form'].iloc[0]
                primary_metric = self._get_primary_metric(ml_form, metric_type)
                
                # Get baseline and finetuned values
                baseline_rows = df_task[df_task['init_type'] == 'baseline']
                finetuned_rows = df_task[df_task['init_type'] == 'finetuned']
                
                if baseline_rows.empty or finetuned_rows.empty:
                    continue
                
                # Check if metric exists in data
                if is_aggregated:
                    metric_col = f'{primary_metric}_mean'
                else:
                    metric_col = primary_metric
                
                if metric_col not in baseline_rows.columns or metric_col not in finetuned_rows.columns:
                    continue
                
                if is_aggregated:
                    b_val = baseline_rows[f'{primary_metric}_mean'].values[0]
                    f_val = finetuned_rows[f'{primary_metric}_mean'].values[0]
                else:
                    b_val = baseline_rows[primary_metric].values[0]
                    f_val = finetuned_rows[primary_metric].values[0]
                
                # Compute relative improvement
                improvement = self._compute_relative_improvement(b_val, f_val, primary_metric)
                
                if not np.isnan(improvement):
                    task_improvements.append((task_id, improvement))
                    if improvement > 0:
                        wins += 1
            
            if not task_improvements:
                continue
            
            # Sort by improvement to find best and worst
            task_improvements.sort(key=lambda x: x[1], reverse=True)
            best_task, best_delta = task_improvements[0]
            worst_task, worst_delta = task_improvements[-1]
            
            improvements = [imp for _, imp in task_improvements]
            avg_improvement = np.mean(improvements)
            win_rate = f"{wins}/{len(improvements)}"
            
            # Format improvement with color
            if avg_improvement > 0:
                improv_style = "green"
            elif avg_improvement < 0:
                improv_style = "red"
            else:
                improv_style = "white"
            
            improv_str = f"[{improv_style}]{avg_improvement:+.3f}[/{improv_style}]"
            
            best_delta_str = f"[green]{best_delta:+.3f}[/green]" if best_delta > 0 else f"[red]{best_delta:+.3f}[/red]"
            worst_delta_str = f"[green]{worst_delta:+.3f}[/green]" if worst_delta > 0 else f"[red]{worst_delta:+.3f}[/red]"
            
            table.add_row(
                PERSPECTIVE_DISPLAY.get(perspective, perspective),
                str(n_tasks),
                improv_str,
                win_rate,
                best_task,
                best_delta_str,
                worst_task,
                worst_delta_str
            )
        
        self.console.print(table)
        self.console.print()
    
    def print_categorical_summary_table_by_model(
        self,
        df: pd.DataFrame,
        model_type: Optional[str] = None,
        checkpoint_type: str = 'best',
        metric_type: str = 'acc'
    ):
        """
        Print a summary table grouped by perspective, with models separated for win rate.
        
        Similar to print_categorical_summary_table, but treats each (task, model) pair
        as a separate entry for win rate calculation. This gives a more granular view
        when multiple models are tested on the same tasks.
        
        Shows:
        - Number of (task, model) pairs per perspective
        - Average relative improvement (%)
        - Win rate vs baseline (per task-model pair)
        - Top task-model and its delta
        
        Args:
            df: DataFrame with results (should be aggregated if from repeated experiments)
            model_type: Filter to specific model, or None for all models
            checkpoint_type: 'best' or 'last'
            metric_type: 'acc' for accuracy-based or 'f1' for F1-based metrics
        """
        if df.empty:
            self.console.print("[yellow]No results to summarize[/yellow]")
            return
        
        is_aggregated = self._is_aggregated_df(df)
        
        # Filter by model if specified
        if model_type:
            df = df[df['model_type'] == model_type]
        
        if df.empty:
            self.console.print(f"[yellow]No results for model: {model_type}[/yellow]")
            return
        
        # Add perspective column
        df = df.copy()
        df['perspective'] = df['task_id'].apply(self._get_task_perspective)
        df = df[df['perspective'].notna()]
        
        if df.empty:
            self.console.print("[yellow]No tasks with recognized perspectives[/yellow]")
            return
        
        # Build title
        model_str = model_type.upper() if model_type else "All Models"
        metric_str = "F1" if metric_type == 'f1' else "ACC"
        title = f"Categorical Summary ({metric_str}) | {model_str} | {checkpoint_type.upper()} | Per-Model Win Rate"
        
        table = Table(
            title=title,
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        table.add_column("Perspective", style="cyan", no_wrap=True)
        table.add_column("# Pairs", justify="center")  # (task, model) pairs
        table.add_column("Avg Improv. (%)", justify="right")
        table.add_column("Win Rate", justify="center")
        table.add_column("Best Task-Model", style="dim")
        table.add_column("Best Δ (%)", justify="right")
        table.add_column("Worst Task-Model", style="dim")
        table.add_column("Worst Δ (%)", justify="right")
        
        # Process each perspective
        PERSPECTIVE_ORDER = ['global', 'enemy', 'teammate', 'self']
        PERSPECTIVE_DISPLAY = {
            'global': 'Global',
            'enemy': 'Enemy',
            'teammate': 'Teammate',
            'self': 'Self'
        }
        
        for perspective in PERSPECTIVE_ORDER:
            df_persp = df[df['perspective'] == perspective]
            
            if df_persp.empty:
                continue
            
            # Get unique (task_id, model_type) pairs in this perspective
            task_model_pairs = df_persp[['task_id', 'model_type']].drop_duplicates()
            n_pairs = len(task_model_pairs)
            
            pair_improvements = []  # List of (task_id, model_type, improvement)
            wins = 0
            
            for _, row in task_model_pairs.iterrows():
                task_id = row['task_id']
                model = row['model_type']
                df_pair = df_persp[(df_persp['task_id'] == task_id) & (df_persp['model_type'] == model)]
                
                if df_pair.empty:
                    continue
                
                ml_form = df_pair['ml_form'].iloc[0]
                primary_metric = self._get_primary_metric(ml_form, metric_type)
                
                # Get baseline and finetuned values
                baseline_rows = df_pair[df_pair['init_type'] == 'baseline']
                finetuned_rows = df_pair[df_pair['init_type'] == 'finetuned']
                
                if baseline_rows.empty or finetuned_rows.empty:
                    continue
                
                # Check if metric exists in data
                if is_aggregated:
                    metric_col = f'{primary_metric}_mean'
                else:
                    metric_col = primary_metric
                
                if metric_col not in baseline_rows.columns or metric_col not in finetuned_rows.columns:
                    continue
                
                if is_aggregated:
                    b_val = baseline_rows[f'{primary_metric}_mean'].values[0]
                    f_val = finetuned_rows[f'{primary_metric}_mean'].values[0]
                else:
                    b_val = baseline_rows[primary_metric].values[0]
                    f_val = finetuned_rows[primary_metric].values[0]
                
                # Compute relative improvement
                improvement = self._compute_relative_improvement(b_val, f_val, primary_metric)
                
                if not np.isnan(improvement):
                    pair_improvements.append((task_id, model, improvement))
                    if improvement > 0:
                        wins += 1
            
            if not pair_improvements:
                continue
            
            # Sort by improvement to find best and worst
            pair_improvements.sort(key=lambda x: x[2], reverse=True)
            best_task, best_model, best_delta = pair_improvements[0]
            worst_task, worst_model, worst_delta = pair_improvements[-1]
            
            improvements = [imp for _, _, imp in pair_improvements]
            avg_improvement = np.mean(improvements)
            win_rate = f"{wins}/{len(improvements)}"
            
            # Format improvement with color
            if avg_improvement > 0:
                improv_style = "green"
            elif avg_improvement < 0:
                improv_style = "red"
            else:
                improv_style = "white"
            
            improv_str = f"[{improv_style}]{avg_improvement:+.3f}[/{improv_style}]"
            
            best_delta_str = f"[green]{best_delta:+.3f}[/green]" if best_delta > 0 else f"[red]{best_delta:+.3f}[/red]"
            worst_delta_str = f"[green]{worst_delta:+.3f}[/green]" if worst_delta > 0 else f"[red]{worst_delta:+.3f}[/red]"
            
            # Format task-model display (truncate long task names)
            best_display = f"{best_task[:20]}|{best_model}" if len(best_task) > 20 else f"{best_task}|{best_model}"
            worst_display = f"{worst_task[:20]}|{worst_model}" if len(worst_task) > 20 else f"{worst_task}|{worst_model}"
            
            table.add_row(
                PERSPECTIVE_DISPLAY.get(perspective, perspective),
                str(n_pairs),
                improv_str,
                win_rate,
                best_display,
                best_delta_str,
                worst_display,
                worst_delta_str
            )
        
        self.console.print(table)
        self.console.print()
    
    def print_task_ranking_table(
        self,
        df: pd.DataFrame,
        model_type: Optional[str] = None,
        checkpoint_type: str = 'best',
        metric_type: str = 'acc'
    ):
        """
        Print a full ranking table of all tasks sorted by relative improvement.
        
        Shows all tasks ranked from most improved to most degraded,
        with baseline, finetuned, and relative improvement (%).
        
        Args:
            df: DataFrame with results (should be aggregated if from repeated experiments)
            model_type: Filter to specific model, or None for all models
            checkpoint_type: 'best' or 'last'
            metric_type: 'acc' for accuracy-based or 'f1' for F1-based metrics
        """
        if df.empty:
            self.console.print("[yellow]No results to display[/yellow]")
            return
        
        is_aggregated = self._is_aggregated_df(df)
        
        # Filter by model if specified
        if model_type:
            df = df[df['model_type'] == model_type]
        
        if df.empty:
            self.console.print(f"[yellow]No results for model: {model_type}[/yellow]")
            return
        
        # Collect all task improvements
        task_data = []  # List of dicts with task info
        
        for task_id in df['task_id'].unique():
            df_task = df[df['task_id'] == task_id]
            
            if df_task.empty:
                continue
            
            ml_form = df_task['ml_form'].iloc[0]
            primary_metric = self._get_primary_metric(ml_form, metric_type)
            perspective = self._get_task_perspective(task_id)
            
            # Get baseline and finetuned values
            baseline_rows = df_task[df_task['init_type'] == 'baseline']
            finetuned_rows = df_task[df_task['init_type'] == 'finetuned']
            
            if baseline_rows.empty or finetuned_rows.empty:
                continue
            
            # Check if metric exists in data
            if is_aggregated:
                metric_col = f'{primary_metric}_mean'
            else:
                metric_col = primary_metric
            
            if metric_col not in baseline_rows.columns or metric_col not in finetuned_rows.columns:
                continue
            
            # Get values
            if is_aggregated:
                b_mean, b_std = self._get_metric_value(
                    baseline_rows.iloc[0], primary_metric, is_aggregated
                )
                f_mean, f_std = self._get_metric_value(
                    finetuned_rows.iloc[0], primary_metric, is_aggregated
                )
            else:
                b_mean = baseline_rows[primary_metric].values[0]
                b_std = None
                f_mean = finetuned_rows[primary_metric].values[0]
                f_std = None
            
            if b_mean is None or f_mean is None:
                continue
            
            # Compute relative improvement
            improvement = self._compute_relative_improvement(b_mean, f_mean, primary_metric)
            
            if np.isnan(improvement):
                continue
            
            task_data.append({
                'task_id': task_id,
                'perspective': perspective,
                'metric': primary_metric,
                'baseline': b_mean,
                'baseline_std': b_std,
                'finetuned': f_mean,
                'finetuned_std': f_std,
                'improvement': improvement,
                'is_aggregated': is_aggregated
            })
        
        if not task_data:
            self.console.print("[yellow]No tasks with valid improvements[/yellow]")
            return
        
        # Sort by improvement (most improved first)
        task_data.sort(key=lambda x: x['improvement'], reverse=True)
        
        # Build title
        model_str = model_type.upper() if model_type else "All Models"
        metric_str = "F1" if metric_type == 'f1' else "ACC"
        title = f"Task Ranking by Improvement ({metric_str}) | {model_str} | {checkpoint_type.upper()}"
        
        table = Table(
            title=title,
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        table.add_column("Rank", justify="center", style="dim")
        table.add_column("Task", style="cyan", no_wrap=True)
        table.add_column("Perspective", justify="center")
        table.add_column("Metric", justify="center", style="dim")
        table.add_column("Baseline", justify="right")
        table.add_column("CECL", justify="right")
        table.add_column("Δ Rel. (%)", justify="right")
        
        PERSPECTIVE_COLORS = {
            'global': 'blue',
            'enemy': 'magenta',
            'teammate': 'yellow',
            'self': 'red'
        }
        
        for rank, data in enumerate(task_data, 1):
            as_pct = self._is_percentage_metric(data['metric'])
            
            baseline_str = self._format_metric_value(
                data['baseline'], data['baseline_std'], 
                show_std=data['is_aggregated'], as_percentage=as_pct
            )
            finetuned_str = self._format_metric_value(
                data['finetuned'], data['finetuned_std'],
                show_std=data['is_aggregated'], as_percentage=as_pct
            )
            
            # Format improvement with color
            improvement = data['improvement']
            if improvement > 0:
                improv_style = "green"
            elif improvement < 0:
                improv_style = "red"
            else:
                improv_style = "white"
            improv_str = f"[{improv_style}]{improvement:+.3f}[/{improv_style}]"
            
            # Format perspective with color
            persp = data['perspective'] or 'unknown'
            persp_color = PERSPECTIVE_COLORS.get(persp, 'white')
            persp_str = f"[{persp_color}]{persp.capitalize()}[/{persp_color}]"
            
            table.add_row(
                str(rank),
                data['task_id'],
                persp_str,
                data['metric'].upper(),
                baseline_str,
                finetuned_str,
                improv_str
            )
        
        self.console.print(table)
        self.console.print()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Print result tables comparing baseline vs finetuned models.'
    )
    parser.add_argument(
        '--ui-mask', '-u',
        type=str,
        default='all',
        choices=['all', 'minimap_only', 'none'],
        help='UI mask setting to filter results (default: all)'
    )
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default='best',
        choices=['best', 'last'],
        help='Checkpoint type to show (default: best)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        choices=['clip', 'dinov2', 'siglip2', 'vjepa2'],
        help='Filter by specific model (default: show all models)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed tables with all columns'
    )
    parser.add_argument(
        '--aggregate', '-a',
        action='store_true',
        default=True,
        help='Aggregate results from repeated experiments (show mean±std)'
    )
    parser.add_argument(
        '--narrative', '-n',
        action='store_true',
        default=True,
        help='Show narrative tables (categorical summary + flagship tasks) for paper (default)'
    )
    parser.add_argument(
        '--no-narrative',
        action='store_true',
        help='Disable narrative tables and show detailed per-ML-form tables'
    )
    return parser.parse_args()


def main():
    """Main function to print result tables."""
    args = parse_args()
    
    project_root = Path(__file__).parent.parent.parent.parent
    output_dir = project_root / 'output'
    task_defs = project_root / 'data' / 'labels' / 'task_definitions.csv'
    
    # Collect results
    collector = ResultsCollector(output_dir, task_defs)
    all_results = collector.collect_all_results()
    
    if not all_results:
        print("No results found!")
        return
    
    printer = TablePrinter()
    
    # Create dataframe and filter by ui_mask
    df = collector.create_results_dataframe(
        all_results, 
        args.checkpoint,
        aggregate_repeats=args.aggregate
    )
    
    if df.empty:
        print(f"No results for checkpoint type: {args.checkpoint}")
        return
    
    # Filter by ui_mask
    df_filtered = df[df['ui_mask'] == args.ui_mask]
    
    if df_filtered.empty:
        print(f"No results for ui_mask: {args.ui_mask}")
        print(f"Available ui_masks: {df['ui_mask'].unique().tolist()}")
        return
    
    # Filter by model if specified
    if args.model:
        df_filtered = df_filtered[df_filtered['model_type'] == args.model]
        if df_filtered.empty:
            print(f"No results for model: {args.model}")
            return
    
    agg_str = " (aggregated)" if args.aggregate else ""
    printer.console.print("\n")
    printer.console.rule(
        f"[bold blue]Results: UI Mask={args.ui_mask} | Checkpoint={args.checkpoint.upper()}{agg_str}[/bold blue]"
    )
    printer.console.print()
    
    # Always show narrative tables by default
    if args.narrative and not args.no_narrative:
        # Show narrative tables for paper (categorical summary + task ranking)
        # ACC-based tables
        printer.console.rule("[bold green]Table 1: Categorical Summary by Perspective (ACC)[/bold green]")
        printer.console.print()
        printer.print_categorical_summary_table(
            df_filtered,
            model_type=args.model,
            checkpoint_type=args.checkpoint,
            metric_type='acc'
        )
        
        printer.console.rule("[bold green]Table 2: Categorical Summary by Perspective (ACC) - Per-Model Win Rate[/bold green]")
        printer.console.print()
        printer.print_categorical_summary_table_by_model(
            df_filtered,
            model_type=args.model,
            checkpoint_type=args.checkpoint,
            metric_type='acc'
        )
        
        printer.console.rule("[bold green]Table 3: Full Task Ranking by Improvement (ACC)[/bold green]")
        printer.console.print()
        printer.print_task_ranking_table(
            df_filtered,
            model_type=args.model,
            checkpoint_type=args.checkpoint,
            metric_type='acc'
        )
        
        # F1-based tables
        printer.console.rule("[bold green]Table 4: Categorical Summary by Perspective (F1)[/bold green]")
        printer.console.print()
        printer.print_categorical_summary_table(
            df_filtered,
            model_type=args.model,
            checkpoint_type=args.checkpoint,
            metric_type='f1'
        )
        
        printer.console.rule("[bold green]Table 5: Categorical Summary by Perspective (F1) - Per-Model Win Rate[/bold green]")
        printer.console.print()
        printer.print_categorical_summary_table_by_model(
            df_filtered,
            model_type=args.model,
            checkpoint_type=args.checkpoint,
            metric_type='f1'
        )
        
        printer.console.rule("[bold green]Table 6: Full Task Ranking by Improvement (F1)[/bold green]")
        printer.console.print()
        printer.print_task_ranking_table(
            df_filtered,
            model_type=args.model,
            checkpoint_type=args.checkpoint,
            metric_type='f1'
        )
    
    # Show detailed per-ML-form tables
    printer.console.rule("[bold green]Detailed Tables by ML Form[/bold green]")
    printer.console.print()
    
    ml_form_order = ['binary_cls', 'multi_cls', 'multi_label_cls', 'regression']
    
    for ml_form in ml_form_order:
        df_ml = df_filtered[df_filtered['ml_form'] == ml_form]
        if not df_ml.empty:
            printer.print_baseline_finetuned_table(
                df_ml, 
                ml_form, 
                args.ui_mask,
                args.checkpoint
            )
    
    # Print summary
    printer.console.print()
    printer.console.rule("[bold blue]Summary[/bold blue]")
    printer.console.print()
    
    # Count results
    n_baseline = len(df_filtered[df_filtered['init_type'] == 'baseline'])
    n_cecl = len(df_filtered[df_filtered['init_type'] == 'finetuned'])
    n_tasks = df_filtered['task_id'].nunique()
    n_models = df_filtered['model_type'].nunique()
    
    if args.aggregate:
        # For aggregated results, show unique experiment settings
        printer.console.print(f"  Unique experiment settings: {len(df_filtered)}")
        total_repeats = df_filtered['n_repeats'].sum() if 'n_repeats' in df_filtered.columns else len(df_filtered)
        printer.console.print(f"  Total experiment runs: {int(total_repeats)}")
    else:
        printer.console.print(f"  Total experiments: {len(df_filtered)}")
    printer.console.print(f"  Unique tasks: {n_tasks}")
    printer.console.print(f"  Models: {df_filtered['model_type'].unique().tolist()}")
    printer.console.print(f"  Baseline results: {n_baseline}")
    printer.console.print(f"  CECL results: {n_cecl}")
    printer.console.print()


if __name__ == '__main__':
    main()
