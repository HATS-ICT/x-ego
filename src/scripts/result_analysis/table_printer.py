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
        precision: int = 1,
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
        
        # Add columns: Task ID, Model, then for each metric: Baseline, Finetuned, Delta
        table.add_column("Task ID", style="cyan", no_wrap=True)
        table.add_column("Model", style="green")
        
        if is_aggregated:
            table.add_column("N", justify="center", style="dim")  # Number of repeats
        
        for metric in metric_cols:
            table.add_column(f"{metric.upper()} (B)", justify="right", style="dim")
            table.add_column(f"{metric.upper()} (F)", justify="right", style="white")
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
                        delta_str = f"[{delta_style}]{delta:+.1f}[/{delta_style}]"
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
            init_style = "green" if init_type == 'finetuned' else "dim"
            
            row_data = [
                row['task_id'],
                row['model_type'],
                row.get('ui_mask', 'unknown'),
                f"[{init_style}]{init_type}[/{init_style}]",
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
        # For multi_label_cls: acc_exact, hamming_acc (derived from hamming_dist), f1, auroc
        metric_map = {
            'binary_cls': ['acc', 'f1', 'auroc'],
            'multi_cls': ['acc', 'f1', 'acc_top3', 'acc_top5'],
            'multi_label_cls': ['acc_exact', 'hamming_acc', 'f1', 'auroc'],
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
                        metric_avgs.append(f"{val:.1f}")
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
        
        self.console.rule(f"[bold blue]Baseline vs Finetuned Comparison - {checkpoint_type.upper()}[/bold blue]")
        self.console.print()
        
        # For each ML form, show comparison
        for ml_form in df['ml_form'].unique():
            df_ml = df[df['ml_form'] == ml_form]
            metric_cols = self._get_metric_columns(ml_form, df_ml)
            
            if not metric_cols:
                continue
            
            table = Table(
                title=f"{ml_form.upper()} - Baseline vs Finetuned by Model & UI Mask",
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
                                metric_avgs.append(f"{val:.1f}")
                            else:
                                metric_avgs.append("N/A")
                        
                        init_style = "green" if init_type == 'finetuned' else "dim"
                        
                        table.add_row(
                            model,
                            ui_mask,
                            f"[{init_style}]{init_type}[/{init_style}]",
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
        self.console.rule(f"[bold blue]Task-Level Baseline vs Finetuned - {checkpoint_type.upper()}{title_suffix}[/bold blue]")
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
            table.add_column("Finetuned", justify="right")
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
                            delta_str = f"[{delta_style}]{delta:+.1f}[/{delta_style}]"
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
    
    if args.verbose:
        # Show detailed tables with all columns
        results_by_ml_form = {
            ml_form: df_filtered[df_filtered['ml_form'] == ml_form]
            for ml_form in df_filtered['ml_form'].unique()
        }
        printer.print_all_tables(results_by_ml_form, args.checkpoint)
        printer.print_summary_statistics(df_filtered)
    else:
        # Show baseline vs finetuned comparison tables
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
    n_finetuned = len(df_filtered[df_filtered['init_type'] == 'finetuned'])
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
    printer.console.print(f"  Finetuned results: {n_finetuned}")
    printer.console.print()


if __name__ == '__main__':
    main()
