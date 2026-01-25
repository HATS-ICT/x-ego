#!/usr/bin/env python3
"""
Table Printer

Print results tables using rich library, grouped by ML form.
"""

import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

from results_collector import ResultsCollector


class TablePrinter:
    """Print formatted results tables using rich library."""
    
    def __init__(self):
        """Initialize table printer with console."""
        self.console = Console()
    
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
            df: DataFrame with results for this ML form
            ml_form: ML form name (binary_cls, multi_cls, etc.)
            ui_mask: UI mask setting to filter
            checkpoint_type: 'best' or 'last'
        """
        if df.empty:
            self.console.print(f"[yellow]No results for {ml_form}[/yellow]")
            return
        
        # Determine metric columns based on ml_form
        metric_cols = self._get_metric_columns(ml_form, df)
        
        if not metric_cols:
            self.console.print(f"[yellow]No metrics found for {ml_form}[/yellow]")
            return
        
        # Create rich table
        table = Table(
            title=f"{ml_form.upper()} | UI Mask: {ui_mask} | {checkpoint_type.upper()} Checkpoint",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        # Add columns: Task ID, Model, then for each metric: Baseline, Finetuned, Delta
        table.add_column("Task ID", style="cyan", no_wrap=True)
        table.add_column("Model", style="green")
        
        for metric in metric_cols:
            table.add_column(f"{metric.upper()} (B)", justify="right", style="dim")
            table.add_column(f"{metric.upper()} (F)", justify="right", style="white")
            table.add_column("Δ", justify="right")
        
        # Group by task and model
        for task_id in sorted(df['task_id'].unique()):
            df_task = df[df['task_id'] == task_id]
            
            for model in sorted(df_task['model_type'].unique()):
                df_model = df_task[df_task['model_type'] == model]
                
                baseline_row = df_model[df_model['init_type'] == 'baseline']
                finetuned_row = df_model[df_model['init_type'] == 'finetuned']
                
                row_values = [task_id, model]
                
                for metric in metric_cols:
                    baseline_val = baseline_row[metric].values[0] if len(baseline_row) > 0 and pd.notna(baseline_row[metric].values[0]) else None
                    finetuned_val = finetuned_row[metric].values[0] if len(finetuned_row) > 0 and pd.notna(finetuned_row[metric].values[0]) else None
                    
                    baseline_str = f"{baseline_val:.4f}" if baseline_val is not None else "N/A"
                    finetuned_str = f"{finetuned_val:.4f}" if finetuned_val is not None else "N/A"
                    
                    if baseline_val is not None and finetuned_val is not None:
                        delta = finetuned_val - baseline_val
                        delta_style = "green" if delta > 0 else ("red" if delta < 0 else "white")
                        delta_str = f"[{delta_style}]{delta:+.4f}[/{delta_style}]"
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
            df: DataFrame with results for this ML form
            ml_form: ML form name (binary_cls, multi_cls, etc.)
            checkpoint_type: 'best' or 'last'
        """
        if df.empty:
            self.console.print(f"[yellow]No results for {ml_form}[/yellow]")
            return
        
        # Determine metric columns based on ml_form
        metric_cols = self._get_metric_columns(ml_form, df)
        
        # Create rich table
        table = Table(
            title=f"{ml_form.upper()} - {checkpoint_type.upper()} Checkpoint",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        # Add columns
        table.add_column("Task ID", style="cyan", no_wrap=True)
        table.add_column("Model", style="green")
        table.add_column("UI Mask", style="blue")
        table.add_column("Init", style="yellow")
        table.add_column("Category", style="dim")
        
        # Add metric columns
        for metric in metric_cols:
            table.add_column(metric.upper(), justify="right", style="white")
        
        # Add rows
        for _, row in df.iterrows():
            metric_values = [
                f"{row[metric]:.4f}" if pd.notna(row.get(metric)) else "N/A"
                for metric in metric_cols
            ]
            
            init_type = row.get('init_type', 'unknown')
            init_style = "green" if init_type == 'finetuned' else "dim"
            
            table.add_row(
                row['task_id'],
                row['model_type'],
                row.get('ui_mask', 'unknown'),
                f"[{init_style}]{init_type}[/{init_style}]",
                row['category'],
                *metric_values
            )
        
        self.console.print(table)
        self.console.print()
    
    def _get_metric_columns(self, ml_form: str, df: pd.DataFrame) -> List[str]:
        """
        Get metric column names for a specific ML form.
        
        Args:
            ml_form: ML form name
            df: DataFrame to check for available columns
        
        Returns:
            List of metric column names
        """
        # Define expected metrics for each ML form
        metric_map = {
            'binary_cls': ['acc', 'f1', 'auroc'],
            'multi_cls': ['acc', 'f1', 'acc_top3', 'acc_top5'],
            'multi_label_cls': ['acc_subset', 'acc_hamming', 'f1', 'auroc'],
            'regression': ['mse', 'mae', 'r2']
        }
        
        expected_metrics = metric_map.get(ml_form, [])
        
        # Only return metrics that exist in the dataframe
        available_metrics = [m for m in expected_metrics if m in df.columns]
        
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
            df: DataFrame with all results
            checkpoint_type: 'best' or 'last'
        """
        if df.empty:
            return
        
        model_types = df['model_type'].unique()
        
        if len(model_types) <= 1:
            self.console.print("[yellow]Only one model type found, skipping comparison[/yellow]")
            return
        
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
                
                metric_avgs = [
                    f"{df_model[metric].mean():.4f}" if metric in df_model.columns else "N/A"
                    for metric in metric_cols
                ]
                
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
            df: DataFrame with all results
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
                        
                        metric_avgs = [
                            f"{df_init[metric].mean():.4f}" if metric in df_init.columns else "N/A"
                            for metric in metric_cols
                        ]
                        
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
            df: DataFrame with all results
            checkpoint_type: 'best' or 'last'
        """
        if df.empty:
            return
        
        if 'init_type' not in df.columns:
            return
        
        self.console.rule(f"[bold blue]Task-Level Baseline vs Finetuned - {checkpoint_type.upper()}[/bold blue]")
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
                        
                        baseline_val = df_ui[df_ui['init_type'] == 'baseline'][primary_metric].values
                        finetuned_val = df_ui[df_ui['init_type'] == 'finetuned'][primary_metric].values
                        
                        baseline_str = f"{baseline_val[0]:.4f}" if len(baseline_val) > 0 else "N/A"
                        finetuned_str = f"{finetuned_val[0]:.4f}" if len(finetuned_val) > 0 else "N/A"
                        
                        if len(baseline_val) > 0 and len(finetuned_val) > 0:
                            delta = finetuned_val[0] - baseline_val[0]
                            delta_style = "green" if delta > 0 else ("red" if delta < 0 else "white")
                            delta_str = f"[{delta_style}]{delta:+.4f}[/{delta_style}]"
                        else:
                            delta_str = "N/A"
                        
                        table.add_row(
                            task_id,
                            model,
                            ui_mask,
                            baseline_str,
                            finetuned_str,
                            delta_str
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
    df = collector.create_results_dataframe(all_results, args.checkpoint)
    
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
    
    printer.console.print("\n")
    printer.console.rule(
        f"[bold blue]Results: UI Mask={args.ui_mask} | Checkpoint={args.checkpoint.upper()}[/bold blue]"
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
    
    printer.console.print(f"  Total experiments: {len(df_filtered)}")
    printer.console.print(f"  Unique tasks: {n_tasks}")
    printer.console.print(f"  Models: {df_filtered['model_type'].unique().tolist()}")
    printer.console.print(f"  Baseline results: {n_baseline}")
    printer.console.print(f"  Finetuned results: {n_finetuned}")
    printer.console.print()


if __name__ == '__main__':
    main()
