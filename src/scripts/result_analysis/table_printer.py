#!/usr/bin/env python3
"""
Table Printer

Print results tables using rich library, grouped by ML form.
"""

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
        table.add_column("Category", style="blue")
        table.add_column("Temporal", style="yellow")
        table.add_column("Horizon", justify="right", style="yellow")
        
        # Add metric columns
        for metric in metric_cols:
            table.add_column(metric.upper(), justify="right", style="white")
        
        # Add rows
        for _, row in df.iterrows():
            metric_values = [
                f"{row[metric]:.4f}" if pd.notna(row.get(metric)) else "N/A"
                for metric in metric_cols
            ]
            
            horizon_str = str(int(row['horizon_sec'])) if pd.notna(row['horizon_sec']) else "N/A"
            
            table.add_row(
                row['task_id'],
                row['model_type'],
                row['category'],
                row['temporal_type'],
                horizon_str,
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


def main():
    """Main function to print all result tables."""
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
    
    # Print tables for best checkpoint
    printer.console.print("\n")
    results_by_ml_form_best = collector.get_results_by_ml_form(all_results, 'best')
    printer.print_all_tables(results_by_ml_form_best, 'best')
    
    df_best = collector.create_results_dataframe(all_results, 'best')
    printer.print_summary_statistics(df_best)
    printer.print_model_comparison(df_best, 'best')
    
    # Print tables for last checkpoint
    printer.console.print("\n")
    results_by_ml_form_last = collector.get_results_by_ml_form(all_results, 'last')
    printer.print_all_tables(results_by_ml_form_last, 'last')
    
    df_last = collector.create_results_dataframe(all_results, 'last')
    printer.print_summary_statistics(df_last)
    printer.print_model_comparison(df_last, 'last')


if __name__ == '__main__':
    main()
