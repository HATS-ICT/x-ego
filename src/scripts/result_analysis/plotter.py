#!/usr/bin/env python3
"""
Results Plotter

Generate all plots for results analysis and save them to artifacts/results.
"""

from pathlib import Path
import sys

from results_collector import ResultsCollector
from plotting_utils import (
    plot_performance_by_category,
    plot_temporal_horizon_performance,
    plot_metric_distributions,
    plot_category_heatmap,
    plot_ml_form_comparison,
    plot_model_comparison
)


class ResultsPlotter:
    """Generate and save all result plots."""
    
    def __init__(self, output_dir: Path, artifacts_dir: Path):
        """
        Initialize results plotter.
        
        Args:
            output_dir: Path to artifacts/results directory
            artifacts_dir: Path to save plots
        """
        self.output_dir = Path(output_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_all(
        self, 
        collector: ResultsCollector,
        all_results: dict
    ):
        """
        Generate all plots for both best and last checkpoints.
        
        Args:
            collector: ResultsCollector instance
            all_results: Results dictionary from collector
        """
        if not all_results:
            print("No results to plot!")
            return
        
        print("\n" + "="*60)
        print("Generating plots...")
        print("="*60 + "\n")
        
        # Generate plots for both checkpoint types
        for checkpoint_type in ['best', 'last']:
            print(f"\n--- {checkpoint_type.upper()} Checkpoint ---\n")
            
            df = collector.create_results_dataframe(all_results, checkpoint_type)
            
            if df.empty:
                print(f"No data for {checkpoint_type} checkpoint")
                continue
            
            # Determine primary metric based on most common ML form
            primary_metric = self._get_primary_metric(df)
            
            # 1. Performance by category
            output_path = self.artifacts_dir / f'performance_by_category_{checkpoint_type}.png'
            plot_performance_by_category(df, checkpoint_type, output_path, primary_metric)
            
            # 2. Temporal horizon performance
            output_path = self.artifacts_dir / f'temporal_horizon_{checkpoint_type}.png'
            plot_temporal_horizon_performance(df, checkpoint_type, output_path, primary_metric)
            
            # 3. Metric distributions
            output_path = self.artifacts_dir / f'metric_distributions_{checkpoint_type}.png'
            plot_metric_distributions(df, checkpoint_type, output_path)
            
            # 4. Category heatmap
            output_path = self.artifacts_dir / f'category_heatmap_{checkpoint_type}.png'
            plot_category_heatmap(df, checkpoint_type, output_path, primary_metric)
            
            # 5. ML form comparison
            output_path = self.artifacts_dir / f'ml_form_comparison_{checkpoint_type}.png'
            plot_ml_form_comparison(df, checkpoint_type, output_path)
            
            # 6. Model comparison (if multiple models)
            if len(df['model_type'].unique()) > 1:
                output_path = self.artifacts_dir / f'model_comparison_{checkpoint_type}.png'
                plot_model_comparison(df, checkpoint_type, output_path)
        
        print("\n" + "="*60)
        print(f"All plots saved to: {self.artifacts_dir}")
        print("="*60 + "\n")
    
    def _get_primary_metric(self, df):
        """
        Determine primary metric based on ML forms present.
        
        Args:
            df: DataFrame with results
        
        Returns:
            Primary metric name
        """
        # Count ML forms
        ml_form_counts = df['ml_form'].value_counts()
        most_common = ml_form_counts.index[0]
        
        # Map to primary metric
        metric_map = {
            'binary_cls': 'acc',
            'multi_cls': 'acc',
            'multi_label_cls': 'acc_subset',
            'regression': 'mae'
        }
        
        return metric_map.get(most_common, 'acc')


def main():
    """Main function to generate all plots."""
    project_root = Path(__file__).parent.parent.parent.parent
    output_dir = project_root / 'output'
    task_defs = project_root / 'data' / 'labels' / 'task_definitions.csv'
    artifacts_dir = project_root / 'artifacts' / 'results'
    
    # Check if task definitions exist
    if not task_defs.exists():
        print(f"Error: Task definitions not found at {task_defs}")
        sys.exit(1)
    
    # Check if output directory exists
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
    
    print(f"Found {len(all_results)} experiments with results")
    
    # Create plotter and generate all plots
    plotter = ResultsPlotter(output_dir, artifacts_dir)
    plotter.plot_all(collector, all_results)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
