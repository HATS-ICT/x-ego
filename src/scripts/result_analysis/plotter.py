#!/usr/bin/env python3
"""
Results Plotter

Generate all plots for results analysis and save them to artifacts/results.
"""

from pathlib import Path
import sys

from results_collector import ResultsCollector
from plotting_utils import (
    plot_baseline_vs_finetuned_per_model,
    plot_by_task_prefix,
    plot_time_horizon_lines,
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
        
        Organizes plots into subfolders: {checkpoint_type}/{ui_mask}/
        
        For each ML form, for each UI setting:
        1. Baseline vs finetuned on all tasks (one plot per model)
        2. Group by prefix: enemy, self, global, teammate
        3. Line plots for tasks with time horizon suffix
        
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
            
            # Get unique ML forms and UI masks
            ml_forms = df['ml_form'].unique()
            ui_masks = df['ui_mask'].unique()
            
            print(f"ML forms: {list(ml_forms)}")
            print(f"UI masks: {list(ui_masks)}")
            
            # For each UI setting
            for ui_mask in ui_masks:
                # Create subfolder for this checkpoint/ui_mask combination
                subfolder = self.artifacts_dir / checkpoint_type / ui_mask
                subfolder.mkdir(parents=True, exist_ok=True)
                
                print(f"\n  UI Mask: {ui_mask}")
                
                # For each ML form
                for ml_form in ml_forms:
                    df_filtered = df[(df['ml_form'] == ml_form) & (df['ui_mask'] == ui_mask)]
                    
                    if df_filtered.empty:
                        continue
                    
                    print(f"    ML Form: {ml_form}")
                    
                    # 1. Baseline vs finetuned per model
                    plot_baseline_vs_finetuned_per_model(
                        df_filtered, ml_form, ui_mask, checkpoint_type, subfolder
                    )
                    
                    # 2. Group by task prefix
                    plot_by_task_prefix(
                        df_filtered, ml_form, ui_mask, checkpoint_type, subfolder
                    )
                    
                    # 3. Time horizon line plots
                    plot_time_horizon_lines(
                        df_filtered, ml_form, ui_mask, checkpoint_type, subfolder
                    )
        
        print("\n" + "="*60)
        print(f"All plots saved to: {self.artifacts_dir}")
        print("="*60 + "\n")


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
