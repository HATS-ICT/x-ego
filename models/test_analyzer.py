"""
Test Analysis and Visualization for Location Prediction Models.

This module provides utilities for analyzing test results, computing metrics,
and creating visualizations for multi-label classification location prediction.
"""

import json

from utils.plot_utils import create_prediction_plots
from utils.serialization_utils import json_serializable


class TestAnalyzer:
    """
    Handles test analysis, visualization, and metric logging for location prediction models.
    """
    
    def __init__(self, model, cfg, metrics_calculator):
        """
        Initialize TestAnalyzer.
        
        Args:
            model: The model being tested (CrossEgoVideoLocationNet)
            cfg: Configuration object
            metrics_calculator: MetricsCalculator instance
        """
        self.model = model
        self.cfg = cfg
        self.metrics_calculator = metrics_calculator
    
    def log_team_metrics(self, team_specific_metrics):
        """Log team-specific metrics to logger."""
        # Get checkpoint-specific prefix (test/last or test/best)
        checkpoint_name = getattr(self.model, 'checkpoint_name', 'last')
        base_prefix = f'test/{checkpoint_name}'
        
        for team in ['ct', 't']:
            if team not in team_specific_metrics:
                continue
            
            metrics = team_specific_metrics[team]
            team_prefix = f'{base_prefix}/{team.lower()}'
            
            # Multi-label classification metrics
            if 'hamming_loss' in metrics:
                self.model.safe_log(f'{team_prefix}_hamming_loss', 
                             metrics['hamming_loss'], on_epoch=True)
            if 'hamming_accuracy' in metrics:
                self.model.safe_log(f'{team_prefix}_hamming_accuracy',
                             metrics['hamming_accuracy'], on_epoch=True)
            if 'subset_accuracy' in metrics:
                self.model.safe_log(f'{team_prefix}_subset_accuracy',
                             metrics['subset_accuracy'], on_epoch=True)
            if 'micro_f1' in metrics:
                self.model.safe_log(f'{team_prefix}_micro_f1',
                             metrics['micro_f1'], on_epoch=True)
            if 'macro_f1' in metrics:
                self.model.safe_log(f'{team_prefix}_macro_f1',
                             metrics['macro_f1'], on_epoch=True)
    
    def log_overall_metrics(self, test_results):
        """Log overall test metrics to logger."""
        # Get checkpoint-specific prefix (test/last or test/best)
        checkpoint_name = getattr(self.model, 'checkpoint_name', 'last')
        base_prefix = f'test/{checkpoint_name}'
        
        # Multi-label classification metrics
        metric_names = ['hamming_loss', 'hamming_accuracy', 'subset_accuracy', 'micro_f1', 'macro_f1']
        for metric_name in metric_names:
            if metric_name in test_results:
                self.model.safe_log(f'{base_prefix}/{metric_name}', 
                             test_results[metric_name], on_epoch=True)
    
    def save_results_to_json(self, test_results, plots_dir):
        """Save test results to JSON file."""
        results_file = plots_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=json_serializable)
        print(f"Test results saved to: {results_file}")
    
    def create_visualization_plots(self, predictions, targets, plots_dir, pov_team_sides):
        """Create visualization plots for test results."""
        create_prediction_plots(predictions, targets, plots_dir, pov_team_sides)
