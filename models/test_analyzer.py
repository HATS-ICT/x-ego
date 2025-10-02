"""
Test Analysis and Visualization for Location Prediction Models.

This module provides utilities for analyzing test results, computing metrics,
and creating visualizations for location prediction tasks.
"""

import torch
import numpy as np
from pathlib import Path
import json

from utils.plot_utils import create_prediction_plots, create_prediction_heatmaps_grid
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
        self.task_form = cfg.data.task_form
    
    def log_team_metrics(self, team_specific_metrics):
        """Log team-specific metrics to logger."""
        for team in ['CT', 'T']:
            if team not in team_specific_metrics:
                continue
            
            metrics = team_specific_metrics[team]
            team_prefix = f'test/{team.lower()}'
            
            if self.task_form in ['coord-reg', 'coord-gen']:
                self.model.safe_log(f'{team_prefix}_mse', metrics['mse'], on_epoch=True)
                self.model.safe_log(f'{team_prefix}_mae', metrics['mae'], on_epoch=True)
                
                if 'geometric_distances' in metrics:
                    geom = metrics['geometric_distances']
                    self.model.safe_log(f'{team_prefix}_chamfer_distance', 
                                 geom['chamfer_distance_mean'], on_epoch=True)
                    self.model.safe_log(f'{team_prefix}_wasserstein_distance',
                                 geom['wasserstein_distance_mean'], on_epoch=True)
            
            elif self.task_form in ['multi-label-cls', 'grid-cls']:
                # Multi-label classification metrics
                if 'hamming_loss' in metrics:
                    self.model.safe_log(f'{team_prefix}_hamming_loss', 
                                 metrics['hamming_loss'], on_epoch=True)
                if 'subset_accuracy' in metrics:
                    self.model.safe_log(f'{team_prefix}_subset_accuracy',
                                 metrics['subset_accuracy'], on_epoch=True)
                if 'micro_f1' in metrics:
                    self.model.safe_log(f'{team_prefix}_micro_f1',
                                 metrics['micro_f1'], on_epoch=True)
                if 'macro_f1' in metrics:
                    self.model.safe_log(f'{team_prefix}_macro_f1',
                                 metrics['macro_f1'], on_epoch=True)
            
            elif self.task_form in ['multi-output-reg', 'density-cls']:
                if 'exact_accuracy' in metrics:
                    self.model.safe_log(f'{team_prefix}_exact_accuracy', 
                                 metrics['exact_accuracy'], on_epoch=True)
                if 'l1_count_error' in metrics:
                    self.model.safe_log(f'{team_prefix}_l1_count_error',
                                 metrics['l1_count_error'], on_epoch=True)
                if 'kl_divergence' in metrics:
                    self.model.safe_log(f'{team_prefix}_kl_divergence',
                                 metrics['kl_divergence'], on_epoch=True)
    
    def log_overall_metrics(self, test_results):
        """Log overall test metrics to logger."""
        if self.task_form in ['coord-reg', 'coord-gen']:
            if 'geometric_distances' in test_results:
                geom = test_results['geometric_distances']
                self.model.safe_log('test/chamfer_distance', 
                             geom['chamfer_distance_mean'], on_epoch=True)
                self.model.safe_log('test/wasserstein_distance',
                             geom['wasserstein_distance_mean'], on_epoch=True)
        
        elif self.task_form in ['multi-label-cls', 'grid-cls']:
            # Multi-label classification metrics
            metric_names = ['hamming_loss', 'subset_accuracy', 'micro_f1', 'macro_f1']
            for metric_name in metric_names:
                if metric_name in test_results:
                    self.model.safe_log(f'test/{metric_name}', 
                                 test_results[metric_name], on_epoch=True)
        
        elif self.task_form in ['multi-output-reg', 'density-cls']:
            metric_names = ['exact_accuracy', 'l1_count_error', 'kl_divergence', 'multinomial_loss']
            for metric_name in metric_names:
                if metric_name in test_results:
                    self.model.safe_log(f'test/{metric_name}', 
                                 test_results[metric_name], on_epoch=True)
    
    def add_coordinate_metadata(self, test_results):
        """Add coordinate-specific metadata to test results."""
        test_results['loss_function'] = self.model.loss_computer.loss_fn
        test_results['coordinate_scaling'] = self.model.coordinate_scaler is not None
        
        if self.model.coordinate_scaler is not None:
            test_results['scaler_data_min'] = self.model.coordinate_scaler.data_min_.tolist()
            test_results['scaler_data_max'] = self.model.coordinate_scaler.data_max_.tolist()
            test_results['scaler_scale'] = self.model.coordinate_scaler.scale_.tolist()
        
        if self.model.loss_computer.loss_fn == 'sinkhorn':
            test_results['sinkhorn_blur'] = self.model.loss_computer.sinkhorn_blur
            test_results['sinkhorn_scaling'] = self.model.loss_computer.sinkhorn_scaling
        
        if self.task_form == 'coord-gen':
            test_results['latent_dim'] = self.model.latent_dim
            test_results['kl_weight'] = self.cfg.model.vae.kl_weight
    
    def create_prediction_heatmaps(self, plots_dir, test_raw_samples_by_team):
        """Create KDE heatmaps for selected test samples."""
        # Combine samples from both teams
        combined_samples = []
        for team in ['T', 'CT']:
            combined_samples.extend(test_raw_samples_by_team[team])
        
        if len(combined_samples) == 0:
            return
        
        print(f"Creating KDE heatmaps for {len(combined_samples)} test samples...")
        print(f"  T samples: {len(test_raw_samples_by_team['T'])}")
        print(f"  CT samples: {len(test_raw_samples_by_team['CT'])}")
        
        predictions_list = []
        targets_list = []
        pov_team_sides_list = []
        scaled_predictions_list = []
        scaled_targets_list = []
        
        for sample in combined_samples:
            # Generate multiple predictions for this sample
            multi_predictions, target = self.generate_multiple_predictions(
                sample, num_predictions=100
            )
            
            # Get scaled version for Chamfer distance calculation
            first_pred_unscaled = torch.tensor(multi_predictions[0:1], dtype=torch.float32)
            
            if self.model.coordinate_scaler is not None:
                first_pred_flat = first_pred_unscaled.view(-1, 3).numpy()
                first_pred_scaled = self.model.coordinate_scaler.transform(first_pred_flat)
                scaled_pred = torch.tensor(first_pred_scaled.reshape(1, 5, 3), 
                                         dtype=torch.float32)
            else:
                scaled_pred = first_pred_unscaled
            
            # Get target key dynamically (supports both enemy_locations and future_locations)
            target_key = 'enemy_locations' if 'enemy_locations' in sample else 'future_locations'
            scaled_target = sample[target_key]  # Already scaled
            
            # Move to device
            device = next(self.model.parameters()).device
            scaled_pred = scaled_pred.to(device)
            scaled_target = scaled_target.to(device)
            
            predictions_list.append(multi_predictions)
            targets_list.append(target)
            pov_team_sides_list.append(sample['pov_team_side'])
            scaled_predictions_list.append(scaled_pred)
            scaled_targets_list.append(scaled_target)
        
        # Create heatmap grid
        create_prediction_heatmaps_grid(
            predictions_list, targets_list, pov_team_sides_list,
            scaled_predictions_list, scaled_targets_list,
            plots_dir, map_name="de_mirage"
        )
    
    @torch.inference_mode()
    def generate_multiple_predictions(self, sample, num_predictions=100):
        """
        Generate multiple predictions for a single test sample.
        
        Args:
            sample: Dictionary containing 'video', 'pov_team_side_encoded', 'enemy_locations'
            num_predictions: Number of predictions to generate
            
        Returns:
            predictions: numpy array of shape [num_predictions, 5, 3] (unscaled)
            target: numpy array of shape [5, 3] (unscaled ground truth)
        """
        self.model.eval()
        
        predictions = []
        for _ in range(num_predictions):
            if self.task_form == 'coord-gen':
                outputs = self.model.forward(sample, mode='sampling')
            else:
                outputs = self.model.forward(sample, mode='full')
            
            pred = outputs['predictions']  # [1, 5, 3]
            
            if self.task_form in ['coord-reg', 'coord-gen']:
                pred_unscaled = self.model.unscale_coordinates(pred)
                predictions.append(pred_unscaled.cpu().numpy()[0])
            else:
                raise NotImplementedError(
                    "Multiple predictions generation only supported for coordinate-based tasks"
                )
        
        predictions = np.array(predictions)  # [num_predictions, 5, 3]
        
        # Get unscaled ground truth
        target_key = 'enemy_locations' if 'enemy_locations' in sample else 'future_locations'
        target_unscaled = self.model.unscale_coordinates(sample[target_key])
        target = target_unscaled.cpu().numpy()[0]  # [5, 3]
        
        return predictions, target
    
    def save_results_to_json(self, test_results, plots_dir):
        """Save test results to JSON file."""
        results_file = plots_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=json_serializable)
        print(f"Test results saved to: {results_file}")
    
    def create_visualization_plots(self, predictions, targets, plots_dir, pov_team_sides):
        """Create visualization plots for test results."""
        create_prediction_plots(self.task_form, predictions, targets, plots_dir, pov_team_sides)

