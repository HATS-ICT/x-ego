"""
Test Analysis and Visualization for Location Prediction Models.

This module provides utilities for analyzing test results, computing metrics,
and creating visualizations for location prediction tasks.
"""

import torch
import numpy as np
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
        # Get checkpoint-specific prefix (test/last or test/best)
        checkpoint_name = getattr(self.model, 'checkpoint_name', 'last')
        base_prefix = f'test/{checkpoint_name}'
        
        for team in ['ct', 't']:
            if team not in team_specific_metrics:
                continue
            
            metrics = team_specific_metrics[team]
            team_prefix = f'{base_prefix}/{team.lower()}'
            
            if self.task_form == 'traj-gen':
                self.model.safe_log(f'{team_prefix}_mse', metrics['mse'], on_epoch=True)
                self.model.safe_log(f'{team_prefix}_mae', metrics['mae'], on_epoch=True)
                self.model.safe_log(f'{team_prefix}_ade', metrics['ade'], on_epoch=True)
                self.model.safe_log(f'{team_prefix}_fde', metrics['fde'], on_epoch=True)
                
                # Log horizon-specific metrics
                horizons = [1, 3, 5, 10, 15]
                for h in horizons:
                    if f'ade@{h}s' in metrics:
                        self.model.safe_log(f'{team_prefix}_ade@{h}s', metrics[f'ade@{h}s'], on_epoch=True)
                    if f'fde@{h}s' in metrics:
                        self.model.safe_log(f'{team_prefix}_fde@{h}s', metrics[f'fde@{h}s'], on_epoch=True)
            
            elif self.task_form in ['coord-reg', 'coord-gen']:
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
        # Get checkpoint-specific prefix (test/last or test/best)
        checkpoint_name = getattr(self.model, 'checkpoint_name', 'last')
        base_prefix = f'test/{checkpoint_name}'
        
        if self.task_form == 'traj-gen':
            # Log trajectory-specific metrics
            if 'ade' in test_results:
                self.model.safe_log(f'{base_prefix}/ade', test_results['ade'], on_epoch=True)
            if 'fde' in test_results:
                self.model.safe_log(f'{base_prefix}/fde', test_results['fde'], on_epoch=True)
            
            # Log horizon-specific metrics
            horizons = [1, 3, 5, 10, 15]
            for h in horizons:
                if f'ade@{h}s' in test_results:
                    self.model.safe_log(f'{base_prefix}/ade@{h}s', test_results[f'ade@{h}s'], on_epoch=True)
                if f'fde@{h}s' in test_results:
                    self.model.safe_log(f'{base_prefix}/fde@{h}s', test_results[f'fde@{h}s'], on_epoch=True)
        
        elif self.task_form in ['coord-reg', 'coord-gen']:
            if 'geometric_distances' in test_results:
                geom = test_results['geometric_distances']
                self.model.safe_log(f'{base_prefix}/chamfer_distance', 
                             geom['chamfer_distance_mean'], on_epoch=True)
                self.model.safe_log(f'{base_prefix}/wasserstein_distance',
                             geom['wasserstein_distance_mean'], on_epoch=True)
        
        elif self.task_form in ['multi-label-cls', 'grid-cls']:
            # Multi-label classification metrics
            metric_names = ['hamming_loss', 'hamming_accuracy', 'subset_accuracy', 'micro_f1', 'macro_f1']
            for metric_name in metric_names:
                if metric_name in test_results:
                    self.model.safe_log(f'{base_prefix}/{metric_name}', 
                                 test_results[metric_name], on_epoch=True)
        
        elif self.task_form in ['multi-output-reg', 'density-cls']:
            metric_names = ['exact_accuracy', 'l1_count_error', 'kl_divergence', 'multinomial_loss']
            for metric_name in metric_names:
                if metric_name in test_results:
                    self.model.safe_log(f'{base_prefix}/{metric_name}', 
                                 test_results[metric_name], on_epoch=True)
    
    def add_coordinate_metadata(self, test_results):
        """Add coordinate-specific metadata to test results."""
        import pickle
        from pathlib import Path
        
        test_results['loss_function'] = self.model.loss_computer.loss_fn
        
        # Check if scaler file exists
        scaler_path = Path(self.model.scaler_path)
        test_results['coordinate_scaling'] = scaler_path.exists()
        
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            test_results['scaler_data_min'] = scaler.data_min_.tolist()
            test_results['scaler_data_max'] = scaler.data_max_.tolist()
            test_results['scaler_scale'] = scaler.scale_.tolist()
        
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
        for team in ['t', 'ct']:
            combined_samples.extend(test_raw_samples_by_team[team])
        
        if len(combined_samples) == 0:
            return
        
        print(f"Creating KDE heatmaps for {len(combined_samples)} test samples...")
        print(f"  t samples: {len(test_raw_samples_by_team['t'])}")
        print(f"  ct samples: {len(test_raw_samples_by_team['ct'])}")
        
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
            # Note: multi_predictions are unscaled, so we need to scale them
            # However, since our data is now normalized, predictions are already normalized
            # So we just convert to tensor
            first_pred_normalized = torch.tensor(multi_predictions[0:1], dtype=torch.float32)
            scaled_pred = first_pred_normalized
            
            # Get target key dynamically
            if 'enemy_locations' in sample:
                target_key = 'enemy_locations'
            elif 'future_locations' in sample:
                target_key = 'future_locations'
            elif 'trajectories' in sample:
                target_key = 'trajectories'
            else:
                target_key = 'teammate_locations'
            scaled_target = sample[target_key]  # Already normalized
            
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
            
            if self.task_form in ['coord-reg', 'coord-gen', 'traj-gen']:
                pred_unscaled = self.model.unscale_coordinates(pred)
                predictions.append(pred_unscaled.cpu().numpy()[0])
            else:
                raise NotImplementedError(
                    "Multiple predictions generation only supported for coordinate-based tasks"
                )
        
        predictions = np.array(predictions)  # [num_predictions, 5, 3]
        
        # Get unscaled ground truth
        if 'enemy_locations' in sample:
            target_key = 'enemy_locations'
        elif 'future_locations' in sample:
            target_key = 'future_locations'
        elif 'trajectories' in sample:
            target_key = 'trajectories'
        else:
            target_key = 'teammate_locations'
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

