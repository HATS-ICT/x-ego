"""
Metrics Calculation for Multi-Agent Location Prediction

This module provides comprehensive metrics calculation for different task formulations:
- Coordinate regression: MSE, MAE, Chamfer distance, Wasserstein distance
- Multi-label classification: Hamming loss, subset accuracy, micro/macro F1
- Count regression: Exact match, L1 error, KL divergence
- Density estimation: MSE, MAE, KL divergence
"""

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from sklearn.metrics import hamming_loss, accuracy_score, f1_score

from utils.metric_utils import (
    exact_match_accuracy, l1_count_error, 
    kl_divergence_histogram, chamfer_distance_batch
)


class MetricsCalculator:
    """Handles metrics calculation for different task formulations."""
    
    def __init__(self, task_form):
        """
        Initialize metrics calculator.
        
        Args:
            task_form: Type of task ('coord-reg', 'coord-gen', 'multi-label-cls', etc.)
        """
        self.task_form = task_form
    
    def calculate_metrics(self, predictions, targets):
        """
        Calculate comprehensive test metrics based on task form.
        
        Args:
            predictions: numpy array of predictions
            targets: numpy array of targets
            
        Returns:
            Dictionary of calculated metrics
        """
        if self.task_form in ['coord-reg', 'coord-gen']:
            return self._calculate_coordinate_metrics(predictions, targets)
        elif self.task_form in ['multi-label-cls', 'grid-cls']:
            return self._calculate_classification_metrics(predictions, targets)
        elif self.task_form == 'multi-output-reg':
            return self._calculate_count_regression_metrics(predictions, targets)
        elif self.task_form == 'density-cls':
            return self._calculate_density_metrics(predictions, targets)
        else:
            raise ValueError(f"Unknown task_form: {self.task_form}")
    
    def _calculate_coordinate_metrics(self, predictions, targets):
        """Calculate metrics for coordinate regression tasks."""
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        
        # Per-dimension metrics
        per_dim_metrics = {}
        predictions_flat = predictions.reshape(-1, 15)
        targets_flat = targets.reshape(-1, 15)
        
        for i in range(5):  # 5 agents
            for j, coord in enumerate(['X', 'Y', 'Z']):
                dim_idx = i * 3 + j
                pred_dim = predictions_flat[:, dim_idx]
                target_dim = targets_flat[:, dim_idx]
                per_dim_metrics[f'player_{i}_{coord}_mse'] = float(np.mean((pred_dim - target_dim) ** 2))
                per_dim_metrics[f'player_{i}_{coord}_mae'] = float(np.mean(np.abs(pred_dim - target_dim)))
        
        return {
            'overall_mse': float(mse),
            'overall_mae': float(mae),
            'per_dimension_metrics': per_dim_metrics,
            'num_samples': len(targets),
            'predictions_shape': list(predictions.shape),
            'targets_shape': list(targets.shape)
        }
    
    def _calculate_classification_metrics(self, predictions, targets):
        """Calculate metrics for multi-label classification tasks."""
        predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
        pred_probs = torch.sigmoid(predictions_tensor).numpy()
        pred_binary = (pred_probs > 0.5).astype(int)
        targets_int = targets.astype(int)
        
        # Multi-label classification metrics using sklearn
        hamming = hamming_loss(targets_int, pred_binary)
        hamming_acc = 1.0 - hamming
        subset_accuracy = accuracy_score(targets_int, pred_binary)
        micro_f1 = f1_score(targets_int, pred_binary, average='micro', zero_division=0)
        macro_f1 = f1_score(targets_int, pred_binary, average='macro', zero_division=0)
        
        # Per-label F1 scores for detailed analysis
        per_label_f1 = f1_score(targets_int, pred_binary, average=None, zero_division=0)
        
        # Per-cell metrics
        per_dim_metrics = {}
        for i in range(predictions.shape[1]):
            per_dim_metrics[f'label_{i}_f1'] = float(per_label_f1[i])
            per_dim_metrics[f'label_{i}_accuracy'] = float(np.mean(pred_binary[:, i] == targets_int[:, i]))
        
        return {
            'hamming_loss': float(hamming),
            'hamming_accuracy': float(hamming_acc),
            'subset_accuracy': float(subset_accuracy),
            'micro_f1': float(micro_f1),
            'macro_f1': float(macro_f1),
            'per_dimension_metrics': per_dim_metrics,
            'num_samples': len(targets),
            'predictions_shape': list(predictions.shape),
            'targets_shape': list(targets.shape)
        }
    
    def _calculate_count_regression_metrics(self, predictions, targets):
        """Calculate metrics for count regression tasks."""
        pred_counts = predictions
        
        # Count-specific metrics
        exact_accuracy = exact_match_accuracy(pred_counts, targets)
        l1_error = l1_count_error(pred_counts, targets)
        
        # Convert to distribution for KL divergence
        pred_probs = pred_counts / (pred_counts.sum(axis=1, keepdims=True) + 1e-8)
        kl_div = kl_divergence_histogram(pred_probs, targets, n_agents=5)
        
        # Per-place metrics
        per_dim_metrics = {}
        for i in range(predictions.shape[1]):
            pred_place = pred_counts[:, i]
            target_place = targets[:, i]
            per_dim_metrics[f'place_{i}_accuracy'] = float(np.mean(np.round(pred_place) == target_place))
            per_dim_metrics[f'place_{i}_pred_count_mean'] = float(np.mean(pred_place))
            per_dim_metrics[f'place_{i}_actual_count_mean'] = float(np.mean(target_place))
        
        mse_loss = np.mean((pred_counts - targets) ** 2)
        
        return {
            'mse_loss': float(mse_loss),
            'exact_accuracy': float(exact_accuracy),
            'l1_count_error': float(l1_error),
            'kl_divergence': float(kl_div),
            'per_dimension_metrics': per_dim_metrics,
            'num_samples': len(targets),
            'predictions_shape': list(predictions.shape),
            'targets_shape': list(targets.shape)
        }
    
    def _calculate_density_metrics(self, predictions, targets):
        """Calculate metrics for density distribution tasks."""
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        
        # KL divergence for density distributions
        pred_probs = predictions / (predictions.sum(axis=1, keepdims=True) + 1e-8)
        target_probs = targets / (targets.sum(axis=1, keepdims=True) + 1e-8)
        kl_div = np.mean(np.sum(target_probs * np.log((target_probs + 1e-8) / (pred_probs + 1e-8)), axis=1))
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'kl_divergence': float(kl_div),
            'per_dimension_metrics': {},
            'num_samples': len(targets),
            'predictions_shape': list(predictions.shape),
            'targets_shape': list(targets.shape)
        }
    
    def calculate_geometric_distances(self, pred_tensors, target_tensors):
        """
        Calculate Chamfer and Wasserstein distances for coordinate-based tasks.
        
        Args:
            pred_tensors: List of prediction tensors
            target_tensors: List of target tensors
            
        Returns:
            Dictionary with geometric distance metrics
        """
        if self.task_form not in ['coord-reg', 'coord-gen']:
            raise ValueError("Geometric distances only for coord-reg/coord-gen tasks")
        
        all_predictions = torch.cat(pred_tensors, dim=0).float()
        all_targets = torch.cat(target_tensors, dim=0).float()
        
        # Chamfer distances
        chamfer_distances = chamfer_distance_batch(all_predictions, all_targets).cpu().numpy()
        
        # Wasserstein distances
        wasserstein_distances = []
        for i in range(all_predictions.shape[0]):
            pred_flat = all_predictions[i].cpu().numpy().flatten()
            target_flat = all_targets[i].cpu().numpy().flatten()
            wd = wasserstein_distance(pred_flat, target_flat)
            wasserstein_distances.append(wd)
        
        wasserstein_distances = np.array(wasserstein_distances)
        
        return {
            'chamfer_distance_mean': float(np.mean(chamfer_distances)),
            'chamfer_distance_std': float(np.std(chamfer_distances)),
            'wasserstein_distance_mean': float(np.mean(wasserstein_distances)),
            'wasserstein_distance_std': float(np.std(wasserstein_distances)),
            'num_valid_samples': len(chamfer_distances)
        }
    
    def calculate_team_metrics(self, predictions, targets, pov_team_sides, pred_tensors, target_tensors):
        """
        Calculate metrics separately for each team.
        
        Args:
            predictions: numpy array of predictions
            targets: numpy array of targets
            pov_team_sides: numpy array of team labels
            pred_tensors: List of prediction tensors (for geometric distances)
            target_tensors: List of target tensors (for geometric distances)
            
        Returns:
            Dictionary with team-specific metrics
        """
        team_metrics = {}
        
        for team in ['CT', 'T']:
            team_mask = pov_team_sides == team
            if not np.any(team_mask):
                continue
            
            team_predictions = predictions[team_mask]
            team_targets = targets[team_mask]
            
            if len(team_predictions) == 0:
                continue
            
            # Calculate task-specific metrics
            if self.task_form in ['coord-reg', 'coord-gen']:
                team_metrics[team] = self._calculate_team_coordinate_metrics(
                    team_predictions, team_targets, team_mask, pred_tensors, target_tensors
                )
            elif self.task_form in ['multi-label-cls', 'grid-cls']:
                team_metrics[team] = self._calculate_team_classification_metrics(
                    team_predictions, team_targets
                )
            elif self.task_form == 'multi-output-reg':
                team_metrics[team] = self._calculate_team_count_metrics(
                    team_predictions, team_targets
                )
            elif self.task_form == 'density-cls':
                team_metrics[team] = self._calculate_team_density_metrics(
                    team_predictions, team_targets
                )
        
        return team_metrics
    
    def _calculate_team_coordinate_metrics(self, team_predictions, team_targets, 
                                          team_mask, pred_tensors, target_tensors):
        """Calculate coordinate metrics for a specific team."""
        mse = np.mean((team_predictions - team_targets) ** 2)
        mae = np.mean(np.abs(team_predictions - team_targets))
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'num_samples': len(team_predictions)
        }
        
        # Calculate geometric distances
        team_indices = np.where(team_mask)[0]
        team_pred_tensors = []
        team_target_tensors = []
        
        current_idx = 0
        for tensor_batch in pred_tensors:
            batch_size = tensor_batch.shape[0]
            batch_indices = np.arange(current_idx, current_idx + batch_size)
            team_batch_mask = np.isin(batch_indices, team_indices)
            
            if np.any(team_batch_mask):
                team_pred_tensors.append(tensor_batch[team_batch_mask])
            
            current_idx += batch_size
        
        current_idx = 0
        for tensor_batch in target_tensors:
            batch_size = tensor_batch.shape[0]
            batch_indices = np.arange(current_idx, current_idx + batch_size)
            team_batch_mask = np.isin(batch_indices, team_indices)
            
            if np.any(team_batch_mask):
                team_target_tensors.append(tensor_batch[team_batch_mask])
            
            current_idx += batch_size
        
        if team_pred_tensors and team_target_tensors:
            team_pred_combined = torch.cat(team_pred_tensors, dim=0)
            team_target_combined = torch.cat(team_target_tensors, dim=0)
            
            chamfer_distances = chamfer_distance_batch(
                team_pred_combined, team_target_combined
            ).cpu().numpy()
            
            wasserstein_distances = []
            for i in range(team_pred_combined.shape[0]):
                pred_i = team_pred_combined[i].cpu().numpy().flatten()
                target_i = team_target_combined[i].cpu().numpy().flatten()
                wd = wasserstein_distance(pred_i, target_i)
                wasserstein_distances.append(wd)
            
            wasserstein_distances = np.array(wasserstein_distances)
            
            metrics['geometric_distances'] = {
                'chamfer_distance_mean': float(np.mean(chamfer_distances)),
                'chamfer_distance_std': float(np.std(chamfer_distances)),
                'wasserstein_distance_mean': float(np.mean(wasserstein_distances)),
                'wasserstein_distance_std': float(np.std(wasserstein_distances)),
                'num_valid_samples': len(chamfer_distances)
            }
        
        return metrics
    
    def _calculate_team_classification_metrics(self, team_predictions, team_targets):
        """Calculate multi-label classification metrics for a specific team."""
        team_pred_tensor = torch.tensor(team_predictions, dtype=torch.float32)
        team_pred_probs = torch.sigmoid(team_pred_tensor).numpy()
        team_pred_binary = (team_pred_probs > 0.5).astype(int)
        team_targets_int = team_targets.astype(int)
        
        # Multi-label classification metrics using sklearn
        hamming = hamming_loss(team_targets_int, team_pred_binary)
        hamming_acc = 1.0 - hamming
        subset_accuracy = accuracy_score(team_targets_int, team_pred_binary)
        micro_f1 = f1_score(team_targets_int, team_pred_binary, average='micro', zero_division=0)
        macro_f1 = f1_score(team_targets_int, team_pred_binary, average='macro', zero_division=0)
        
        return {
            'hamming_loss': float(hamming),
            'hamming_accuracy': float(hamming_acc),
            'subset_accuracy': float(subset_accuracy),
            'micro_f1': float(micro_f1),
            'macro_f1': float(macro_f1),
            'num_samples': len(team_predictions)
        }
    
    def _calculate_team_count_metrics(self, team_predictions, team_targets):
        """Calculate count regression metrics for a specific team."""
        exact_accuracy = exact_match_accuracy(team_predictions, team_targets)
        l1_error = l1_count_error(team_predictions, team_targets)
        
        team_pred_probs = team_predictions / (team_predictions.sum(axis=1, keepdims=True) + 1e-8)
        kl_div = kl_divergence_histogram(team_pred_probs, team_targets, n_agents=5)
        
        return {
            'exact_accuracy': float(exact_accuracy),
            'l1_count_error': float(l1_error),
            'kl_divergence': float(kl_div),
            'num_samples': len(team_predictions)
        }
    
    def _calculate_team_density_metrics(self, team_predictions, team_targets):
        """Calculate density metrics for a specific team."""
        mse = np.mean((team_predictions - team_targets) ** 2)
        mae = np.mean(np.abs(team_predictions - team_targets))
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'num_samples': len(team_predictions)
        }
