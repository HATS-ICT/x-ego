"""
Metrics Calculation for Multi-Agent Location Prediction

This module provides metrics calculation for multi-label classification:
- Hamming loss, subset accuracy, micro/macro F1
"""

import numpy as np
import torch
from sklearn.metrics import hamming_loss, accuracy_score, f1_score


class MetricsCalculator:
    """Handles metrics calculation for multi-label classification."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        pass
    
    def calculate_metrics(self, predictions, targets):
        """
        Calculate comprehensive test metrics for multi-label classification.
        
        Args:
            predictions: numpy array of predictions (logits)
            targets: numpy array of targets (multi-hot)
            
        Returns:
            Dictionary of calculated metrics
        """
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
        
        # Per-label metrics
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
    
    def calculate_team_metrics(self, predictions, targets, pov_team_sides, pred_tensors, target_tensors):
        """
        Calculate metrics separately for each team.
        
        Args:
            predictions: numpy array of predictions
            targets: numpy array of targets
            pov_team_sides: numpy array of team labels
            pred_tensors: List of prediction tensors (unused, kept for API compatibility)
            target_tensors: List of target tensors (unused, kept for API compatibility)
            
        Returns:
            Dictionary with team-specific metrics
        """
        team_metrics = {}
        
        for team in ['ct', 't']:
            team_mask = pov_team_sides == team
            if not np.any(team_mask):
                continue
            
            team_predictions = predictions[team_mask]
            team_targets = targets[team_mask]
            
            if len(team_predictions) == 0:
                continue
            
            team_metrics[team] = self._calculate_team_classification_metrics(
                team_predictions, team_targets
            )
        
        return team_metrics
    
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
