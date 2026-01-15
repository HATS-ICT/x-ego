"""
Loss Functions for Multi-Agent Location Prediction

This module provides loss computation for multi-label classification (BCE, Focal Loss).
"""

import torch
import torch.nn.functional as F
import json
import os
from pathlib import Path


class LossComputer:
    """
    Handles loss computation for multi-label classification.
    
    Supported loss functions: 'bce', 'focal'
    """
    
    def __init__(self, loss_fn, cfg):
        """
        Initialize loss computer.
        
        Args:
            loss_fn: Loss function ('bce' or 'focal')
            cfg: Root configuration object
        """
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.model_cfg = cfg.model
        self.class_weights = None
        self.class_weights_method = None
        
        self._validate_loss_fn()
        self._init_class_weights()
    
    def _validate_loss_fn(self):
        """Validate loss function."""
        valid_loss_fns = ['bce', 'focal']
        if self.loss_fn not in valid_loss_fns:
            raise ValueError(
                f"Invalid loss_fn '{self.loss_fn}'. "
                f"Must be one of {valid_loss_fns}"
            )
    
    def _init_class_weights(self):
        """
        Initialize class weights for binary classification.
        
        Config options for class_weights:
        - None/null: No class weighting (uniform weights)
        - "inverse": Inverse frequency weighting
        - "inverse_sqrt": Square root of inverse frequency
        - "effective_num": Effective number of samples method
        - "pos_weight": BCEWithLogitsLoss pos_weight format
        
        Class weights are loaded from: data/class_weights/{task_name}/{method}.json
        where task_name is from cfg.meta.task (e.g., enemy_location_nowcast)
        """
        # Only load class weights for BCE loss
        if self.loss_fn != 'bce':
            return
        
        # Check if class_weights is specified in config
        if not hasattr(self.model_cfg, 'class_weights') or self.model_cfg.class_weights is None:
            return
        
        class_weights_method = self.model_cfg.class_weights
        
        # Handle string values like "none" or "null" (case-insensitive)
        if isinstance(class_weights_method, str) and class_weights_method.lower() in ['none', 'null']:
            return
        
        # Load class weights from JSON file
        # Path format: data/class_weights/{task_name}/{method}.json
        data_base_path = Path(os.getenv('DATA_BASE_PATH'))
        task_name = self.cfg.meta.task
        weights_path = data_base_path / "class_weights" / task_name / f"{class_weights_method}.json"
        
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Class weights file not found: {weights_path}\n"
                f"Run 'python scripts/inspect_labels/compute_class_weights.py --tasks {task_name}' to generate it."
            )
        
        with open(weights_path, 'r') as f:
            data = json.load(f)
            weights_dict = data['weights']
        
        # Get place_names from root config and create weight tensor
        place_names = self.cfg.place_names
        weights_list = [weights_dict[place] for place in place_names]
        
        # Convert to tensor (will be moved to appropriate device during forward pass)
        self.class_weights = torch.tensor(weights_list, dtype=torch.float32)
        self.class_weights_method = class_weights_method
        
        print(f"Loaded class weights for '{task_name}': {class_weights_method} ({len(weights_list)} classes)")
        print(f"  Weight range: [{self.class_weights.min():.4f}, {self.class_weights.max():.4f}]")
        if class_weights_method == 'pos_weight':
            print("  Using pos_weight parameter (weights positive samples only)")
        else:
            print("  Using manual weighting (weights all samples)")
    
    def compute_loss(self, outputs, targets):
        """
        Compute multi-label classification loss.
        
        Args:
            outputs: Model outputs containing 'predictions'
            targets: Ground truth targets (multi-hot encoded)
            
        Returns:
            loss: Computed loss value
            loss_components: Empty dict (for API compatibility)
        """
        predictions = outputs['predictions']
        
        if self.loss_fn == 'bce':
            if self.class_weights is not None:
                # Move weights to same device as predictions
                weights = self.class_weights.to(predictions.device)
                
                # Check if using pos_weight method (specifically for BCEWithLogitsLoss)
                if hasattr(self, 'class_weights_method') and self.class_weights_method == 'pos_weight':
                    # Use pos_weight parameter (weights positive class only)
                    loss = F.binary_cross_entropy_with_logits(
                        predictions, targets, pos_weight=weights
                    )
                else:
                    # Use manual weighting for other methods (inverse, inverse_sqrt, effective_num)
                    loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
                    # Apply weights: [B, N] * [N] -> [B, N]
                    weighted_loss = loss * weights.unsqueeze(0)
                    loss = weighted_loss.mean()
            else:
                loss = F.binary_cross_entropy_with_logits(predictions, targets)
        elif self.loss_fn == 'focal':
            loss = self._compute_focal_loss(predictions, targets, self.model_cfg.focal.alpha, self.model_cfg.focal.gamma)
        
        return loss, {}
    
    def _compute_focal_loss(self, predictions, targets, alpha, gamma):
        """
        Compute Focal Loss for imbalanced binary classification.
        
        Focal Loss: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        
        This loss down-weights easy examples and focuses on hard negatives,
        making it particularly effective for highly imbalanced datasets.
        
        Args:
            predictions: [B, N] - raw logits (before sigmoid)
            targets: [B, N] - binary targets (0 or 1)
            alpha: float - weighting factor for positive class
            gamma: float - focusing parameter
                          Higher gamma puts more focus on hard examples
            
        Returns:
            loss: Scalar focal loss value
            
        Reference:
            Lin et al. "Focal Loss for Dense Object Detection" (2017)
            https://arxiv.org/abs/1708.02002
        """
        # Get probabilities from logits
        probs = torch.sigmoid(predictions)
        
        # Compute binary cross entropy without reduction
        ce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        
        # Compute p_t: probability of the correct class
        # p_t = p if target = 1, else (1 - p)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        p_t = torch.clamp(p_t, 1e-8, 1 - 1e-8)  # avoid log(0)
        
        # Compute focal weight: (1 - p_t)^gamma
        # This down-weights easy examples (high p_t) and focuses on hard examples (low p_t)
        focal_weight = (1 - p_t) ** gamma
        
        # Apply alpha weighting for class balance
        # α_t = α if target = 1, else (1 - α)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        
        # Compute focal loss
        focal_loss = alpha_t * focal_weight * ce_loss
        
        return focal_loss.mean()
