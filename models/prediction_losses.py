"""
Loss Functions for Multi-Agent Location Prediction

This module provides loss computation functions for different task formulations:
- Coordinate regression (MSE, Sinkhorn, Hausdorff, Energy)
- VAE reconstruction + KL divergence
- Binary classification (BCE, Focal Loss)
- Count regression and density estimation (MSE)
"""

import torch
import torch.nn.functional as F
from geomloss import SamplesLoss
import json
import os
from pathlib import Path


class LossComputer:
    """
    Handles loss computation for different task formulations.
    
    Task-specific loss functions:
    - coord-reg, coord-gen: 'mse', 'sinkhorn', 'hausdorff', 'energy'
    - multi-label-cls, grid-cls: 'bce', 'focal'
    - multi-output-reg, density-cls: 'mse', 'mae', 'kl'
    """
    
    def __init__(self, task_form, loss_fn, cfg):
        """
        Initialize loss computer.
        
        Args:
            task_form: Type of task ('coord-reg', 'coord-gen', 'multi-label-cls', etc.)
            loss_fn: Task-specific loss function (see class docstring for options)
            cfg: Root configuration object
        """
        self.task_form = task_form
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.model_cfg = cfg.model
        self.geometric_loss = None
        self.class_weights = None
        self.class_weights_method = None
        
        self._init_loss_functions()
        self._init_class_weights()
    
    def _init_loss_functions(self):
        """Initialize and validate task-specific loss functions."""
        # Define valid loss functions for each task form
        valid_loss_fns = {
            'coord-reg': ['mse', 'sinkhorn', 'hausdorff', 'energy'],
            'coord-gen': ['mse', 'sinkhorn', 'hausdorff', 'energy'],
            'multi-label-cls': ['bce', 'focal'],
            'grid-cls': ['bce', 'focal'],
            'multi-output-reg': ['mse', 'mae', 'kl'],
            'density-cls': ['mse', 'mae', 'kl']
        }
        
        # Validate loss function for current task form
        if self.task_form in valid_loss_fns:
            allowed_fns = valid_loss_fns[self.task_form]
            if self.loss_fn not in allowed_fns:
                raise ValueError(
                    f"Invalid loss_fn '{self.loss_fn}' for task_form '{self.task_form}'. "
                    f"Must be one of {allowed_fns}"
                )
        
        # Initialize geometric loss functions for coordinate-based tasks
        if self.task_form in ['coord-reg', 'coord-gen']:
            if self.loss_fn == 'sinkhorn':
                # Store sinkhorn parameters as attributes for later reference
                self.sinkhorn_blur = self.model_cfg.sinkhorn.blur
                self.sinkhorn_scaling = self.model_cfg.sinkhorn.scaling
                self.geometric_loss = SamplesLoss(
                    loss="sinkhorn", 
                    p=self.model_cfg.sinkhorn.p, 
                    blur=self.sinkhorn_blur, 
                    scaling=self.sinkhorn_scaling
                )
            elif self.loss_fn == 'hausdorff':
                self.geometric_loss = SamplesLoss(loss="hausdorff", p=self.model_cfg.hausdorff.p)
            elif self.loss_fn == 'energy':
                self.geometric_loss = SamplesLoss(loss="energy", p=self.model_cfg.energy.p)
            
            if self.loss_fn != 'mse':
                print(f"Initialized {self.loss_fn} loss for {self.task_form} mode")
    
    def _init_class_weights(self):
        """
        Initialize class weights for binary classification tasks.
        Only applies to multi-label-cls and grid-cls tasks with BCE loss.
        
        Config options for class_weights:
        - None/null: No class weighting (uniform weights)
        - "inverse": Inverse frequency weighting
        - "inverse_sqrt": Square root of inverse frequency
        - "effective_num": Effective number of samples method
        - "pos_weight": BCEWithLogitsLoss pos_weight format
        """
        # Only load class weights for classification tasks with BCE loss
        if self.task_form not in ['multi-label-cls', 'grid-cls'] or self.loss_fn != 'bce':
            return
        
        # Check if class_weights is specified in config
        if not hasattr(self.model_cfg, 'class_weights') or self.model_cfg.class_weights is None:
            return
        
        class_weights_method = self.model_cfg.class_weights
        
        # Handle string values like "none" or "null" (case-insensitive)
        if isinstance(class_weights_method, str) and class_weights_method.lower() in ['none', 'null']:
            return
        
        # Load class weights from JSON file
        data_base_path = Path(os.getenv('DATA_BASE_PATH'))
        weights_path = data_base_path / "class_weights" / f"{class_weights_method}.json"
        
        with open(weights_path, 'r') as f:
            data = json.load(f)
            weights_dict = data['weights']
        
        # Get place_names from root config and create weight tensor
        place_names = self.cfg.place_names
        weights_list = [weights_dict[place] for place in place_names]
        
        # Convert to tensor (will be moved to appropriate device during forward pass)
        self.class_weights = torch.tensor(weights_list, dtype=torch.float32)
        self.class_weights_method = class_weights_method
        
        print(f"Loaded class weights: {class_weights_method} ({len(weights_list)} classes)")
        print(f"  Weight range: [{self.class_weights.min():.4f}, {self.class_weights.max():.4f}]")
        if class_weights_method == 'pos_weight':
            print("  Using pos_weight parameter (weights positive samples only)")
        else:
            print("  Using manual weighting (weights all samples)")
    
    def compute_loss(self, outputs, targets):
        """
        Compute loss based on task form.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            outputs: Full model outputs (for VAE components)
            kl_weight: Weight for KL divergence (coord-gen mode only)
            
        Returns:
            loss: Computed loss value
            loss_components: Dictionary of loss components (for VAE)
        """
        predictions = outputs['predictions']
        if self.task_form == 'coord-reg':
            return self._compute_regression_loss(predictions, targets), {}
        
        elif self.task_form == 'coord-gen':
            if outputs is None or 'mu' not in outputs or 'logvar' not in outputs:
                raise ValueError("outputs with 'mu' and 'logvar' required for coord-gen mode")
            if self.model_cfg.vae.kl_weight is None:
                raise ValueError("kl_weight required for coord-gen mode")
            
            mu, logvar = outputs['mu'], outputs['logvar']
            total_loss, recon_loss, kl_loss = self._compute_vae_loss(
                predictions, targets, mu, logvar, self.model_cfg.vae.kl_weight
            )
            return total_loss, {
                'recon_loss': recon_loss,
                'kl_loss': kl_loss
            }
        
        elif self.task_form in ['multi-label-cls', 'grid-cls']:
            # Multi-label/grid classification losses
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
        
        elif self.task_form in ['multi-output-reg', 'density-cls']:
            # Regression and density losses
            if self.loss_fn == 'mse':
                loss = F.mse_loss(predictions, targets)
            elif self.loss_fn == 'mae':
                loss = F.l1_loss(predictions, targets)
            elif self.loss_fn == 'kl':
                # KL divergence for density distributions
                # TODO: Implement KL divergence loss
                raise NotImplementedError("KL divergence loss not yet implemented")
            return loss, {}
        
        else:
            raise ValueError(f"Unknown task_form: {self.task_form}")
    
    def _compute_regression_loss(self, predictions, targets):
        """Compute regression loss for coordinate-based tasks."""
        if self.loss_fn == 'mse':
            return F.mse_loss(predictions, targets)
        
        elif self.loss_fn in ['sinkhorn', 'hausdorff', 'energy']:
            # Geometric loss functions
            batch_size = predictions.shape[0]
            predictions = predictions.float()
            targets = targets.float()
            
            total_loss = 0.0
            for i in range(batch_size):
                pred_i = predictions[i]  # [5, 3]
                target_i = targets[i]    # [5, 3]
                loss_i = self.geometric_loss(pred_i, target_i)
                total_loss += loss_i
            
            return total_loss / batch_size
        
        else:
            raise ValueError(f"Unknown loss function: {self.loss_fn}")
    
    def _compute_vae_loss(self, predictions, targets, mu, logvar, kl_weight):
        """
        Compute VAE loss: reconstruction loss + KL divergence.
        
        Args:
            predictions: [B, 5, 3] - reconstructed targets
            targets: [B, 5, 3] - ground truth targets
            mu: [B, latent_dim] - latent mean
            logvar: [B, latent_dim] - latent log variance
            kl_weight: float - weight for KL divergence term
            
        Returns:
            total_loss: VAE loss
            recon_loss: Reconstruction loss component
            kl_loss: KL divergence component
        """
        # Reconstruction loss
        recon_loss = self._compute_regression_loss(predictions, targets)
        
        # KL divergence: D_KL(q(z|x) || p(z)) where p(z) = N(0, I)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(kl_loss)
        
        # Total VAE loss
        total_loss = recon_loss + kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
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
