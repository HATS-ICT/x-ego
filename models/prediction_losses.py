"""
Loss Functions for Multi-Agent Location Prediction

This module provides loss computation functions for different task formulations:
- Coordinate regression (MSE, Sinkhorn, Hausdorff, Energy)
- VAE reconstruction + KL divergence
- Binary classification (BCE)
- Count regression and density estimation (MSE)
"""

import torch
import torch.nn.functional as F
from geomloss import SamplesLoss


class LossComputer:
    """Handles loss computation for different task formulations."""
    
    def __init__(self, task_form, loss_fn='mse', sinkhorn_blur=0.05, sinkhorn_scaling=0.9):
        """
        Initialize loss computer.
        
        Args:
            task_form: Type of task ('coord-reg', 'generative', 'multi-label-cls', etc.)
            loss_fn: Loss function for coordinate tasks ('mse', 'sinkhorn', 'hausdorff', 'energy')
            sinkhorn_blur: Blur parameter for Sinkhorn loss
            sinkhorn_scaling: Scaling parameter for Sinkhorn loss
        """
        self.task_form = task_form
        self.loss_fn = loss_fn
        self.sinkhorn_blur = sinkhorn_blur
        self.sinkhorn_scaling = sinkhorn_scaling
        self.geometric_loss = None
        
        self._init_loss_functions()
    
    def _init_loss_functions(self):
        """Initialize geometric loss functions if needed."""
        if self.task_form in ['coord-reg', 'generative']:
            valid_loss_fns = ['mse', 'sinkhorn', 'hausdorff', 'energy']
            if self.loss_fn not in valid_loss_fns:
                raise ValueError(f"Invalid loss_fn '{self.loss_fn}'. Must be one of {valid_loss_fns}")
            
            if self.loss_fn == 'sinkhorn':
                self.geometric_loss = SamplesLoss(
                    loss="sinkhorn", 
                    p=2, 
                    blur=self.sinkhorn_blur, 
                    scaling=self.sinkhorn_scaling
                )
            elif self.loss_fn == 'hausdorff':
                self.geometric_loss = SamplesLoss(loss="hausdorff", p=2)
            elif self.loss_fn == 'energy':
                self.geometric_loss = SamplesLoss(loss="energy", p=2)
            
            if self.loss_fn != 'mse':
                print(f"Initialized {self.loss_fn} loss for {self.task_form} mode")
    
    def compute_loss(self, predictions, targets, outputs=None, kl_weight=None):
        """
        Compute loss based on task form.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            outputs: Full model outputs (for VAE components)
            kl_weight: Weight for KL divergence (generative mode only)
            
        Returns:
            loss: Computed loss value
            loss_components: Dictionary of loss components (for VAE)
        """
        if self.task_form == 'coord-reg':
            return self._compute_regression_loss(predictions, targets), {}
        
        elif self.task_form == 'generative':
            if outputs is None or 'mu' not in outputs or 'logvar' not in outputs:
                raise ValueError("outputs with 'mu' and 'logvar' required for generative mode")
            if kl_weight is None:
                raise ValueError("kl_weight required for generative mode")
            
            mu, logvar = outputs['mu'], outputs['logvar']
            total_loss, recon_loss, kl_loss = self._compute_vae_loss(
                predictions, targets, mu, logvar, kl_weight
            )
            return total_loss, {
                'recon_loss': recon_loss,
                'kl_loss': kl_loss
            }
        
        elif self.task_form in ['multi-label-cls', 'grid-cls']:
            # Binary cross-entropy for multi-label/grid classification
            return F.binary_cross_entropy_with_logits(predictions, targets), {}
        
        elif self.task_form in ['multi-output-reg', 'density-cls']:
            # MSE loss for count regression and density distribution
            return F.mse_loss(predictions, targets), {}
        
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
