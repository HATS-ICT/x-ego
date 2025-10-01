"""
Multi-Agent Enemy Location Prediction Model

This module implements a PyTorch Lightning model for predicting enemy locations
in multi-agent scenarios, supporting six different task formulations:

Task Forms:
- coord-reg: Direct regression of (x, y, z) coordinates for each agent
- generative: VAE-based generative model for sampling agent coordinates
- multi-label-cls: Binary classification over predefined locations
- multi-output-reg: Regression of agent counts per location
- grid-cls: Binary classification over spatial grid cells
- density-cls: Smoothed density distribution over grid cells
"""

import torch
import torch.nn as nn
import lightning as L
from torch.optim import AdamW
import torch._dynamo
import numpy as np
import json
from pathlib import Path
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from torchmetrics.classification import (
    MultilabelHammingDistance, 
    MultilabelExactMatch,
    MultilabelF1Score
)

from utils.plot_utils import create_prediction_plots, create_prediction_heatmaps_grid
from utils.serialization_utils import json_serializable

# Import refactored components
from models.prediction_losses import LossComputer
from models.prediction_metrics import MetricsCalculator
from models.coordinate_utils import CoordinateScalerMixin
from models.vae_mixins import VAEMixin

try:
    from video_encoder_baseline import VideoEncoderBaseline
except ImportError:
    from .video_encoder_baseline import VideoEncoderBaseline


class MultiAgentEnemyLocationPredictionModel(L.LightningModule, CoordinateScalerMixin, VAEMixin):
    """
    Multi-agent enemy location prediction model supporting multiple task formulations.
    
    The model encodes multi-agent video inputs and team information to predict
    enemy locations using various output representations based on the task form.
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        # Core model components
        self.video_encoder = self.init_video_encoder(cfg.model.encoder.video)
        self.num_agents = cfg.data.num_agents
        self.task_form = cfg.data.task_form
        
        # Team encoder: embed team side (T=0, CT=1) into a vector
        self.team_embed_dim = cfg.model.team_embed_dim
        self.team_encoder = nn.Embedding(2, self.team_embed_dim)  # 2 teams: T, CT
        
        # Agent fusion configuration
        self.agent_fusion_method = cfg.model.agent_fusion_method
        self._init_agent_fusion()
        
        # Predictor head configuration
        hidden_dim = cfg.model.hidden_dim
        dropout = cfg.model.dropout
        num_hidden_layers = cfg.model.num_hidden_layers
        
        # Combined dimension: video + team embeddings
        self.combined_dim = self.video_encoder.embed_dim + self.team_embed_dim
        
        # Initialize task-specific components
        self._init_task_specific_components(cfg, hidden_dim, dropout, num_hidden_layers)
        
        # Initialize loss and metrics calculators
        # Extract task-specific loss function
        loss_fn = cfg.data.loss_fn[self.task_form]
        sinkhorn_blur = cfg.data.sinkhorn_blur
        sinkhorn_scaling = cfg.data.sinkhorn_scaling
        self.loss_computer = LossComputer(self.task_form, loss_fn, sinkhorn_blur, sinkhorn_scaling)
        self.metrics_calculator = MetricsCalculator(self.task_form)
        
        # Coordinate scaler for coordinate-based tasks
        self.coordinate_scaler = None
        
        # Test tracking
        self.test_predictions = []
        self.test_targets = []
        self.output_dir = cfg.path.exp
    
    def init_video_encoder(self, video_encoder_cfg):
        """Initialize the video encoder."""
        return VideoEncoderBaseline(video_encoder_cfg)
    
    def _get_target_locations(self, batch):
        """
        Get target locations from batch, supporting both enemy and teammate tasks.
        
        Args:
            batch: Input batch dictionary
            
        Returns:
            Target locations tensor
        """
        if 'enemy_locations' in batch:
            return batch['enemy_locations']
        elif 'future_locations' in batch:
            return batch['future_locations']
        else:
            raise KeyError("Batch must contain either 'enemy_locations' or 'future_locations'")
    
    def _init_agent_fusion(self):
        """Initialize agent fusion mechanism."""
        if self.agent_fusion_method == 'attention':
            self.agent_attention = nn.MultiheadAttention(
                embed_dim=self.video_encoder.embed_dim,
                num_heads=8,
                batch_first=True
            )
        elif self.agent_fusion_method == 'concat':
            self.agent_fusion_proj = nn.Linear(
                self.video_encoder.embed_dim * self.num_agents, 
                self.video_encoder.embed_dim
            )
    
    def _init_task_specific_components(self, cfg, hidden_dim, dropout, num_hidden_layers):
        """Initialize task-specific output heads and metrics."""
        if self.task_form in ['coord-reg', 'generative']:
            # Coordinate regression: output [num_agents * 3 coordinates]
            self.output_dim = cfg.model.num_target_agents * 3
            self.train_mse = MeanSquaredError()
            self.val_mse = MeanSquaredError()
            self.train_mae = MeanAbsoluteError()
            self.val_mae = MeanAbsoluteError()
            
            if self.task_form == 'generative':
                # VAE-specific components
                self.latent_dim = cfg.model.vae.latent_dim
                encoder_input_dim = self.output_dim + self.combined_dim
                self.vae_encoder = self._build_mlp(
                    encoder_input_dim, self.latent_dim * 2, 
                    num_hidden_layers, hidden_dim, dropout
                )
                decoder_input_dim = self.latent_dim + self.combined_dim
                self.predictor = self._build_mlp(
                    decoder_input_dim, self.output_dim, 
                    num_hidden_layers, hidden_dim, dropout
                )
            else:
                # Standard coordinate regression
                self.predictor = self._build_mlp(
                    self.combined_dim, self.output_dim, 
                    num_hidden_layers, hidden_dim, dropout
                )
        
        elif self.task_form in ['multi-label-cls', 'multi-output-reg']:
            # Place-based tasks: output [num_places]
            self.output_dim = cfg.num_places
            self.predictor = self._build_mlp(
                self.combined_dim, self.output_dim, 
                num_hidden_layers, hidden_dim, dropout
            )
            
            # Initialize multi-label classification metrics
            if self.task_form == 'multi-label-cls':
                num_labels = cfg.num_places
                self.train_hamming = MultilabelHammingDistance(num_labels=num_labels)
                self.val_hamming = MultilabelHammingDistance(num_labels=num_labels)
                self.train_exact_match = MultilabelExactMatch(num_labels=num_labels)
                self.val_exact_match = MultilabelExactMatch(num_labels=num_labels)
                self.train_micro_f1 = MultilabelF1Score(num_labels=num_labels, average='micro')
                self.val_micro_f1 = MultilabelF1Score(num_labels=num_labels, average='micro')
                self.train_macro_f1 = MultilabelF1Score(num_labels=num_labels, average='macro')
                self.val_macro_f1 = MultilabelF1Score(num_labels=num_labels, average='macro')
        
        elif self.task_form in ['grid-cls', 'density-cls']:
            # Grid-based tasks: output [grid_resolution^2]
            grid_resolution = cfg.data.grid_resolution
            self.output_dim = grid_resolution * grid_resolution
            self.predictor = self._build_mlp(
                self.combined_dim, self.output_dim, 
                num_hidden_layers, hidden_dim, dropout
            )
            
            # Initialize multi-label classification metrics for grid-cls
            if self.task_form == 'grid-cls':
                num_labels = self.output_dim
                self.train_hamming = MultilabelHammingDistance(num_labels=num_labels)
                self.val_hamming = MultilabelHammingDistance(num_labels=num_labels)
                self.train_exact_match = MultilabelExactMatch(num_labels=num_labels)
                self.val_exact_match = MultilabelExactMatch(num_labels=num_labels)
                self.train_micro_f1 = MultilabelF1Score(num_labels=num_labels, average='micro')
                self.val_micro_f1 = MultilabelF1Score(num_labels=num_labels, average='micro')
                self.train_macro_f1 = MultilabelF1Score(num_labels=num_labels, average='macro')
                self.val_macro_f1 = MultilabelF1Score(num_labels=num_labels, average='macro')
    
    def _build_mlp(self, input_dim, output_dim, num_hidden_layers, hidden_dim, dropout):
        """Build a multi-layer perceptron with specified number of hidden layers."""
        layers = []
        current_dim = input_dim
        
        for i in range(num_hidden_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)
    
    # ============================================================================
    # Multi-Agent Video Processing
    # ============================================================================
    
    def process_multi_agent_video(self, video):
        """
        Process multi-agent video input and fuse agent embeddings.
        
        Args:
            video: [B, A, T, C, H, W] - multi-agent video tensor
            
        Returns:
            fused_embeddings: [B, embed_dim] - fused video features
            agent_embeddings: [B, A, embed_dim] - per-agent embeddings
        """
        B, A, T, C, H, W = video.shape
        
        # Reshape for batch processing
        video_reshaped = video.view(B * A, T, C, H, W)
        
        # Handle tubelet attention pooling
        if self.agent_fusion_method == 'tublet_attentive_pooler':
            agent_embeddings = self.video_encoder(
                video_reshaped, return_last_hidden_state=True
            )  # [B*A, seq_len, embed_dim]
            num_tublets = agent_embeddings.shape[1]
            cross_agent_embeddings = agent_embeddings.view(B, A * num_tublets, -1)
            fused_embeddings = self.video_encoder.video_encoder.pooler(cross_agent_embeddings)
            return fused_embeddings, agent_embeddings
        
        # Standard agent fusion methods
        agent_embeddings = self.video_encoder(video_reshaped)  # [B*A, embed_dim]
        agent_embeddings = agent_embeddings.view(B, A, -1)  # [B, A, embed_dim]
        
        if self.agent_fusion_method == 'mean':
            fused_embeddings = torch.mean(agent_embeddings, dim=1)
        elif self.agent_fusion_method == 'max':
            fused_embeddings, _ = torch.max(agent_embeddings, dim=1)
        elif self.agent_fusion_method == 'attention':
            fused_embeddings, _ = self.agent_attention(
                agent_embeddings, agent_embeddings, agent_embeddings
            )
            fused_embeddings = torch.mean(fused_embeddings, dim=1)
        elif self.agent_fusion_method == 'concat':
            fused_embeddings = agent_embeddings.view(B, -1)  # [B, A*embed_dim]
            fused_embeddings = self.agent_fusion_proj(fused_embeddings)
        else:
            raise ValueError(f"Unknown agent fusion method: {self.agent_fusion_method}")
        
        return fused_embeddings, agent_embeddings
    
    # ============================================================================
    # Forward Pass
    # ============================================================================
    
    def forward(self, batch, mode='full'):
        """
        Forward pass through the model.
        
        Args:
            batch: Input batch containing:
                - video: [B, A, T, C, H, W] - multi-agent video
                - team_side_encoded: [B] - team encoding (0=T, 1=CT)
                - enemy_locations or future_locations: [B, 5, 3] - targets (for training/full mode)
            mode: 'full' for full VAE forward pass, 'sampling' for sampling from prior
            
        Returns:
            Dictionary containing:
                - predictions: Model predictions
                - fused_embeddings: Fused video features
                - team_embeddings: Team embeddings
                - combined_features: Combined video + team features
                - agent_embeddings: Per-agent embeddings
                - mu, logvar, z: VAE latent variables (generative mode only)
        """
        video = batch['video']  # [B, A, T, C, H, W]
        team_side_encoded = batch['team_side_encoded']  # [B]
        
        if len(video.shape) != 6:
            raise ValueError(f"Expected video shape [B, A, T, C, H, W], got {video.shape}")
        
        # Encode video and team information
        fused_embeddings, agent_embeddings = self.process_multi_agent_video(video)
        team_embeddings = self.team_encoder(team_side_encoded)  # [B, team_embed_dim]
        combined_features = torch.cat([fused_embeddings, team_embeddings], dim=1)
        
        # Task-specific forward pass
        if self.task_form == 'generative':
            return self._forward_generative(batch, combined_features, 
                                           fused_embeddings, team_embeddings, 
                                           agent_embeddings, mode)
        else:
            return self._forward_standard(combined_features, fused_embeddings, 
                                         team_embeddings, agent_embeddings)
    
    def _forward_generative(self, batch, combined_features, fused_embeddings, 
                           team_embeddings, agent_embeddings, mode):
        """Forward pass for generative (VAE) mode."""
        if mode == 'sampling':
            # Sample from prior
            predictions = self.sample_from_prior(combined_features, num_samples=1)
            predictions = predictions.squeeze(1)  # [B, 5, 3]
            
            return {
                'predictions': predictions,
                'fused_embeddings': fused_embeddings,
                'team_embeddings': team_embeddings,
                'combined_features': combined_features,
                'agent_embeddings': agent_embeddings,
                'mu': None,
                'logvar': None,
                'z': None
            }
        else:
            # Full VAE: encode, reparameterize, decode
            target_locations = self._get_target_locations(batch)
            mu, logvar = self.encode(target_locations, combined_features)
            z = self.reparameterize(mu, logvar)
            predictions = self.decode(z, combined_features)
            
            return {
                'predictions': predictions,
                'fused_embeddings': fused_embeddings,
                'team_embeddings': team_embeddings,
                'combined_features': combined_features,
                'agent_embeddings': agent_embeddings,
                'mu': mu,
                'logvar': logvar,
                'z': z
            }
    
    def _forward_standard(self, combined_features, fused_embeddings, 
                         team_embeddings, agent_embeddings):
        """Forward pass for standard (non-generative) modes."""
        predictions = self.predictor(combined_features)
        
        # Reshape for coordinate regression if needed
        if self.task_form == 'coord-reg':
            predictions = predictions.view(-1, 5, 3)
        
        return {
            'predictions': predictions,
            'fused_embeddings': fused_embeddings,
            'team_embeddings': team_embeddings,
            'combined_features': combined_features,
            'agent_embeddings': agent_embeddings
        }
    
    # ============================================================================
    # Training, Validation, and Testing
    # ============================================================================
    
    @torch._dynamo.disable
    def safe_log(self, *args, **kwargs):
        """Safe logging that disables dynamo compilation."""
        return self.log(*args, **kwargs)
    
    @torch._dynamo.disable
    def _tensor_only_batch(self, batch):
        """Extract only tensor elements from batch."""
        return {k: v for k, v in batch.items() if torch.is_tensor(v)}
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        batch_size = batch["video"].shape[0]
        targets = self._get_target_locations(batch)
        
        # Forward pass
        outputs = self.forward(batch, mode='full')
        predictions = outputs['predictions']
        
        # Compute loss
        kl_weight = self.cfg.model.vae.kl_weight if self.task_form == 'generative' else None
        loss, loss_components = self.loss_computer.compute_loss(
            predictions, targets, outputs, kl_weight
        )
        
        # Log main loss
        self.safe_log('train/loss', loss, batch_size=batch_size, 
                     on_step=True, on_epoch=True, prog_bar=True)
        
        # Log VAE components if applicable
        if loss_components:
            for name, value in loss_components.items():
                self.safe_log(f'train/{name}', value, batch_size=batch_size,
                            on_step=True, on_epoch=True, prog_bar=False)
        
        # Log coordinate-based metrics
        if self.task_form in ['coord-reg', 'generative']:
            self.train_mse(predictions.view(batch_size, -1), targets.view(batch_size, -1))
            self.train_mae(predictions.view(batch_size, -1), targets.view(batch_size, -1))
            self.safe_log('train/mse', self.train_mse, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=True)
            self.safe_log('train/mae', self.train_mae, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=True)
        
        # Log multi-label classification metrics
        elif self.task_form in ['multi-label-cls', 'grid-cls']:
            # Convert logits to probabilities and then to binary predictions
            pred_probs = torch.sigmoid(predictions)
            targets_int = targets.int()
            
            # Update metrics
            self.train_hamming(pred_probs, targets_int)
            self.train_exact_match(pred_probs, targets_int)
            self.train_micro_f1(pred_probs, targets_int)
            self.train_macro_f1(pred_probs, targets_int)
            
            # Log metrics
            self.safe_log('train/hamming_loss', self.train_hamming, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=True)
            self.safe_log('train/subset_accuracy', self.train_exact_match, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=True)
            self.safe_log('train/micro_f1', self.train_micro_f1, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=True)
            self.safe_log('train/macro_f1', self.train_macro_f1, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        batch = self._tensor_only_batch(batch)
        batch_size = batch["video"].shape[0]
        targets = self._get_target_locations(batch)
        
        # Forward pass
        outputs = self.forward(batch, mode='full')
        predictions = outputs['predictions']
        
        # Compute loss
        kl_weight = self.cfg.model.vae.kl_weight if self.task_form == 'generative' else None
        loss, loss_components = self.loss_computer.compute_loss(
            predictions, targets, outputs, kl_weight
        )
        
        # Log main loss
        self.safe_log('val/loss', loss, batch_size=batch_size,
                     on_step=False, on_epoch=True, prog_bar=True)
        
        # Log VAE components
        if loss_components:
            for name, value in loss_components.items():
                self.safe_log(f'val/{name}', value, batch_size=batch_size,
                            on_step=False, on_epoch=True, prog_bar=False)
        
        # Test generative sampling for VAE
        if self.task_form == 'generative':
            gen_outputs = self.forward(batch, mode='sampling')
            gen_predictions = gen_outputs['predictions']
            gen_loss = self.loss_computer._compute_regression_loss(gen_predictions, targets)
            self.safe_log('val/gen_loss', gen_loss, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=False)
        
        # Log coordinate-based metrics
        if self.task_form in ['coord-reg', 'generative']:
            self.val_mse(predictions.view(batch_size, -1), targets.view(batch_size, -1))
            self.val_mae(predictions.view(batch_size, -1), targets.view(batch_size, -1))
            self.safe_log('val/mse', self.val_mse, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=True)
            self.safe_log('val/mae', self.val_mae, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=True)
        
        # Log multi-label classification metrics
        elif self.task_form in ['multi-label-cls', 'grid-cls']:
            # Convert logits to probabilities and then to binary predictions
            pred_probs = torch.sigmoid(predictions)
            targets_int = targets.int()
            
            # Update metrics
            self.val_hamming(pred_probs, targets_int)
            self.val_exact_match(pred_probs, targets_int)
            self.val_micro_f1(pred_probs, targets_int)
            self.val_macro_f1(pred_probs, targets_int)
            
            # Log metrics
            self.safe_log('val/hamming_loss', self.val_hamming, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=True)
            self.safe_log('val/subset_accuracy', self.val_exact_match, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=True)
            self.safe_log('val/micro_f1', self.val_micro_f1, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=True)
            self.safe_log('val/macro_f1', self.val_macro_f1, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_test_start(self):
        """Initialize test tracking variables."""
        self.test_team_sides = []
        self.test_targets = []
        self.test_predictions = []
        self.test_targets_unscaled = []
        self.test_predictions_unscaled = []
        self.test_raw_samples_by_team = {'T': [], 'CT': []}
        
        # Initialize test metrics for multi-label classification
        if self.task_form in ['multi-label-cls', 'grid-cls']:
            num_labels = self.output_dim
            self.test_hamming = MultilabelHammingDistance(num_labels=num_labels).to(self.device)
            self.test_exact_match = MultilabelExactMatch(num_labels=num_labels).to(self.device)
            self.test_micro_f1 = MultilabelF1Score(num_labels=num_labels, average='micro').to(self.device)
            self.test_macro_f1 = MultilabelF1Score(num_labels=num_labels, average='macro').to(self.device)
    
    def test_step(self, batch, batch_idx):
        """Test step with sample collection for visualization."""
        # Store raw samples for visualization (5 per team)
        team_sides = batch['team_side']
        samples_per_team = 5
        
        for i in range(batch['video'].shape[0]):
            team_side = team_sides[i]
            if len(self.test_raw_samples_by_team[team_side]) < samples_per_team:
                # Get target locations key dynamically
                target_key = 'enemy_locations' if 'enemy_locations' in batch else 'future_locations'
                sample = {
                    'video': batch['video'][i:i+1].clone(),
                    'team_side_encoded': batch['team_side_encoded'][i:i+1].clone(),
                    target_key: batch[target_key][i:i+1].clone(),
                    'team_side': team_side
                }
                self.test_raw_samples_by_team[team_side].append(sample)
            
            # Break if we have enough samples
            if all(len(samples) >= samples_per_team 
                   for samples in self.test_raw_samples_by_team.values()):
                break
        
        # Process batch
        batch = self._tensor_only_batch(batch)
        batch_size = batch["video"].shape[0]
        targets = self._get_target_locations(batch)
        
        # Forward pass
        outputs = self.forward(batch, mode='full')
        predictions = outputs['predictions']
        
        # Compute loss
        kl_weight = self.cfg.model.vae.kl_weight if self.task_form == 'generative' else None
        loss, loss_components = self.loss_computer.compute_loss(
            predictions, targets, outputs, kl_weight
        )
        
        # Log main loss
        self.safe_log('test/loss', loss, batch_size=batch_size,
                     on_step=False, on_epoch=True, prog_bar=True)
        
        # Log VAE components
        if loss_components:
            for name, value in loss_components.items():
                self.safe_log(f'test/{name}', value, batch_size=batch_size,
                            on_step=False, on_epoch=True, prog_bar=False)
        
        # Test generative sampling for VAE
        if self.task_form == 'generative':
            gen_outputs = self.forward(batch, mode='sampling')
            gen_predictions = gen_outputs['predictions']
            gen_loss = self.loss_computer._compute_regression_loss(gen_predictions, targets)
            self.safe_log('test/gen_loss', gen_loss, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=False)
        
        # Initialize and log coordinate-based metrics
        if self.task_form in ['coord-reg', 'generative']:
            if not hasattr(self, 'test_mse'):
                self.test_mse = MeanSquaredError().to(self.device)
                self.test_mae = MeanAbsoluteError().to(self.device)
            
            self.test_mse(predictions.view(batch_size, -1), targets.view(batch_size, -1))
            self.test_mae(predictions.view(batch_size, -1), targets.view(batch_size, -1))
            self.safe_log('test/mse', self.test_mse, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=True)
            self.safe_log('test/mae', self.test_mae, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=True)
        
        # Log multi-label classification metrics
        elif self.task_form in ['multi-label-cls', 'grid-cls']:
            # Convert logits to probabilities and then to binary predictions
            pred_probs = torch.sigmoid(predictions)
            targets_int = targets.int()
            
            # Update metrics
            self.test_hamming(pred_probs, targets_int)
            self.test_exact_match(pred_probs, targets_int)
            self.test_micro_f1(pred_probs, targets_int)
            self.test_macro_f1(pred_probs, targets_int)
            
            # Log metrics
            self.safe_log('test/hamming_loss', self.test_hamming, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=True)
            self.safe_log('test/subset_accuracy', self.test_exact_match, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=True)
            self.safe_log('test/micro_f1', self.test_micro_f1, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=True)
            self.safe_log('test/macro_f1', self.test_macro_f1, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=True)
        
        # Store predictions and targets
        self.test_team_sides.extend(team_sides)
        
        if self.task_form in ['coord-reg', 'generative']:
            # Store both scaled and unscaled versions
            predictions_unscaled = self.unscale_coordinates(predictions)
            targets_unscaled = self.unscale_coordinates(targets)
            
            self.test_predictions.append(predictions.cpu().float())
            self.test_targets.append(targets.cpu().float())
            self.test_predictions_unscaled.extend(predictions_unscaled.cpu().float().numpy())
            self.test_targets_unscaled.extend(targets_unscaled.cpu().float().numpy())
        else:
            # Store predictions and targets for classification/grid tasks
            self.test_predictions.append(predictions.cpu().float())
            self.test_targets.append(targets.cpu().float())
        
        return loss
    
    def on_test_epoch_end(self):
        """Calculate metrics and create visualizations at end of test epoch."""
        # Concatenate all predictions and targets
        predictions = torch.cat(self.test_predictions, dim=0).cpu().numpy()
        targets = torch.cat(self.test_targets, dim=0).cpu().numpy()
        team_sides = np.array(self.test_team_sides)
        unique_teams, team_counts = np.unique(team_sides, return_counts=True)
        
        # Setup output directory
        output_dir = Path(self.output_dir)
        checkpoint_suffix = f"_{self.checkpoint_name}" if hasattr(self, 'checkpoint_name') else ""
        plots_dir = output_dir / "test_analysis" / f"enemy_location{checkpoint_suffix}"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving enemy location prediction test analysis to: {plots_dir}")
        
        # Calculate metrics using MetricsCalculator
        test_results = self.metrics_calculator.calculate_metrics(predictions, targets)
        
        # Add geometric distances for coordinate-based tasks
        if self.task_form in ['coord-reg', 'generative']:
            geometric_metrics = self.metrics_calculator.calculate_geometric_distances(
                self.test_predictions, self.test_targets
            )
            test_results['geometric_distances'] = geometric_metrics
        
        # Calculate team-specific metrics
        team_specific_metrics = self.metrics_calculator.calculate_team_metrics(
            predictions, targets, team_sides, self.test_predictions, self.test_targets
        )
        test_results['team_specific_metrics'] = team_specific_metrics
        
        # Log metrics
        self._log_team_metrics(team_specific_metrics)
        self._log_overall_metrics(test_results)
        
        # Add experiment metadata
        test_results['num_agents'] = self.cfg.data.num_agents
        test_results['agent_fusion_method'] = self.agent_fusion_method
        test_results['task_form'] = self.task_form
        test_results['team_distribution'] = dict(zip(unique_teams.tolist(), team_counts.tolist()))
        
        # Add task-specific metadata
        if self.task_form in ['coord-reg', 'generative']:
            self._add_coordinate_metadata(test_results)
        
        # Save results to JSON
        results_file = plots_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=json_serializable)
        print(f"Test results saved to: {results_file}")
        
        # Create visualization plots
        create_prediction_plots(self.task_form, predictions, targets, plots_dir, team_sides)
        
        # Create KDE heatmaps for coordinate-based tasks
        if self.task_form in ['coord-reg', 'generative']:
            self._create_prediction_heatmaps(plots_dir)
    
    def _log_team_metrics(self, team_specific_metrics):
        """Log team-specific metrics to logger."""
        for team in ['CT', 'T']:
            if team not in team_specific_metrics:
                continue
            
            metrics = team_specific_metrics[team]
            team_prefix = f'test/{team.lower()}'
            
            if self.task_form in ['coord-reg', 'generative']:
                self.safe_log(f'{team_prefix}_mse', metrics['mse'], on_epoch=True)
                self.safe_log(f'{team_prefix}_mae', metrics['mae'], on_epoch=True)
                
                if 'geometric_distances' in metrics:
                    geom = metrics['geometric_distances']
                    self.safe_log(f'{team_prefix}_chamfer_distance', 
                                 geom['chamfer_distance_mean'], on_epoch=True)
                    self.safe_log(f'{team_prefix}_wasserstein_distance',
                                 geom['wasserstein_distance_mean'], on_epoch=True)
            
            elif self.task_form in ['multi-label-cls', 'grid-cls']:
                # Multi-label classification metrics
                if 'hamming_loss' in metrics:
                    self.safe_log(f'{team_prefix}_hamming_loss', 
                                 metrics['hamming_loss'], on_epoch=True)
                if 'subset_accuracy' in metrics:
                    self.safe_log(f'{team_prefix}_subset_accuracy',
                                 metrics['subset_accuracy'], on_epoch=True)
                if 'micro_f1' in metrics:
                    self.safe_log(f'{team_prefix}_micro_f1',
                                 metrics['micro_f1'], on_epoch=True)
                if 'macro_f1' in metrics:
                    self.safe_log(f'{team_prefix}_macro_f1',
                                 metrics['macro_f1'], on_epoch=True)
            
            elif self.task_form in ['multi-output-reg', 'density-cls']:
                if 'exact_accuracy' in metrics:
                    self.safe_log(f'{team_prefix}_exact_accuracy', 
                                 metrics['exact_accuracy'], on_epoch=True)
                if 'l1_count_error' in metrics:
                    self.safe_log(f'{team_prefix}_l1_count_error',
                                 metrics['l1_count_error'], on_epoch=True)
                if 'kl_divergence' in metrics:
                    self.safe_log(f'{team_prefix}_kl_divergence',
                                 metrics['kl_divergence'], on_epoch=True)
    
    def _log_overall_metrics(self, test_results):
        """Log overall test metrics to logger."""
        if self.task_form in ['coord-reg', 'generative']:
            if 'geometric_distances' in test_results:
                geom = test_results['geometric_distances']
                self.safe_log('test/chamfer_distance', 
                             geom['chamfer_distance_mean'], on_epoch=True)
                self.safe_log('test/wasserstein_distance',
                             geom['wasserstein_distance_mean'], on_epoch=True)
        
        elif self.task_form in ['multi-label-cls', 'grid-cls']:
            # Multi-label classification metrics
            metric_names = ['hamming_loss', 'subset_accuracy', 'micro_f1', 'macro_f1']
            for metric_name in metric_names:
                if metric_name in test_results:
                    self.safe_log(f'test/{metric_name}', 
                                 test_results[metric_name], on_epoch=True)
        
        elif self.task_form in ['multi-output-reg', 'density-cls']:
            metric_names = ['exact_accuracy', 'l1_count_error', 'kl_divergence', 'multinomial_loss']
            for metric_name in metric_names:
                if metric_name in test_results:
                    self.safe_log(f'test/{metric_name}', 
                                 test_results[metric_name], on_epoch=True)
    
    def _add_coordinate_metadata(self, test_results):
        """Add coordinate-specific metadata to test results."""
        test_results['loss_function'] = self.loss_computer.loss_fn
        test_results['coordinate_scaling'] = self.coordinate_scaler is not None
        
        if self.coordinate_scaler is not None:
            test_results['scaler_data_min'] = self.coordinate_scaler.data_min_.tolist()
            test_results['scaler_data_max'] = self.coordinate_scaler.data_max_.tolist()
            test_results['scaler_scale'] = self.coordinate_scaler.scale_.tolist()
        
        if self.loss_computer.loss_fn == 'sinkhorn':
            test_results['sinkhorn_blur'] = self.loss_computer.sinkhorn_blur
            test_results['sinkhorn_scaling'] = self.loss_computer.sinkhorn_scaling
        
        if self.task_form == 'generative':
            test_results['latent_dim'] = self.latent_dim
            test_results['kl_weight'] = self.cfg.model.vae.kl_weight
    
    def _create_prediction_heatmaps(self, plots_dir):
        """Create KDE heatmaps for selected test samples."""
        # Combine samples from both teams
        combined_samples = []
        for team in ['T', 'CT']:
            combined_samples.extend(self.test_raw_samples_by_team[team])
        
        if len(combined_samples) == 0:
            return
        
        print(f"Creating KDE heatmaps for {len(combined_samples)} test samples...")
        print(f"  T samples: {len(self.test_raw_samples_by_team['T'])}")
        print(f"  CT samples: {len(self.test_raw_samples_by_team['CT'])}")
        
        predictions_list = []
        targets_list = []
        team_sides_list = []
        scaled_predictions_list = []
        scaled_targets_list = []
        
        for sample in combined_samples:
            # Generate multiple predictions for this sample
            multi_predictions, target = self.generate_multiple_predictions(
                sample, num_predictions=100
            )
            
            # Get scaled version for Chamfer distance calculation
            first_pred_unscaled = torch.tensor(multi_predictions[0:1], dtype=torch.float32)
            
            if self.coordinate_scaler is not None:
                first_pred_flat = first_pred_unscaled.view(-1, 3).numpy()
                first_pred_scaled = self.coordinate_scaler.transform(first_pred_flat)
                scaled_pred = torch.tensor(first_pred_scaled.reshape(1, 5, 3), 
                                         dtype=torch.float32)
            else:
                scaled_pred = first_pred_unscaled
            
            # Get target key dynamically (supports both enemy_locations and future_locations)
            target_key = 'enemy_locations' if 'enemy_locations' in sample else 'future_locations'
            scaled_target = sample[target_key]  # Already scaled
            
            # Move to device
            device = next(self.parameters()).device
            scaled_pred = scaled_pred.to(device)
            scaled_target = scaled_target.to(device)
            
            predictions_list.append(multi_predictions)
            targets_list.append(target)
            team_sides_list.append(sample['team_side'])
            scaled_predictions_list.append(scaled_pred)
            scaled_targets_list.append(scaled_target)
        
        # Create heatmap grid
        create_prediction_heatmaps_grid(
            predictions_list, targets_list, team_sides_list,
            scaled_predictions_list, scaled_targets_list,
            plots_dir, map_name="de_mirage"
        )
    
    # ============================================================================
    # Optimizer Configuration
    # ============================================================================
    
    def configure_optimizers(self):
        """Configure optimizer for training."""
        opt_config = self.cfg.optimization
        lr = opt_config.lr
        weight_decay = opt_config.weight_decay
        fused_optimizer = opt_config.fused_optimizer
        
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr,
            weight_decay=weight_decay,
            fused=fused_optimizer,
        )
        
        return {'optimizer': optimizer}