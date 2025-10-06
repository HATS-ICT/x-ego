"""
Multi-Agent Enemy Location Prediction Model

This module implements a PyTorch Lightning model for predicting enemy locations
in multi-agent scenarios, supporting six different task formulations:

Task Forms:
- coord-reg: Direct regression of (x, y, z) coordinates for each agent
- coord-gen: VAE-based coord-gen model for sampling agent coordinates
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
from pathlib import Path
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from torchmetrics.classification import (
    MultilabelHammingDistance, 
    MultilabelExactMatch,
    MultilabelF1Score
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments


# Import refactored components
from models.prediction_losses import LossComputer
from models.prediction_metrics import MetricsCalculator
from models.coordinate_scaler import unscale_coordinates
from models.vae import ConditionalVariationalAutoencoder
from models.architecture_utils import ACT2CLS, build_mlp
from models.test_analyzer import TestAnalyzer
from pathlib import Path

try:
    from video_encoder import VideoEncoder
    from agent_fuser import AgentFuser
    from cross_ego_contrastive import CrossEgoContrastive
    from team_trajectory_decoder import TeamTrajectoryDecoder
except ImportError:
    from .video_encoder import VideoEncoder
    from .agent_fuser import AgentFuser
    from .cross_ego_contrastive import CrossEgoContrastive
    from .team_trajectory_decoder import TeamTrajectoryDecoder


class CrossEgoVideoLocationNet(L.LightningModule):
    """
    Multi-agent enemy location prediction model supporting multiple task formulations.
    
    The model encodes multi-agent video inputs and team information to predict
    enemy locations using various output representations based on the task form.
    
    Video frames are sampled with optional temporal jitter (configured via 
    data.time_jitter_max_seconds) to augment training data and improve robustness.
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        # Core model components
        self.video_encoder = VideoEncoder(cfg.model.encoder.video)
        
        self.video_projector = build_mlp(
            input_dim=self.video_encoder.embed_dim,
            output_dim=cfg.model.encoder.proj_dim,
            num_hidden_layers=1,
            hidden_dim=cfg.model.encoder.proj_dim,
            dropout=cfg.model.dropout,
            activation=cfg.model.activation
        )
        
        self.num_agents = cfg.data.num_pov_agents
        self.task_form = cfg.data.task_form
        
        # Optional contrastive learning module (between video encoder and agent fuser)
        self.use_contrastive = cfg.model.contrastive.enable
        if self.use_contrastive:
            self.contrastive = CrossEgoContrastive(
                embed_dim=cfg.model.encoder.proj_dim,
                init_logit_scale=cfg.model.contrastive.logit_scale_init,
                init_logit_bias=cfg.model.contrastive.logit_bias_init,
                learnable_temp=True,
                turn_off_bias=cfg.model.contrastive.turn_off_bias
            )
        
        # Agent fusion module
        self.agent_fuser = AgentFuser(
            embed_dim=cfg.model.encoder.proj_dim,
            num_agents=self.num_agents,
            fusion_cfg=cfg.model.agent_fusion,
            activation_fn=ACT2CLS[cfg.model.activation],
            dropout=cfg.model.dropout
        )
        self.fused_agent_dim = cfg.model.agent_fusion.fused_agent_dim
        
        # Team encoder: embed team side (T=0, CT=1) into a vector
        self.team_embed_dim = cfg.model.team_embed_dim
        self.team_encoder = nn.Embedding(2, self.team_embed_dim)  # 2 teams: T, CT
        
        # Predictor head configuration
        hidden_dim = cfg.model.hidden_dim
        dropout = cfg.model.dropout
        num_hidden_layers = cfg.model.num_hidden_layers
        
        # Combined dimension: fused agent features + team embeddings
        self.combined_dim = self.fused_agent_dim + self.team_embed_dim
        
        # Initialize task-specific components
        self._init_task_specific_components(cfg, hidden_dim, dropout, num_hidden_layers)
        
        # Initialize trajectory decoder for traj-gen
        if self.task_form == 'traj-gen':
            self.trajectory_decoder = TeamTrajectoryDecoder(cfg)
        
        # Initialize loss and metrics calculators
        # Extract task-specific loss function
        loss_fn = cfg.model.loss_fn[self.task_form]
        
        self.loss_computer = LossComputer(
            self.task_form, loss_fn, cfg
        )
        self.metrics_calculator = MetricsCalculator(self.task_form)
        
        # Scaler path for coordinate-based tasks
        self.scaler_path = Path(cfg.path.data) / "trajectory_minmax_scaler.pkl"
        
        # Test tracking
        self.test_predictions = []
        self.test_targets = []
        self.output_dir = cfg.path.exp
    
    @staticmethod
    def _get_target_locations(batch):
        if 'enemy_locations' in batch:
            return batch['enemy_locations']
        elif 'future_locations' in batch:
            return batch['future_locations']
        elif 'teammate_locations' in batch:
            return batch['teammate_locations']
        elif 'trajectories' in batch:
            return batch['trajectories']
        else:
            raise KeyError("Batch must contain either 'enemy_locations', 'future_locations', 'teammate_locations', or 'trajectories'")
    
    def _init_multilabel_metrics(self, num_labels):
        """Initialize train/val metric pairs for multilabel classification."""
        for split in ['train', 'val']:
            setattr(self, f'{split}_hamming', MultilabelHammingDistance(num_labels=num_labels))
            setattr(self, f'{split}_exact_match', MultilabelExactMatch(num_labels=num_labels))
            setattr(self, f'{split}_micro_f1', MultilabelF1Score(num_labels=num_labels, average='micro'))
            setattr(self, f'{split}_macro_f1', MultilabelF1Score(num_labels=num_labels, average='macro'))
    
    def _init_task_specific_components(self, cfg, hidden_dim, dropout, num_hidden_layers):
        """Initialize task-specific output heads and metrics."""
        if self.task_form in ['coord-reg', 'coord-gen']:
            # Coordinate regression: output [num_agents * 2 coordinates] (X, Y only)
            self.output_dim = cfg.model.num_target_agents * 2
            self.train_mse, self.val_mse = MeanSquaredError(), MeanSquaredError()
            self.train_mae, self.val_mae = MeanAbsoluteError(), MeanAbsoluteError()
            
            if self.task_form == 'coord-gen':
                # VAE-specific components
                self.latent_dim = cfg.model.vae.latent_dim
                self.vae = ConditionalVariationalAutoencoder(
                    output_dim=self.output_dim,
                    combined_dim=self.combined_dim,
                    latent_dim=self.latent_dim,
                    num_hidden_layers=num_hidden_layers,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    activation_fn=ACT2CLS[cfg.model.activation]
                )
            else:
                # Standard coordinate regression
                self.predictor = build_mlp(
                    self.combined_dim, self.output_dim, 
                    num_hidden_layers, hidden_dim, dropout,
                    activation=cfg.model.activation
                )
        
        elif self.task_form == 'traj-gen':
            # Trajectory generation: output [5, 60, 2]
            self.train_mse, self.val_mse = MeanSquaredError(), MeanSquaredError()
            self.train_mae, self.val_mae = MeanAbsoluteError(), MeanAbsoluteError()
        
        elif self.task_form in ['multi-label-cls', 'multi-output-reg']:
            # Place-based tasks: output [num_places]
            self.output_dim = cfg.num_places
            self.predictor = build_mlp(
                self.combined_dim, self.output_dim, 
                num_hidden_layers, hidden_dim, dropout,
                activation=cfg.model.activation
            )
            if self.task_form == 'multi-label-cls':
                self._init_multilabel_metrics(cfg.num_places)
        
        elif self.task_form in ['grid-cls', 'density-cls']:
            # Grid-based tasks: output [grid_resolution^2]
            grid_resolution = cfg.data.grid_resolution
            self.output_dim = grid_resolution * grid_resolution
            self.predictor = build_mlp(
                self.combined_dim, self.output_dim, 
                num_hidden_layers, hidden_dim, dropout,
                activation=cfg.model.activation
            )
            if self.task_form == 'grid-cls':
                self._init_multilabel_metrics(self.output_dim)
    
    # ============================================================================
    # Forward Pass
    # ============================================================================
    
    def _select_agent_subset(self, agent_embeddings, num_agents):
        """
        Randomly select a subset of agent embeddings.
        
        Args:
            agent_embeddings: [B, A, proj_dim] agent embeddings
            num_agents: number of agents to select
            
        Returns:
            [B, num_agents, proj_dim] selected agent embeddings
        """
        B, A, proj_dim = agent_embeddings.shape
        
        if A > num_agents:
            # Randomly select num_agents indices for each batch
            selected_indices = torch.stack([
                torch.randperm(A, device=agent_embeddings.device)[:num_agents]
                for _ in range(B)
            ])  # [B, num_agents]
            
            # Gather selected embeddings: [B, num_agents, proj_dim]
            selected_embeddings = torch.gather(
                agent_embeddings,
                dim=1,
                index=selected_indices.unsqueeze(-1).expand(-1, -1, proj_dim)
            )
            return selected_embeddings
        else:
            return agent_embeddings
    
    def forward(self, batch, mode='full'):
        """
        Forward pass through the model.
        
        Args:
            batch: Input batch containing:
                - video: [B, A, T, C, H, W] - multi-agent video
                  When contrastive is enabled: A=5 (all agents), then num_pov_agents randomly selected for inference
                  When contrastive is disabled: A=num_pov_agents
                - pov_team_side_encoded: [B] - team encoding (0=T, 1=CT)
                - enemy_locations, future_locations, or teammate_locations: [B, 5, 3] - targets (for training/full mode)
            mode: 'full' for full VAE forward pass, 'sampling' for sampling from prior
            
        Returns:
            Dictionary containing:
                - predictions: Model predictions
                - fused_embeddings: Fused video features
                - team_embeddings: Team embeddings
                - combined_features: Combined video + team features
                - agent_embeddings: Per-agent embeddings
                - mu, logvar, z: VAE latent variables (coord-gen mode only)
        """
        video = batch['video']  # [B, A, T, C, H, W]
        pov_team_side_encoded = batch['pov_team_side_encoded']  # [B]
        
        if len(video.shape) != 6:
            raise ValueError(f"Expected video shape [B, A, T, C, H, W], got {video.shape}")
        
        # Encode each agent's video
        B, A, T, C, H, W = video.shape
        video_reshaped = video.view(B * A, T, C, H, W)
        agent_embeddings = self.video_encoder(video_reshaped)  # [B*A, embed_dim]
        agent_embeddings = self.video_projector(agent_embeddings)  # [B*A, proj_dim]
        agent_embeddings = agent_embeddings.view(B, A, -1)  # [B, A, proj_dim]
        
        # Optional contrastive learning (align agents from same batch)
        contrastive_loss = None
        contrastive_metrics = None
        if self.use_contrastive:
            # Compute contrastive loss/metrics during training or validation (not test)
            contrastive_out = self.contrastive(agent_embeddings)
            agent_embeddings_contrastive = contrastive_out['embeddings']
            contrastive_loss = contrastive_out['loss']
            contrastive_metrics = contrastive_out['retrieval_metrics']
            # Add temperature and bias to metrics
            contrastive_metrics['temperature'] = contrastive_out['temperature']
            if contrastive_out['bias'] is not None:
                contrastive_metrics['bias'] = contrastive_out['bias']
            
            # When contrastive is enabled, randomly select num_agents from all A agents for inference
            # This happens after video embeddings are computed to avoid recomputation
            agent_embeddings_for_inference = self._select_agent_subset(
                agent_embeddings_contrastive, self.num_agents
            )
        else:
            agent_embeddings_for_inference = agent_embeddings
        
        # Task-specific forward pass
        if self.task_form == 'traj-gen':
            # Trajectory generation: bypass agent fuser, use per-agent embeddings directly
            target_team_encoded = batch['target_team_side_encoded']  # [B]
            predictions = self.trajectory_decoder(
                agent_embeddings_for_inference, 
                pov_team_side_encoded,
                target_team_encoded
            )  # [B, 5, 60, 2]
            
            outputs = {
                'agent_embeddings': agent_embeddings_for_inference,
                'predictions': predictions,
                'contrastive_loss': contrastive_loss,
                'contrastive_metrics': contrastive_metrics
            }
        else:
            # Standard flow: fuse agent embeddings
            fused_embeddings = self.agent_fuser(agent_embeddings_for_inference)  # [B, proj_dim]
            
            # Encode team information
            team_embeddings = self.team_encoder(pov_team_side_encoded)  # [B, team_embed_dim]
            combined_features = torch.cat([fused_embeddings, team_embeddings], dim=1)
            
            outputs = {
                'fused_embeddings': fused_embeddings,
                'team_embeddings': team_embeddings,
                'combined_features': combined_features,
                'agent_embeddings': agent_embeddings_for_inference,
                'contrastive_loss': contrastive_loss,
                'contrastive_metrics': contrastive_metrics
            }
            
            if self.task_form == 'coord-gen':
                # coord-gen (VAE) mode
                if mode == 'sampling':
                    vae_outputs = self.vae(None, combined_features, mode='sampling')
                else:
                    target_locations = self._get_target_locations(batch)
                    vae_outputs = self.vae(target_locations, combined_features, mode='full')
                
                outputs.update({
                    'predictions': vae_outputs['predictions'],
                    'mu': vae_outputs['mu'],
                    'logvar': vae_outputs['logvar'],
                    'z': vae_outputs['z']
                })
            else:
                # Standard (non-coord-gen) mode
                predictions = self.predictor(combined_features)
                if self.task_form == 'coord-reg':
                    predictions = predictions.view(-1, 5, 2)
                outputs['predictions'] = predictions
        
        return outputs
    
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
    
    def _create_trajectory_visualization(self, predictions, targets, num_samples=5):
        """
        Create a 2x5 subplot visualization of trajectories.
        
        Args:
            predictions: [N, 5, 60, 2] predicted trajectories
            targets: [N, 5, 60, 2] ground truth trajectories
            num_samples: Number of samples to visualize (default 5)
            
        Returns:
            matplotlib Figure object
        """
        # Select random samples if we have more than num_samples
        N = predictions.shape[0]
        if N > num_samples:
            indices = np.random.choice(N, num_samples, replace=False)
        else:
            indices = np.arange(min(N, num_samples))
        
        # Create 2x5 subplot (2 rows: ground truth, predictions)
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))
        if num_samples == 1:
            axes = axes.reshape(2, 1)
        
        for i, idx in enumerate(indices):
            pred = predictions[idx].float().cpu().numpy()  # [5, 60, 2]
            target = targets[idx].float().cpu().numpy()  # [5, 60, 2]
            
            # Plot ground truth (top row)
            ax_gt = axes[0, i]
            for agent_idx in range(5):
                ax_gt.plot(target[agent_idx, :, 0], target[agent_idx, :, 1], 
                          '-o', markersize=2, linewidth=1.5, alpha=0.7, 
                          label=f'Agent {agent_idx+1}')
            ax_gt.set_title(f'Ground Truth {i+1}', fontsize=10, fontweight='bold')
            ax_gt.set_xlabel('X', fontsize=9)
            ax_gt.set_ylabel('Y', fontsize=9)
            ax_gt.grid(True, alpha=0.3)
            ax_gt.axis('equal')
            if i == 0:
                ax_gt.legend(loc='best', fontsize=7)
            
            # Plot predictions (bottom row)
            ax_pred = axes[1, i]
            for agent_idx in range(5):
                ax_pred.plot(pred[agent_idx, :, 0], pred[agent_idx, :, 1], 
                           '-o', markersize=2, linewidth=1.5, alpha=0.7,
                           label=f'Agent {agent_idx+1}')
            ax_pred.set_title(f'Predicted {i+1}', fontsize=10, fontweight='bold')
            ax_pred.set_xlabel('X', fontsize=9)
            ax_pred.set_ylabel('Y', fontsize=9)
            ax_pred.grid(True, alpha=0.3)
            ax_pred.axis('equal')
            if i == 0:
                ax_pred.legend(loc='best', fontsize=7)
        
        plt.tight_layout()
        return fig
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        batch_size = batch["video"].shape[0]
        
        # TODO: Debug: to be removed
        # print(batch.keys())
        # for k, v in batch.items():
        #     if torch.is_tensor(v):
        #         print(k, v.shape)
        #     else:
        #         print(k, v)
        # from utils.training_utils import debug_batch_plot
        # debug_batch_plot(batch, self)
        
        targets = self._get_target_locations(batch)
        
        # Forward pass
        outputs = self.forward(batch, mode='full')
        predictions = outputs['predictions']
        
        # Compute loss
        loss, loss_components = self.loss_computer.compute_loss(outputs, targets)
        
        # Add contrastive loss if enabled
        if outputs['contrastive_loss'] is not None:
            contrastive_loss = outputs['contrastive_loss']
            loss = loss + self.cfg.model.contrastive.loss_weight * contrastive_loss
            self.safe_log('train/contrastive_loss', contrastive_loss, batch_size=batch_size,
                         on_step=True, on_epoch=True, prog_bar=False)
            
            # Log contrastive retrieval metrics
            if 'contrastive_metrics' in outputs:
                metrics = outputs['contrastive_metrics']
                for metric_name, metric_value in metrics.items():
                    self.safe_log(f'train/contrastive_{metric_name}', metric_value, 
                                batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=False)
        
        # Log main loss
        self.safe_log('train/loss', loss, batch_size=batch_size, 
                     on_step=True, on_epoch=True, prog_bar=True)
        
        # Log VAE components if applicable
        if loss_components:
            for name, value in loss_components.items():
                self.safe_log(f'train/{name}', value, batch_size=batch_size,
                            on_step=True, on_epoch=True, prog_bar=False)
        
        # Log coordinate-based metrics
        if self.task_form in ['coord-reg', 'coord-gen', 'traj-gen']:
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
    
    def on_validation_epoch_start(self):
        """Initialize validation tracking variables."""
        if self.task_form == 'traj-gen':
            self.val_predictions = []
            self.val_targets = []
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        batch = self._tensor_only_batch(batch)
        batch_size = batch["video"].shape[0]
        targets = self._get_target_locations(batch)
        
        # Forward pass
        outputs = self.forward(batch, mode='full')
        predictions = outputs['predictions']
        
        # Compute loss
        loss, loss_components = self.loss_computer.compute_loss(outputs, targets)
        
        # Log contrastive metrics if enabled (without adding to loss in validation)
        if outputs['contrastive_loss'] is not None:
            contrastive_loss = outputs['contrastive_loss']
            self.safe_log('val/contrastive_loss', contrastive_loss, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=False)
            
            # Log contrastive retrieval metrics  
            if 'contrastive_metrics' in outputs:
                metrics = outputs['contrastive_metrics']
                for metric_name, metric_value in metrics.items():
                    self.safe_log(f'val/contrastive_{metric_name}', metric_value,
                                batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)
        
        # Log main loss (without contrastive loss added in validation)
        self.safe_log('val/loss', loss, batch_size=batch_size,
                     on_step=False, on_epoch=True, prog_bar=True)
        
        # Log VAE components
        if loss_components:
            for name, value in loss_components.items():
                self.safe_log(f'val/{name}', value, batch_size=batch_size,
                            on_step=False, on_epoch=True, prog_bar=False)
        
        # Test coord-gen sampling for VAE
        if self.task_form == 'coord-gen':
            gen_outputs = self.forward(batch, mode='sampling')
            gen_predictions = gen_outputs['predictions']
            gen_loss = self.loss_computer._compute_regression_loss(gen_predictions, targets)
            self.safe_log('val/gen_loss', gen_loss, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=False)
        
        # Log coordinate-based metrics
        if self.task_form in ['coord-reg', 'coord-gen', 'traj-gen']:
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
        
        # Store predictions and targets for trajectory visualization
        if self.task_form == 'traj-gen':
            self.val_predictions.append(predictions.detach())
            self.val_targets.append(targets.detach())
        
        return loss
    
    def on_validation_epoch_end(self):
        """Create and log trajectory visualization at end of validation epoch."""
        if self.task_form == 'traj-gen' and len(self.val_predictions) > 0:
            # Concatenate all predictions and targets
            all_predictions = torch.cat(self.val_predictions, dim=0)  # [N, 5, 60, 2]
            all_targets = torch.cat(self.val_targets, dim=0)  # [N, 5, 60, 2]
            
            # Calculate trajectory-specific metrics
            # ADE (Average Displacement Error): average L2 distance across all timesteps
            # FDE (Final Displacement Error): L2 distance at final timestep
            l2_distances = torch.sqrt(((all_predictions - all_targets) ** 2).sum(dim=-1))  # [N, 5, 60]
            ade = l2_distances.mean()
            fde = l2_distances[:, :, -1].mean()  # Final timestep
            
            # Log trajectory metrics
            self.safe_log('val/ade', ade, on_epoch=True, prog_bar=True)
            self.safe_log('val/fde', fde, on_epoch=True, prog_bar=True)
            
            # Create visualization
            fig = self._create_trajectory_visualization(all_predictions, all_targets, num_samples=5)
            
            # Log to wandb if logger is available
            if self.logger is not None:
                try:
                    import wandb
                    # Convert matplotlib figure to wandb Image
                    self.logger.experiment.log({
                        "val/trajectory_visualization": wandb.Image(fig),
                        "trainer/global_step": self.global_step
                    })
                except Exception as e:
                    print(f"Failed to log trajectory visualization to wandb: {e}")
            
            # Close figure to free memory
            plt.close(fig)
            
            # Clear stored predictions and targets
            self.val_predictions = []
            self.val_targets = []
    
    def _get_test_metric_prefix(self):
        """Get the metric prefix for test logging based on checkpoint type."""
        checkpoint_name = getattr(self, 'checkpoint_name', 'last')
        return f'test/{checkpoint_name}'
    
    def on_test_start(self):
        """Initialize test tracking variables."""
        self.test_pov_team_sides = []
        self.test_targets = []
        self.test_predictions = []
        self.test_targets_unscaled = []
        self.test_predictions_unscaled = []
        self.test_raw_samples_by_team = {'t': [], 'ct': []}
        
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
        pov_team_sides = batch['pov_team_side']
        samples_per_team = 5
        
        for i in range(batch['video'].shape[0]):
            pov_team_side = pov_team_sides[i]
            pov_team_side_lower = pov_team_side.lower()
            if len(self.test_raw_samples_by_team[pov_team_side_lower]) < samples_per_team:
                # Get target locations key dynamically
                if 'enemy_locations' in batch:
                    target_key = 'enemy_locations'
                elif 'future_locations' in batch:
                    target_key = 'future_locations'
                elif 'trajectories' in batch:
                    target_key = 'trajectories'
                else:
                    target_key = 'teammate_locations'
                sample = {
                    'video': batch['video'][i:i+1].clone(),
                    'pov_team_side_encoded': batch['pov_team_side_encoded'][i:i+1].clone(),
                    target_key: batch[target_key][i:i+1].clone(),
                    'pov_team_side': pov_team_side
                }
                self.test_raw_samples_by_team[pov_team_side_lower].append(sample)
            
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
        loss, loss_components = self.loss_computer.compute_loss(outputs, targets)
        
        # Get test metric prefix (test/last or test/best)
        metric_prefix = self._get_test_metric_prefix()
        
        # Log main loss
        self.safe_log(f'{metric_prefix}/loss', loss, batch_size=batch_size,
                     on_step=False, on_epoch=True, prog_bar=True)
        
        # Log contrastive metrics if enabled
        if outputs['contrastive_loss'] is not None:
            contrastive_loss = outputs['contrastive_loss']
            self.safe_log(f'{metric_prefix}/contrastive_loss', contrastive_loss, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=False)
            
            # Log contrastive retrieval metrics
            if 'contrastive_metrics' in outputs:
                metrics = outputs['contrastive_metrics']
                for metric_name, metric_value in metrics.items():
                    self.safe_log(f'{metric_prefix}/contrastive_{metric_name}', metric_value,
                                batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)
        
        # Log VAE components
        if loss_components:
            for name, value in loss_components.items():
                self.safe_log(f'{metric_prefix}/{name}', value, batch_size=batch_size,
                            on_step=False, on_epoch=True, prog_bar=False)
        
        # Test coord-gen sampling for VAE
        if self.task_form == 'coord-gen':
            gen_outputs = self.forward(batch, mode='sampling')
            gen_predictions = gen_outputs['predictions']
            gen_loss = self.loss_computer._compute_regression_loss(gen_predictions, targets)
            self.safe_log(f'{metric_prefix}/gen_loss', gen_loss, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=False)
        
        # Initialize and log coordinate-based metrics
        if self.task_form in ['coord-reg', 'coord-gen', 'traj-gen']:
            if not hasattr(self, 'test_mse'):
                self.test_mse = MeanSquaredError().to(self.device)
                self.test_mae = MeanAbsoluteError().to(self.device)
            
            self.test_mse(predictions.view(batch_size, -1), targets.view(batch_size, -1))
            self.test_mae(predictions.view(batch_size, -1), targets.view(batch_size, -1))
            self.safe_log(f'{metric_prefix}/mse', self.test_mse, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=True)
            self.safe_log(f'{metric_prefix}/mae', self.test_mae, batch_size=batch_size,
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
            self.safe_log(f'{metric_prefix}/hamming_loss', self.test_hamming, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=True)
            self.safe_log(f'{metric_prefix}/subset_accuracy', self.test_exact_match, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=True)
            self.safe_log(f'{metric_prefix}/micro_f1', self.test_micro_f1, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=True)
            self.safe_log(f'{metric_prefix}/macro_f1', self.test_macro_f1, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=True)
        
        # Store predictions and targets
        self.test_pov_team_sides.extend(pov_team_sides)
        
        if self.task_form in ['coord-reg', 'coord-gen']:
            # Store both scaled and unscaled versions
            predictions_unscaled = unscale_coordinates(predictions, self.scaler_path)
            targets_unscaled = unscale_coordinates(targets, self.scaler_path)
            
            self.test_predictions.append(predictions.cpu().float())
            self.test_targets.append(targets.cpu().float())
            self.test_predictions_unscaled.extend(predictions_unscaled.cpu().float().numpy())
            self.test_targets_unscaled.extend(targets_unscaled.cpu().float().numpy())
        elif self.task_form == 'traj-gen':
            # Store trajectory predictions and targets
            self.test_predictions.append(predictions.cpu().float())
            self.test_targets.append(targets.cpu().float())
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
        pov_team_sides = np.array(self.test_pov_team_sides)
        unique_teams, team_counts = np.unique(pov_team_sides, return_counts=True)
        
        # Setup output directory
        output_dir = Path(self.output_dir)
        checkpoint_suffix = f"_{self.checkpoint_name}" if hasattr(self, 'checkpoint_name') else ""
        plots_dir = output_dir / "test_analysis" / f"enemy_location{checkpoint_suffix}"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving enemy location prediction test analysis to: {plots_dir}")
        
        # Initialize TestAnalyzer
        analyzer = TestAnalyzer(self, self.cfg, self.metrics_calculator)
        
        # Calculate metrics using MetricsCalculator
        test_results = self.metrics_calculator.calculate_metrics(predictions, targets)
        
        # Add geometric distances for coordinate-based tasks
        if self.task_form in ['coord-reg', 'coord-gen']:
            geometric_metrics = self.metrics_calculator.calculate_geometric_distances(
                self.test_predictions, self.test_targets
            )
            test_results['geometric_distances'] = geometric_metrics
        
        # Calculate team-specific metrics
        team_specific_metrics = self.metrics_calculator.calculate_team_metrics(
            predictions, targets, pov_team_sides, self.test_predictions, self.test_targets
        )
        test_results['team_specific_metrics'] = team_specific_metrics
        
        # Log metrics using TestAnalyzer
        analyzer.log_team_metrics(team_specific_metrics)
        analyzer.log_overall_metrics(test_results)
        
        # Add experiment metadata
        test_results['num_agents'] = self.cfg.data.num_pov_agents
        test_results['agent_fusion_method'] = self.cfg.model.agent_fusion.method
        test_results['task_form'] = self.task_form
        test_results['team_distribution'] = dict(zip(unique_teams.tolist(), team_counts.tolist()))
        
        # Add task-specific metadata
        if self.task_form in ['coord-reg', 'coord-gen']:
            analyzer.add_coordinate_metadata(test_results)
        
        # Save results to JSON using TestAnalyzer
        analyzer.save_results_to_json(test_results, plots_dir)
        
        # Create visualization plots using TestAnalyzer
        analyzer.create_visualization_plots(predictions, targets, plots_dir, pov_team_sides)
        
        # Create KDE heatmaps for coordinate-based tasks
        if self.task_form in ['coord-reg', 'coord-gen']:
            analyzer.create_prediction_heatmaps(plots_dir, self.test_raw_samples_by_team)
    
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