"""
Multi-Agent Enemy Location Prediction Model

This module implements a PyTorch Lightning model for predicting enemy locations
in multi-agent scenarios using multi-label classification over predefined locations.
"""

import torch
import torch.nn as nn
import lightning as L
from torch.optim import AdamW
import torch._dynamo
import numpy as np
from pathlib import Path
from torchmetrics.classification import (
    MultilabelHammingDistance, 
    MultilabelExactMatch,
    MultilabelF1Score
)

# Import refactored components
from models.prediction_losses import LossComputer
from models.prediction_metrics import MetricsCalculator
from models.architecture_utils import ACT2CLS, build_mlp
from models.test_analyzer import TestAnalyzer
from pathlib import Path

try:
    from video_encoder import VideoEncoder, get_embed_dim_for_model_type
    from agent_fuser import AgentFuser
    from cross_ego_contrastive import CrossEgoContrastive
except ImportError:
    from .video_encoder import VideoEncoder, get_embed_dim_for_model_type
    from .agent_fuser import AgentFuser
    from .cross_ego_contrastive import CrossEgoContrastive


class CrossEgoVideoLocationNet(L.LightningModule):
    """
    Multi-agent enemy location prediction model supporting multiple task formulations.
    
    The model encodes multi-agent video inputs and team information to predict
    enemy locations using various output representations based on the task form.
    
    Video frames are sampled with optional temporal jitter (configured via 
    data.time_jitter_max_seconds) to augment training data and improve robustness.
    
    Memory Optimization:
        When cfg.data.use_precomputed_embeddings=True, the video encoder is NOT 
        initialized, significantly reducing memory usage and checkpoint size. 
        The model expects precomputed embeddings as input in this mode.
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        # Core model components
        # Only initialize video encoder if not using precomputed embeddings
        self.use_precomputed_embeddings = cfg.data.use_precomputed_embeddings
        
        if self.use_precomputed_embeddings:
            # Get embed_dim without initializing the full encoder (saves memory)
            video_embed_dim = get_embed_dim_for_model_type(cfg.model.encoder.video.model_type)
            self.video_encoder = None  # Don't initialize encoder when using precomputed embeddings
            print(f"[Memory Optimization] Video encoder NOT initialized (using precomputed embeddings)")
            print(f"[Memory Optimization] Using {cfg.model.encoder.video.model_type} embed_dim: {video_embed_dim}")
        else:
            # Initialize full video encoder
            self.video_encoder = VideoEncoder(cfg.model.encoder.video)
            video_embed_dim = self.video_encoder.embed_dim
            print(f"[Model Init] Video encoder initialized: {cfg.model.encoder.video.model_type} (embed_dim: {video_embed_dim})")
        
        self.video_projector = build_mlp(
            input_dim=video_embed_dim,
            output_dim=cfg.model.encoder.proj_dim,
            num_hidden_layers=1,
            hidden_dim=cfg.model.encoder.proj_dim,
            dropout=cfg.model.dropout,
            activation=cfg.model.activation
        )
        
        self.num_agents = cfg.data.num_pov_agents
        
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
        
        # Initialize task-specific components (multi-label classification)
        self.output_dim = cfg.num_places
        self.predictor = build_mlp(
            self.combined_dim, self.output_dim, 
            num_hidden_layers, hidden_dim, dropout,
            activation=cfg.model.activation
        )
        self._init_multilabel_metrics(cfg.num_places)
        
        # Initialize loss and metrics calculators
        loss_fn = cfg.model.loss_fn
        
        self.loss_computer = LossComputer(loss_fn, cfg)
        self.metrics_calculator = MetricsCalculator()
        
        # Test tracking
        self.test_predictions = []
        self.test_targets = []
        self.output_dir = cfg.path.exp
    
    @staticmethod
    def _get_target_locations(batch):
        if 'enemy_locations' in batch:
            return batch['enemy_locations']
        elif 'teammate_locations' in batch:
            return batch['teammate_locations']
        else:
            raise KeyError("Batch must contain either 'enemy_locations' or 'teammate_locations'")
    
    def _init_multilabel_metrics(self, num_labels):
        """Initialize train/val metric pairs for multilabel classification."""
        for split in ['train', 'val']:
            setattr(self, f'{split}_hamming', MultilabelHammingDistance(num_labels=num_labels))
            setattr(self, f'{split}_exact_match', MultilabelExactMatch(num_labels=num_labels))
            setattr(self, f'{split}_micro_f1', MultilabelF1Score(num_labels=num_labels, average='micro'))
            setattr(self, f'{split}_macro_f1', MultilabelF1Score(num_labels=num_labels, average='macro'))
    
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
                - enemy_locations or teammate_locations: [B, num_places] - multi-hot targets
            
        Returns:
            Dictionary containing:
                - predictions: Model predictions (logits)
                - fused_embeddings: Fused video features
                - team_embeddings: Team embeddings
                - combined_features: Combined video + team features
                - agent_embeddings: Per-agent embeddings
        """
        video = batch['video']  # [B, A, T, C, H, W] or [B, A, embed_dim] if using pre-computed embeddings
        pov_team_side_encoded = batch['pov_team_side_encoded']  # [B]
        
        # Check if using pre-computed embeddings (shape is [B, A, embed_dim] instead of [B, A, T, C, H, W])
        if len(video.shape) == 3:
            # Pre-computed embeddings: [B, A, embed_dim]
            agent_embeddings = video  # Already embeddings, just rename
            # Still apply projector to match expected dimensions
            B, A, embed_dim = agent_embeddings.shape
            agent_embeddings = agent_embeddings.view(B * A, embed_dim)
            agent_embeddings = self.video_projector(agent_embeddings)  # [B*A, proj_dim]
            agent_embeddings = agent_embeddings.view(B, A, -1)  # [B, A, proj_dim]
        elif len(video.shape) == 6:
            # Raw videos: [B, A, T, C, H, W] - need to encode
            if self.video_encoder is None:
                raise RuntimeError(
                    "Received raw video input but video_encoder is not initialized. "
                    "This happens when use_precomputed_embeddings=True but raw videos are provided. "
                    "Either set use_precomputed_embeddings=False or provide precomputed embeddings."
                )
            B, A, T, C, H, W = video.shape
            video_reshaped = video.view(B * A, T, C, H, W)
            agent_embeddings = self.video_encoder(video_reshaped)  # [B*A, embed_dim]
            agent_embeddings = self.video_projector(agent_embeddings)  # [B*A, proj_dim]
            agent_embeddings = agent_embeddings.view(B, A, -1)  # [B, A, proj_dim]
        else:
            raise ValueError(f"Expected video shape [B, A, T, C, H, W] or [B, A, embed_dim], got {video.shape}")
        
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
        
        # Standard flow: fuse agent embeddings
        fused_embeddings = self.agent_fuser(agent_embeddings_for_inference)  # [B, proj_dim]
        
        # Encode team information
        team_embeddings = self.team_encoder(pov_team_side_encoded)  # [B, team_embed_dim]
        combined_features = torch.cat([fused_embeddings, team_embeddings], dim=1)
        
        # Multi-label classification prediction
        predictions = self.predictor(combined_features)
        
        outputs = {
            'fused_embeddings': fused_embeddings,
            'team_embeddings': team_embeddings,
            'combined_features': combined_features,
            'agent_embeddings': agent_embeddings_for_inference,
            'contrastive_loss': contrastive_loss,
            'contrastive_metrics': contrastive_metrics,
            'predictions': predictions
        }
        
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
        
        # Log multi-label classification metrics
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
        
        # Log multi-label classification metrics
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
    
    def _get_test_metric_prefix(self):
        """Get the metric prefix for test logging based on checkpoint type."""
        checkpoint_name = getattr(self, 'checkpoint_name', 'last')
        return f'test/{checkpoint_name}'
    
    def on_test_start(self):
        """Initialize test tracking variables."""
        self.test_pov_team_sides = []
        self.test_targets = []
        self.test_predictions = []
        self.test_raw_samples_by_team = {'t': [], 'ct': []}
        
        # Initialize test metrics for multi-label classification
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
        
        # Log multi-label classification metrics
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
        test_results['team_distribution'] = dict(zip(unique_teams.tolist(), team_counts.tolist()))
        
        # Save results to JSON using TestAnalyzer
        analyzer.save_results_to_json(test_results, plots_dir)
        
        # Create visualization plots using TestAnalyzer
        analyzer.create_visualization_plots(predictions, targets, plots_dir, pov_team_sides)
    
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