"""
Contrastive Learning Model (Stage 1: Team Alignment)

This model implements contrastive learning for aligning video embeddings
from the same team. It's the first stage of the two-stage training pipeline:
  Stage 1: Learn team-aligned video representations via contrastive loss
  Stage 2: Linear probing on downstream tasks with frozen encoder

Key features:
- Variable agents per sample with no padding (agents concatenated in first dim)
- Uses agent_counts to track sample boundaries for alignment matrix
- Example: batch with [3, 2] agents -> video [5, T, C, H, W], agent_counts [3, 2]
- Creates alignment matrix where positive pairs are agents from same sample

Multi-task Learning (optional):
- Video reconstruction auxiliary loss for regularization
- L_total = L_contrastive + lambda * L_recon
- Reconstructs a randomly selected single view per sample
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW

from src.models.modules.video_encoder import VideoEncoder
from src.models.modules.video_decoder import VideoDecoder
from src.models.modules.architecture_utils import build_mlp


class ContrastiveModel(L.LightningModule):
    """
    Contrastive learning model for team alignment (Stage 1).
    
    Architecture:
        Video Encoder -> Video Projector -> Contrastive Module
                                        |-> Video Decoder (optional reconstruction)
    
    The model learns to align agent embeddings from the same team/batch
    while pushing apart embeddings from different teams/batches.
    
    Multi-task Learning (optional):
        When reconstruction is enabled, adds auxiliary reconstruction loss:
        L_total = L_contrastive + lambda * L_recon
        Only reconstructs 1 randomly selected view per sample for efficiency.
    
    Variable Agent Support:
        Supports batches with variable number of agents per sample.
        Agents are concatenated in the first dimension (no padding).
        Uses agent_counts tensor to track sample boundaries.
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        # Video encoder setup
        self.video_encoder = VideoEncoder(cfg.model.encoder)
        video_embed_dim = self.video_encoder.embed_dim
        print(f"[Model Init] Video encoder: {cfg.model.encoder.model_type} (dim: {video_embed_dim})")
        
        # Video projector
        proj_dim = cfg.model.projector.proj_dim
        self.video_projector = build_mlp(
            input_dim=video_embed_dim,
            output_dim=proj_dim,
            num_hidden_layers=cfg.model.projector.num_hidden_layers,
            hidden_dim=proj_dim,
            activation=cfg.model.activation
        )
        
        # Contrastive parameters (learnable temperature and bias, following SigLIP)
        self.turn_off_bias = cfg.model.contrastive.turn_off_bias
        self.logit_scale = nn.Parameter(
            torch.tensor(cfg.model.contrastive.logit_scale_init, dtype=torch.float32).log(),
            requires_grad=True
        )
        self.logit_bias = nn.Parameter(
            torch.tensor(cfg.model.contrastive.logit_bias_init, dtype=torch.float32),
            requires_grad=True
        )
        
        # Reconstruction decoder (optional)
        self.use_reconstruction = getattr(cfg.model, 'reconstruction', None) is not None \
                                  and cfg.model.reconstruction.enable
        if self.use_reconstruction:
            recon_cfg = cfg.model.reconstruction
            self.recon_loss_weight = recon_cfg.loss_weight
            self.recon_target_size = recon_cfg.target_frame_size
            self.recon_num_frames = recon_cfg.num_reconstruct_frames
            
            self.video_decoder = VideoDecoder(
                input_dim=proj_dim,
                d_model=getattr(recon_cfg, 'd_model', 512),
                n_heads=getattr(recon_cfg, 'n_heads', 8),
                depth=getattr(recon_cfg, 'depth', 4),
                n_latents=getattr(recon_cfg, 'n_latents', 1),
                target_frame_size=recon_cfg.target_frame_size,
                num_frames=recon_cfg.num_reconstruct_frames,
                patch_size=getattr(recon_cfg, 'patch_size', 8),
                dropout=getattr(recon_cfg, 'dropout', 0.0),
                mlp_ratio=getattr(recon_cfg, 'mlp_ratio', 4.0),
                time_every=getattr(recon_cfg, 'time_every', 2),
            )
            print(f"[Model Init] Reconstruction enabled: lambda={self.recon_loss_weight}, "
                  f"target_size={self.recon_target_size}, num_frames={self.recon_num_frames}, "
                  f"d_model={getattr(recon_cfg, 'd_model', 512)}")
        else:
            self.video_decoder = None
        
        # Store dimensions
        self.video_embed_dim = video_embed_dim
        self.proj_dim = proj_dim
        
        # Output directory
        self.output_dir = cfg.path.exp
    
    def create_alignment_matrix(self, agent_counts: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Create alignment matrix for variable number of agents per sample.
        
        For contrastive learning, positive pairs are agents from the same sample.
        This creates a block-diagonal matrix where each block corresponds to 
        agents from the same sample.
        
        Args:
            agent_counts: [B] Number of agents per sample
            device: Device to create tensor on
            
        Returns:
            labels: [total_agents, total_agents] alignment matrix
                   1 for same-sample pairs, 0 otherwise
                   
        Example:
            agent_counts = [3, 2] -> 5x5 matrix with two blocks:
            [[1,1,1,0,0],
             [1,1,1,0,0],
             [1,1,1,0,0],
             [0,0,0,1,1],
             [0,0,0,1,1]]
        """
        # Create batch assignment for each agent using repeat_interleave
        batch_indices = torch.repeat_interleave(
            torch.arange(len(agent_counts), device=device),
            agent_counts
        )
        
        # Create alignment matrix: 1 if same batch, 0 otherwise
        labels = (batch_indices.unsqueeze(0) == batch_indices.unsqueeze(1)).float()
        
        return labels
    
    def forward(self, batch):
        """
        Forward pass for contrastive learning with optional reconstruction.
        
        Args:
            batch: Dictionary containing:
                - video: [total_agents, T, C, H, W] concatenated video tensors
                - agent_counts: [B] number of agents per sample
                
        Returns:
            Dictionary containing:
                - embeddings: [total_agents, proj_dim] projected embeddings
                - loss: Total loss (contrastive + lambda * reconstruction if enabled)
                - metrics: Dictionary of metrics
        """
        video = batch['video']  # [total_agents, T, C, H, W]
        agent_counts = batch['agent_counts']  # [B]
        
        device = video.device
        
        # Process videos: [total_agents, T, C, H, W]
        if len(video.shape) != 5:
            raise ValueError(f"Expected video shape [total_agents, T, C, H, W], got {video.shape}")
        
        # Encode videos
        embeddings = self.video_encoder(video)  # [total_agents, embed_dim]
        
        # Project embeddings
        projected = self.video_projector(embeddings)  # [total_agents, proj_dim]
        
        # Create alignment labels
        labels = self.create_alignment_matrix(agent_counts, device)
        
        # Compute contrastive loss
        contrastive_loss, metrics = self.compute_contrastive_loss(projected, labels)
        
        # Total loss starts with contrastive
        total_loss = contrastive_loss
        
        # Add reconstruction loss if enabled
        if self.use_reconstruction:
            # Use unmasked video as reconstruction target if available
            # (when random_mask is enabled, we want to reconstruct the unmasked version)
            recon_target = batch.get('video_unmasked', video)
            recon_loss, recon_metrics = self.compute_reconstruction_loss(
                video, recon_target, projected, agent_counts
            )
            total_loss = contrastive_loss + self.recon_loss_weight * recon_loss
            metrics.update(recon_metrics)
            metrics['recon_loss_raw'] = recon_loss
            metrics['recon_loss_weighted'] = self.recon_loss_weight * recon_loss
        
        return {
            'embeddings': projected,
            'loss': total_loss,
            'contrastive_loss': contrastive_loss,
            'metrics': metrics,
            'labels': labels,
        }
    
    def compute_reconstruction_loss(self, video: torch.Tensor,
                                    recon_target: torch.Tensor,
                                    projected: torch.Tensor,
                                    agent_counts: torch.Tensor):
        """
        Compute reconstruction loss for a selected view per sample.
        
        For efficiency, we only reconstruct 1 agent's view per sample,
        not all agents. This provides regularization without excessive compute.
        
        During training: randomly select 1 agent per sample
        During eval: deterministically select first agent per sample
        
        Args:
            video: [total_agents, T, C, H, W] input video (possibly masked)
            recon_target: [total_agents, T, C, H, W] reconstruction target (unmasked if random_mask enabled)
            projected: [total_agents, proj_dim] projected embeddings
            agent_counts: [B] number of agents per sample
            
        Returns:
            loss: Reconstruction MSE loss
            metrics: Dictionary with reconstruction metrics
        """
        device = video.device
        batch_size = len(agent_counts)
        
        # Build cumulative indices to find sample boundaries
        cumsum = torch.cumsum(agent_counts, dim=0)
        start_indices = torch.cat([torch.tensor([0], device=device), cumsum[:-1]])
        
        # Select 1 agent per sample for reconstruction
        selected_indices = []
        for i in range(batch_size):
            start = start_indices[i].item()
            count = agent_counts[i].item()
            if self.training:
                # Random index within this sample's agents during training
                rand_offset = torch.randint(0, count, (1,), device=device).item()
            else:
                # Deterministic: use first agent during eval
                rand_offset = 0
            selected_indices.append(start + rand_offset)
        
        selected_indices = torch.tensor(selected_indices, device=device, dtype=torch.long)
        
        # Get selected embeddings (from masked video encoder output)
        selected_proj = projected[selected_indices]  # [B, proj_dim]
        
        # Get selected target videos (unmasked version for reconstruction)
        selected_target = recon_target[selected_indices]  # [B, T, C, H, W]
        
        # Prepare target: downsample and temporally sample the video
        target = self._prepare_reconstruction_target(selected_target)  # [B, num_frames, 3, H, W]
        
        # Decode embeddings to reconstruct frames
        reconstructed = self.video_decoder(selected_proj)  # [B, num_frames, 3, H, W]
        
        # MSE loss
        loss = F.mse_loss(reconstructed, target)
        
        metrics = {
            'recon_mse': loss.detach(),
        }
        
        return loss, metrics
    
    def _prepare_reconstruction_target(self, video: torch.Tensor) -> torch.Tensor:
        """
        Prepare reconstruction target by downsampling spatially and temporally.
        
        Args:
            video: [B, T, C, H, W] input video
            
        Returns:
            target: [B, num_frames, C, target_H, target_W] downsampled video
        """
        B, T, C, H, W = video.shape
        
        # Temporal sampling: uniformly sample num_frames from T frames
        if T <= self.recon_num_frames:
            # If fewer frames than target, just use all and pad/repeat
            indices = torch.linspace(0, T - 1, self.recon_num_frames).long()
        else:
            indices = torch.linspace(0, T - 1, self.recon_num_frames).long()
        
        video_sampled = video[:, indices]  # [B, num_frames, C, H, W]
        
        # Spatial downsampling
        # Reshape for interpolate: [B * num_frames, C, H, W]
        video_flat = video_sampled.view(-1, C, H, W)
        video_down = F.interpolate(
            video_flat, 
            size=(self.recon_target_size, self.recon_target_size),
            mode='bilinear',
            align_corners=False
        )
        
        # Reshape back: [B, num_frames, C, target_H, target_W]
        target = video_down.view(B, self.recon_num_frames, C, 
                                 self.recon_target_size, self.recon_target_size)
        
        return target
    
    def compute_contrastive_loss(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Compute contrastive loss with custom alignment labels.
        
        Uses sigmoid loss formulation (similar to SigLIP).
        
        Args:
            embeddings: [N, proj_dim] normalized embeddings
            labels: [N, N] alignment matrix (1 for positive pairs, 0 for negative)
            
        Returns:
            loss: Contrastive loss scalar
            metrics: Dictionary of metrics
        """
        # L2 normalize embeddings
        normalized = F.normalize(embeddings, p=2, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(normalized, normalized.t())
        
        # Apply learnable temperature and bias
        logit_scale = self.logit_scale.exp()
        logits = logits * logit_scale
        if not self.turn_off_bias:
            logits = logits + self.logit_bias
        
        # Compute sigmoid loss
        m1_diag1 = 2 * labels - 1  # Convert {0,1} to {-1,1}
        loglik = F.logsigmoid(m1_diag1 * logits)
        
        N = logits.shape[0]
        diag_mask = torch.eye(N, device=logits.device, dtype=torch.bool)
        # Set diagonal log-likelihoods to 0.0
        # Since loss is -sum(loglik), adding 0.0 effectively ignores these terms
        loglik = loglik.masked_fill(diag_mask, 0.0)
        
        nll = -torch.sum(loglik, dim=-1)
        loss = nll.mean()
        
        # Compute metrics
        metrics = self._compute_metrics(logits, labels, logit_scale)
        
        return loss, metrics
    
    def _compute_metrics(self, logits: torch.Tensor, labels: torch.Tensor, 
                         temperature: torch.Tensor) -> dict:
        """Compute contrastive learning metrics.
        """
        N = logits.shape[0]
        
        # Binary classification accuracy
        pred_probs = torch.sigmoid(logits)
        pred_binary = (pred_probs > 0.5).float()
        binary_acc = (pred_binary == labels).float().mean()
        
        # Positive/negative pair accuracies
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        metrics = {
            'binary_acc': binary_acc,
            'temperature': temperature,
        }
        
        if pos_mask.any():
            metrics['pos_pair_acc'] = (pred_binary[pos_mask] == labels[pos_mask]).float().mean()
        if neg_mask.any():
            metrics['neg_pair_acc'] = (pred_binary[neg_mask] == labels[neg_mask]).float().mean()
        
        if not self.turn_off_bias:
            metrics['bias'] = self.logit_bias.item()
        
        # Retrieval metrics (simplified for variable agents)
        if N > 1:
            metrics.update(self._compute_retrieval_metrics(logits, labels))
        
        return metrics
    
    def _compute_retrieval_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> dict:
        """Compute retrieval accuracy metrics."""
        N = logits.shape[0]
        
        # Mask out self-similarity
        mask = torch.eye(N, device=logits.device, dtype=torch.bool)
        masked_logits = logits.clone()
        masked_logits[mask] = float('-inf')
        
        # Get top-k indices
        k_max = min(5, N - 1)
        _, top_k_indices = torch.topk(masked_logits, k=k_max, dim=1)
        
        # Ground truth labels (excluding self)
        gt_labels = labels.clone()
        gt_labels[mask] = 0
        
        metrics = {}
        
        for k in [1, 3, 5]:
            if k > k_max:
                continue
            
            top_k_idx = top_k_indices[:, :k]
            batch_indices = torch.arange(N, device=logits.device).unsqueeze(1).expand(-1, k)
            top_k_labels = gt_labels[batch_indices, top_k_idx]
            has_positive = (top_k_labels.sum(dim=1) > 0).float()
            metrics[f'top{k}_acc'] = has_positive.mean()
        
        return metrics
    
    def training_step(self, batch, batch_idx):
        """Training step.
        batch.video.shape: [total_agents, T, C, H, W] e.g. [5, 20, 3, 224, 224]
        batch.agent_counts.shape: [B] e.g. [3, 2] (3+2=5 total agents)
        batch.pov_team_side_encoded.shape: [B] e.g. [2]
        """
        outputs = self.forward(batch)
        loss = outputs['loss']
        contrastive_loss = outputs['contrastive_loss']
        metrics = outputs['metrics']
        
        agent_counts = batch['agent_counts']
        batch_size = len(agent_counts)
        
        # Log total loss
        self.log('train/loss', loss, batch_size=batch_size,
                     on_step=True, on_epoch=True, prog_bar=True)
        
        # Log contrastive loss
        self.log('train/contrastive_loss', contrastive_loss, batch_size=batch_size,
                     on_step=True, on_epoch=True, prog_bar=True)
        
        # Log reconstruction loss if enabled
        if self.use_reconstruction:
            self.log('train/recon_loss_raw', metrics.get('recon_loss_raw', 0), 
                     batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=False)
            self.log('train/recon_loss_weighted', metrics.get('recon_loss_weighted', 0),
                     batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log contrastive metrics
        for name, value in metrics.items():
            if not name.startswith('recon_'):
                self.log(f'train/contrastive_{name}', value, batch_size=batch_size,
                             on_step=True, on_epoch=True, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        outputs = self.forward(batch)
        loss = outputs['loss']
        contrastive_loss = outputs['contrastive_loss']
        metrics = outputs['metrics']
        
        batch_size = len(batch['agent_counts'])
        
        # Log total loss
        self.log('val/loss', loss, batch_size=batch_size,
                     on_step=False, on_epoch=True, prog_bar=True)
        
        # Log contrastive loss
        self.log('val/contrastive_loss', contrastive_loss, batch_size=batch_size,
                     on_step=False, on_epoch=True, prog_bar=True)
        
        # Log reconstruction loss if enabled
        if self.use_reconstruction:
            self.log('val/recon_loss_raw', metrics.get('recon_loss_raw', 0),
                     batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)
            self.log('val/recon_mse', metrics.get('recon_mse', 0),
                     batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)
        
        # Log contrastive metrics
        for name, value in metrics.items():
            if not name.startswith('recon_'):
                self.log(f'val/contrastive_{name}', value, batch_size=batch_size,
                             on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        outputs = self.forward(batch)
        loss = outputs['loss']
        contrastive_loss = outputs['contrastive_loss']
        metrics = outputs['metrics']
        
        batch_size = len(batch['agent_counts'])
        checkpoint_name = getattr(self, 'checkpoint_name', 'last')
        prefix = f'test/{checkpoint_name}'
        
        # Log total loss
        self.log(f'{prefix}/loss', loss, batch_size=batch_size,
                     on_step=False, on_epoch=True, prog_bar=True)
        
        # Log contrastive loss
        self.log(f'{prefix}/contrastive_loss', contrastive_loss, batch_size=batch_size,
                     on_step=False, on_epoch=True, prog_bar=True)
        
        # Log reconstruction loss if enabled
        if self.use_reconstruction:
            self.log(f'{prefix}/recon_loss_raw', metrics.get('recon_loss_raw', 0),
                     batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f'{prefix}/recon_mse', metrics.get('recon_mse', 0),
                     batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)
        
        # Log contrastive metrics
        for name, value in metrics.items():
            if not name.startswith('recon_'):
                self.log(f'{prefix}/contrastive_{name}', value, batch_size=batch_size,
                             on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        opt_config = self.cfg.optimization
        
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=opt_config.lr,
            weight_decay=opt_config.weight_decay,
            fused=opt_config.fused_optimizer,
        )
        
        # Check if scheduler is configured
        if not hasattr(opt_config, 'scheduler') or opt_config.scheduler is None:
            return {'optimizer': optimizer}
        
        sched_config = opt_config.scheduler
        
        if sched_config.type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
            
            warmup_steps = sched_config.warmup_steps
            min_lr_ratio = sched_config.min_lr_ratio
            
            # Total steps from trainer (set by Lightning)
            # Use max_steps if specified, otherwise estimate from max_epochs
            total_steps = self.trainer.estimated_stepping_batches
            
            # Warmup scheduler: linear warmup from 0 to initial LR
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-8 / opt_config.lr,  # Start from near-zero
                end_factor=1.0,
                total_iters=warmup_steps
            )
            
            # Cosine decay scheduler: decay from initial LR to min_lr
            cosine_steps = max(total_steps - warmup_steps, 1)
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=opt_config.lr * min_lr_ratio
            )
            
            # Combine warmup and cosine decay
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            }
        
        return {'optimizer': optimizer}
    
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        Handle checkpoint loading, stripping torch.compile _orig_mod prefix.
        
        Called by Lightning before load_state_dict. We modify the checkpoint
        in-place to handle compiled model state dicts.
        """
        if 'state_dict' in checkpoint:
            checkpoint['state_dict'] = self._strip_orig_mod_prefix(checkpoint['state_dict'])
    
    def get_encoder_state_dict(self):
        """
        Get state dict of the encoder components for Stage 2.
        
        Returns dictionary containing:
            - video_encoder (if not using precomputed)
            - video_projector
            - contrastive parameters (logit_scale, logit_bias)
        """
        state = {
            'video_encoder': self.video_encoder.state_dict(),
            'video_projector': self.video_projector.state_dict(),
            'logit_scale': self.logit_scale,
            'logit_bias': self.logit_bias,
        }
        
        return state
    
    @classmethod
    def load_encoder_from_checkpoint(cls, checkpoint_path: str, cfg):
        """
        Load encoder components from a Stage 1 checkpoint.
        
        Args:
            checkpoint_path: Path to Stage 1 checkpoint
            cfg: Configuration for the model
            
        Returns:
            Tuple of (video_encoder, video_projector) with loaded weights
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['state_dict']
        
        # Handle torch.compile _orig_mod prefix in state dict keys
        state_dict = cls._strip_orig_mod_prefix(state_dict)
        
        # Create model to get architecture
        model = cls(cfg)
        
        # Load weights
        model.load_state_dict(state_dict)
        
        return model.video_encoder, model.video_projector
    
    @staticmethod
    def _strip_orig_mod_prefix(state_dict: dict) -> dict:
        """
        Strip '_orig_mod.' prefix from state dict keys.
        
        torch.compile wraps modules and adds '_orig_mod.' prefix to state dict keys.
        This method removes that prefix to allow loading compiled checkpoints into
        non-compiled models.
        
        Args:
            state_dict: State dict potentially containing '_orig_mod.' prefixed keys
            
        Returns:
            State dict with '_orig_mod.' prefix stripped from all keys
        """
        new_state_dict = {}
        for k, v in state_dict.items():
            # Replace all occurrences of '_orig_mod.' in the key
            new_key = k.replace("._orig_mod.", ".")
            # Also handle case where _orig_mod is at the start
            if new_key.startswith("_orig_mod."):
                new_key = new_key[len("_orig_mod."):]
            new_state_dict[new_key] = v
        return new_state_dict
