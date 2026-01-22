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
"""

import torch
import lightning as L
from torch.optim import AdamW
import torch._dynamo

# Configure dynamo to capture scalar outputs to avoid graph breaks from .item() calls
torch._dynamo.config.capture_scalar_outputs = True

from src.models.modules.video_encoder import VideoEncoder
from src.models.modules.cross_ego_contrastive import CrossEgoContrastive
from src.models.modules.architecture_utils import build_mlp


class ContrastiveModel(L.LightningModule):
    """
    Contrastive learning model for team alignment (Stage 1).
    
    Architecture:
        Video Encoder -> Video Projector -> Contrastive Module
    
    The model learns to align agent embeddings from the same team/batch
    while pushing apart embeddings from different teams/batches.
    
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
        self.video_encoder = VideoEncoder(cfg.model.encoder.video)
        video_embed_dim = self.video_encoder.embed_dim
        print(f"[Model Init] Video encoder: {cfg.model.encoder.video.model_type} (dim: {video_embed_dim})")
        
        # Video projector
        proj_dim = cfg.model.encoder.proj_dim
        self.video_projector = build_mlp(
            input_dim=video_embed_dim,
            output_dim=proj_dim,
            num_hidden_layers=1,
            hidden_dim=proj_dim,
            dropout=cfg.model.dropout,
            activation=cfg.model.activation
        )
        
        # Contrastive module
        self.contrastive = CrossEgoContrastive(
            embed_dim=proj_dim,
            init_logit_scale=cfg.model.contrastive.logit_scale_init,
            init_logit_bias=cfg.model.contrastive.logit_bias_init,
            learnable_temp=True,
            turn_off_bias=cfg.model.contrastive.turn_off_bias
        )
        
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
        Forward pass for contrastive learning.
        
        Args:
            batch: Dictionary containing:
                - video: [total_agents, T, C, H, W] concatenated video tensors
                - agent_counts: [B] number of agents per sample
                
        Returns:
            Dictionary containing:
                - embeddings: [total_agents, proj_dim] projected embeddings
                - loss: Contrastive loss scalar
                - metrics: Dictionary of contrastive metrics
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
        loss, metrics = self.compute_contrastive_loss(projected, labels)
        
        return {
            'embeddings': projected,
            'loss': loss,
            'metrics': metrics,
            'labels': labels,
        }
    
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
        import torch.nn.functional as F
        
        # L2 normalize embeddings
        normalized = F.normalize(embeddings, p=2, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(normalized, normalized.t())
        
        # Apply learnable temperature and bias
        logit_scale = self.contrastive.logit_scale.exp()
        logits = logits * logit_scale
        if not self.contrastive.turn_off_bias:
            logits = logits + self.contrastive.logit_bias
        
        # Compute sigmoid loss
        m1_diag1 = 2 * labels - 1  # Convert {0,1} to {-1,1}
        loglik = F.logsigmoid(m1_diag1 * logits)
        nll = -torch.sum(loglik, dim=-1)
        loss = nll.mean()
        
        # Compute metrics
        metrics = self._compute_metrics(logits, labels, logit_scale)
        
        return loss, metrics
    
    def _compute_metrics(self, logits: torch.Tensor, labels: torch.Tensor, 
                         temperature: torch.Tensor) -> dict:
        """Compute contrastive learning metrics."""
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
        
        if not self.contrastive.turn_off_bias:
            metrics['bias'] = self.contrastive.logit_bias.item()
        
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
    
    @torch._dynamo.disable
    def safe_log(self, *args, **kwargs):
        """Safe logging that disables dynamo compilation."""
        return self.log(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        """Training step.
        batch.video.shape: [total_agents, T, C, H, W] e.g. [5, 20, 3, 224, 224]
        batch.agent_counts.shape: [B] e.g. [3, 2] (3+2=5 total agents)
        batch.pov_team_side_encoded.shape: [B] e.g. [2]
        """
        outputs = self.forward(batch)
        loss = outputs['loss']
        metrics = outputs['metrics']
        
        agent_counts = batch['agent_counts']
        batch_size = len(agent_counts)
        total_agents = agent_counts.sum().item()
        
        # Log loss
        self.safe_log('train/contrastive_loss', loss, batch_size=batch_size,
                     on_step=True, on_epoch=True, prog_bar=True)
        self.safe_log('train/loss', loss, batch_size=batch_size,
                     on_step=True, on_epoch=True, prog_bar=True)
        
        # Log metrics
        for name, value in metrics.items():
            self.safe_log(f'train/contrastive_{name}', value, batch_size=batch_size,
                         on_step=True, on_epoch=True, prog_bar=False)
        
        # Log agent statistics
        self.safe_log('train/total_agents', float(total_agents), batch_size=batch_size,
                     on_step=True, on_epoch=False, prog_bar=False)
        self.safe_log('train/avg_agents_per_sample', float(total_agents) / batch_size,
                     batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        outputs = self.forward(batch)
        loss = outputs['loss']
        metrics = outputs['metrics']
        
        batch_size = len(batch['agent_counts'])
        
        # Log loss
        self.safe_log('val/contrastive_loss', loss, batch_size=batch_size,
                     on_step=False, on_epoch=True, prog_bar=True)
        self.safe_log('val/loss', loss, batch_size=batch_size,
                     on_step=False, on_epoch=True, prog_bar=True)
        
        # Log metrics
        for name, value in metrics.items():
            self.safe_log(f'val/contrastive_{name}', value, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        outputs = self.forward(batch)
        loss = outputs['loss']
        metrics = outputs['metrics']
        
        batch_size = len(batch['agent_counts'])
        checkpoint_name = getattr(self, 'checkpoint_name', 'last')
        prefix = f'test/{checkpoint_name}'
        
        # Log loss
        self.safe_log(f'{prefix}/contrastive_loss', loss, batch_size=batch_size,
                     on_step=False, on_epoch=True, prog_bar=True)
        
        # Log metrics
        for name, value in metrics.items():
            self.safe_log(f'{prefix}/contrastive_{name}', value, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer."""
        opt_config = self.cfg.optimization
        
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=opt_config.lr,
            weight_decay=opt_config.weight_decay,
            fused=opt_config.fused_optimizer,
        )
        
        return {'optimizer': optimizer}
    
    def get_encoder_state_dict(self):
        """
        Get state dict of the encoder components for Stage 2.
        
        Returns dictionary containing:
            - video_encoder (if not using precomputed)
            - video_projector
            - contrastive module
        """
        state = {
            'video_encoder': self.video_encoder.state_dict(),
            'video_projector': self.video_projector.state_dict(),
            'contrastive': self.contrastive.state_dict(),
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
            Tuple of (video_encoder, video_projector, contrastive)
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['state_dict']
        
        # Create model to get architecture
        model = cls(cfg)
        
        # Load weights
        model.load_state_dict(state_dict)
        
        return model.video_encoder, model.video_projector, model.contrastive
