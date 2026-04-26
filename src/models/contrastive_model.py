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
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from src.models.modules.video_encoder import VideoEncoder
from src.models.modules.architecture_utils import build_mlp
from src.models.modules.optimizer_utils import build_optimizer


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
        self.use_contrastive = cfg.model.contrastive.enable
        self.contrastive_loss_weight = cfg.model.contrastive.loss_weight
        self.turn_off_bias = cfg.model.contrastive.turn_off_bias
        self.logit_scale = nn.Parameter(
            torch.tensor(cfg.model.contrastive.logit_scale_init, dtype=torch.float32).log(),
            requires_grad=True
        )
        self.logit_bias = nn.Parameter(
            torch.tensor(cfg.model.contrastive.logit_bias_init, dtype=torch.float32),
            requires_grad=True
        )
        
        # Store dimensions
        self.video_embed_dim = video_embed_dim
        self.proj_dim = proj_dim

        if not self.use_contrastive:
            raise ValueError("model.contrastive.enable must be true for contrastive training")

        self.contrastive_accumulate_batches = int(cfg.training.contrastive_accumulate_batches)
        self.automatic_optimization = self.contrastive_accumulate_batches == 1
        self._embedding_cache = []
        
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
    
    def _compute_projected_embeddings(self, batch):
        """Encode and project one physical microbatch."""
        video = batch['video']  # [total_agents, T, C, H, W]

        if len(video.shape) != 5:
            raise ValueError(f"Expected video shape [total_agents, T, C, H, W], got {video.shape}")

        embeddings = self.video_encoder(video)  # [total_agents, embed_dim]
        return self.video_projector(embeddings)  # [total_agents, proj_dim]

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
                - loss: Contrastive loss
                - metrics: Dictionary of metrics
        """
        agent_counts = batch['agent_counts']  # [B]
        projected = self._compute_projected_embeddings(batch)
        
        # Create alignment labels
        labels = self.create_alignment_matrix(agent_counts, projected.device)
        
        metrics = {}
        contrastive_loss = projected.new_zeros(())
        total_loss = projected.new_zeros(())

        if self.use_contrastive:
            contrastive_loss, metrics = self.compute_contrastive_loss(projected, labels)
            total_loss = total_loss + self.contrastive_loss_weight * contrastive_loss
        
        return {
            'embeddings': projected,
            'loss': total_loss,
            'contrastive_loss': contrastive_loss,
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

    @staticmethod
    def _capture_rng_state() -> dict:
        state = {'cpu': torch.get_rng_state()}
        if torch.cuda.is_available():
            state['cuda'] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: dict) -> None:
        torch.set_rng_state(state['cpu'])
        if 'cuda' in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['cuda'])

    def _is_last_training_batch(self, batch_idx: int) -> bool:
        num_batches = self.trainer.num_training_batches
        return isinstance(num_batches, int) and batch_idx + 1 >= num_batches

    def _cache_training_microbatch(self, batch) -> None:
        rng_state = self._capture_rng_state()
        with torch.no_grad():
            projected = self._compute_projected_embeddings(batch).detach()

        self._embedding_cache.append({
            'batch': batch,
            'projected': projected,
            'agent_counts': batch['agent_counts'].detach().clone(),
            'rng_state': rng_state,
        })

    def _flush_embedding_cache(self):
        """Compute one large contrastive loss, then replay microbatches for encoder grads."""
        if not self._embedding_cache:
            return None

        optimizer = self.optimizers()
        optimizer.zero_grad()

        projected_chunks = [entry['projected'] for entry in self._embedding_cache]
        cached_projected = torch.cat(projected_chunks, dim=0).detach().requires_grad_(True)
        agent_counts = torch.cat(
            [entry['agent_counts'].to(cached_projected.device) for entry in self._embedding_cache],
            dim=0,
        )
        labels = self.create_alignment_matrix(agent_counts, cached_projected.device)

        contrastive_loss, metrics = self.compute_contrastive_loss(cached_projected, labels)
        loss = self.contrastive_loss_weight * contrastive_loss
        self.manual_backward(loss)

        projected_grads = cached_projected.grad.detach()
        start = 0
        for entry in self._embedding_cache:
            end = start + entry['projected'].shape[0]
            grad_chunk = projected_grads[start:end]
            start = end

            self._restore_rng_state(entry['rng_state'])
            projected = self._compute_projected_embeddings(entry['batch'])
            self.manual_backward(torch.sum(projected * grad_chunk))

        clip_val = self._manual_gradient_clip_val()
        if clip_val is not None and clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm="norm",
            )

        optimizer.step()
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()

        batch_size = int(agent_counts.numel())
        self._log_training_outputs(loss.detach(), contrastive_loss.detach(), metrics, batch_size)
        self._embedding_cache.clear()
        return loss.detach()

    def _manual_gradient_clip_val(self):
        optimizer_name = getattr(self.cfg.optimization, "optimizer", "adamw")
        fused_adamw = optimizer_name == "adamw" and self.cfg.optimization.fused_optimizer
        if fused_adamw:
            return None
        return self.cfg.training.gradient_clip_val

    def _log_training_outputs(self, loss, contrastive_loss, metrics, batch_size: int) -> None:
        self.log('train/loss', loss, batch_size=batch_size,
                     on_step=True, on_epoch=True, prog_bar=True)

        self.log('train/contrastive_loss', contrastive_loss, batch_size=batch_size,
                     on_step=True, on_epoch=True, prog_bar=True)

        for name, value in metrics.items():
            self.log(f'train/contrastive_{name}', value, batch_size=batch_size,
                         on_step=True, on_epoch=True, prog_bar=False)
    
    def training_step(self, batch, batch_idx):
        """Training step.
        batch.video.shape: [total_agents, T, C, H, W] e.g. [5, 20, 3, 224, 224]
        batch.agent_counts.shape: [B] e.g. [3, 2] (3+2=5 total agents)
        batch.pov_team_side_encoded.shape: [B] e.g. [2]
        """
        if self.contrastive_accumulate_batches > 1:
            self._cache_training_microbatch(batch)
            should_flush = (
                len(self._embedding_cache) >= self.contrastive_accumulate_batches
                or self._is_last_training_batch(batch_idx)
            )
            if should_flush:
                return self._flush_embedding_cache()
            return None

        outputs = self.forward(batch)
        loss = outputs['loss']
        contrastive_loss = outputs['contrastive_loss']
        metrics = outputs['metrics']
        
        agent_counts = batch['agent_counts']
        batch_size = len(agent_counts)
        self._log_training_outputs(loss, contrastive_loss, metrics, batch_size)
        
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
        
        # Log contrastive metrics
        for name, value in metrics.items():
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
        checkpoint_name = self.checkpoint_name
        prefix = f'test/{checkpoint_name}'
        
        # Log total loss
        self.log(f'{prefix}/loss', loss, batch_size=batch_size,
                     on_step=False, on_epoch=True, prog_bar=True)
        
        # Log contrastive loss
        self.log(f'{prefix}/contrastive_loss', contrastive_loss, batch_size=batch_size,
                     on_step=False, on_epoch=True, prog_bar=True)
        
        # Log contrastive metrics
        for name, value in metrics.items():
            self.log(f'{prefix}/contrastive_{name}', value, batch_size=batch_size,
                         on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        opt_config = self.cfg.optimization
        
        optimizer = build_optimizer(self, opt_config)
        
        # Check if scheduler is configured.
        if opt_config.scheduler is None:
            return {'optimizer': optimizer}
        
        sched_config = opt_config.scheduler
        
        if sched_config.type == 'cosine':
            import math
            from torch.optim.lr_scheduler import LambdaLR
            
            warmup_steps = sched_config.warmup_steps
            min_lr_ratio = sched_config.min_lr_ratio
            
            # Total steps from trainer (set by Lightning)
            # Use max_steps if specified, otherwise estimate from max_epochs
            total_steps = self.trainer.estimated_stepping_batches
            if self.contrastive_accumulate_batches > 1:
                total_steps = math.ceil(total_steps / self.contrastive_accumulate_batches)
            cosine_steps = max(total_steps - warmup_steps, 1)

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return max(step, 1) / max(warmup_steps, 1)

                progress = min((step - warmup_steps) / cosine_steps, 1.0)
                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor

            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            
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
