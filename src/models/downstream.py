"""
Linear Probing Model (Stage 2)

Simple linear classifiers/regressors for downstream tasks.
Uses frozen video encoder - either off-the-shelf from HuggingFace or pretrained from Stage 1 contrastive.

Supports task types (ml_form):
- binary_cls: Binary classification (BCE loss)
- multi_cls: Multi-class classification (CrossEntropy loss)
- multi_label_cls: Multi-label classification (BCE loss)
- regression: Regression (MSE loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW
import torch._dynamo
from torchmetrics.classification import (
    MultilabelAccuracy, MultilabelF1Score, MultilabelAUROC,
    MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC,
    BinaryAccuracy, BinaryF1Score, BinaryAUROC
)
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score

try:
    from modules.video_encoder import VideoEncoder
except ImportError:
    from .modules.video_encoder import VideoEncoder


class LinearProbeHead(nn.Module):
    """
    Linear probe head for a specific task type.
    
    Supports:
    - binary_cls: Single output with sigmoid
    - multi_cls: num_classes outputs with softmax
    - multi_label_cls: num_classes outputs with sigmoid per class
    - regression: output_dim outputs (no activation)
    """
    
    def __init__(self, input_dim: int, ml_form: str, output_dim: int, num_classes: int = None):
        super().__init__()
        
        self.ml_form = ml_form
        self.output_dim = output_dim
        self.num_classes = num_classes
        
        # Determine actual output size
        if ml_form == 'binary_cls':
            self.linear = nn.Linear(input_dim, 1)
        elif ml_form == 'multi_cls':
            self.linear = nn.Linear(input_dim, num_classes)
        elif ml_form == 'multi_label_cls':
            self.linear = nn.Linear(input_dim, num_classes)
        elif ml_form == 'regression':
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            raise ValueError(f"Unknown ml_form: {ml_form}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits (no activation applied)."""
        return self.linear(x)


class LinearProbeModel(L.LightningModule):
    """
    Linear probing model for Stage 2 downstream tasks.
    
    Architecture:
        Video [B, T, C, H, W] -> [Frozen VideoEncoder] -> Linear Head -> Prediction
    
    The encoder is either:
    1. Off-the-shelf HuggingFace model (baseline)
    2. Pretrained from Stage 1 contrastive learning
    
    Only the linear head is trained; the encoder is always frozen.
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        # Task configuration
        self.task_id = cfg.task.task_id
        self.ml_form = cfg.task.ml_form
        self.num_classes = cfg.task.num_classes
        self.output_dim = cfg.task.output_dim
        
        # Initialize video encoder
        self.video_encoder = VideoEncoder(cfg.model.encoder.video)
        video_embed_dim = self.video_encoder.embed_dim
        print(f"[LinearProbe] Video encoder: {cfg.model.encoder.video.model_type} (dim: {video_embed_dim})")
        
        # Load Stage 1 weights if provided (before freezing)
        if cfg.model.stage1_checkpoint:
            self._load_stage1_encoder(cfg.model.stage1_checkpoint)
        else:
            print("[LinearProbe] Using off-the-shelf HuggingFace encoder (baseline)")
        
        # Freeze encoder - always frozen for linear probing
        self._freeze_encoder()
        
        # Linear probe head directly on encoder output
        self.head = LinearProbeHead(
            input_dim=video_embed_dim,
            ml_form=self.ml_form,
            output_dim=self.output_dim,
            num_classes=self.num_classes
        )
        
        # Initialize metrics
        self._init_metrics()
        
        self.output_dir = cfg.path.exp
    
    def _load_stage1_encoder(self, checkpoint_path: str):
        """Load video encoder weights from Stage 1 contrastive checkpoint."""
        print(f"[LinearProbe] Loading Stage 1 encoder from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        # Extract video_encoder weights (keys start with 'video_encoder.')
        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith('video_encoder.'):
                # Remove 'video_encoder.' prefix
                new_key = k.replace('video_encoder.', '')
                encoder_state[new_key] = v
        
        if encoder_state:
            self.video_encoder.load_state_dict(encoder_state)
            print(f"[LinearProbe] Loaded {len(encoder_state)} encoder parameters from Stage 1")
        else:
            print("[LinearProbe] WARNING: No video_encoder weights found in checkpoint!")
    
    def _freeze_encoder(self):
        """Freeze video encoder parameters."""
        for param in self.video_encoder.parameters():
            param.requires_grad = False
        print("[LinearProbe] Video encoder frozen")
    
    def _init_metrics(self):
        """Initialize metrics based on task type."""
        if self.ml_form == 'binary_cls':
            for split in ['train', 'val', 'test']:
                setattr(self, f'{split}_acc', BinaryAccuracy())
                setattr(self, f'{split}_f1', BinaryF1Score())
                setattr(self, f'{split}_auroc', BinaryAUROC())
        
        elif self.ml_form == 'multi_cls':
            for split in ['train', 'val', 'test']:
                setattr(self, f'{split}_acc', MulticlassAccuracy(num_classes=self.num_classes))
                setattr(self, f'{split}_f1', MulticlassF1Score(num_classes=self.num_classes, average='macro'))
        
        elif self.ml_form == 'multi_label_cls':
            for split in ['train', 'val', 'test']:
                setattr(self, f'{split}_acc', MultilabelAccuracy(num_labels=self.num_classes))
                setattr(self, f'{split}_f1', MultilabelF1Score(num_labels=self.num_classes, average='macro'))
        
        elif self.ml_form == 'regression':
            for split in ['train', 'val', 'test']:
                setattr(self, f'{split}_mse', MeanSquaredError())
                setattr(self, f'{split}_mae', MeanAbsoluteError())
                setattr(self, f'{split}_r2', R2Score())
    
    def forward(self, batch):
        """
        Forward pass.
        
        Args:
            batch: Dictionary with 'video' key containing raw video [B, T, C, H, W]
            
        Returns:
            logits: Task predictions
        """
        video = batch['video']  # [B, T, C, H, W]
        
        # Encode video through frozen encoder
        with torch.no_grad():
            video_embedding = self.video_encoder(video)  # [B, embed_dim]
        
        # Linear head
        logits = self.head(video_embedding)
        
        return logits
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss based on task type."""
        if self.ml_form == 'binary_cls':
            return F.binary_cross_entropy_with_logits(logits.squeeze(-1), targets.float())
        
        elif self.ml_form == 'multi_cls':
            return F.cross_entropy(logits, targets.long())
        
        elif self.ml_form == 'multi_label_cls':
            return F.binary_cross_entropy_with_logits(logits, targets.float())
        
        elif self.ml_form == 'regression':
            return F.mse_loss(logits, targets.float())
        
        else:
            raise ValueError(f"Unknown ml_form: {self.ml_form}")
    
    def compute_predictions(self, logits: torch.Tensor):
        """Convert logits to predictions based on task type."""
        if self.ml_form == 'binary_cls':
            return torch.sigmoid(logits.squeeze(-1))
        
        elif self.ml_form == 'multi_cls':
            return logits  # Return logits for metrics (they handle softmax internally)
        
        elif self.ml_form == 'multi_label_cls':
            return torch.sigmoid(logits)
        
        elif self.ml_form == 'regression':
            return logits
    
    @torch._dynamo.disable
    def safe_log(self, *args, **kwargs):
        return self.log(*args, **kwargs)
    
    def _step(self, batch, split: str):
        """Common step for train/val/test."""
        logits = self.forward(batch)
        targets = batch['label']
        
        loss = self.compute_loss(logits, targets)
        preds = self.compute_predictions(logits)
        
        batch_size = targets.shape[0]
        
        # Log loss
        self.safe_log(f'{split}/loss', loss, batch_size=batch_size,
                     on_step=(split == 'train'), on_epoch=True, prog_bar=True)
        
        # Update and log metrics
        if self.ml_form == 'binary_cls':
            acc = getattr(self, f'{split}_acc')
            f1 = getattr(self, f'{split}_f1')
            auroc = getattr(self, f'{split}_auroc')
            
            acc(preds, targets.int())
            f1(preds, targets.int())
            auroc(preds, targets.int())
            
            self.safe_log(f'{split}/acc', acc, batch_size=batch_size, on_epoch=True)
            self.safe_log(f'{split}/f1', f1, batch_size=batch_size, on_epoch=True)
            self.safe_log(f'{split}/auroc', auroc, batch_size=batch_size, on_epoch=True)
        
        elif self.ml_form == 'multi_cls':
            acc = getattr(self, f'{split}_acc')
            f1 = getattr(self, f'{split}_f1')
            
            acc(preds, targets.long())
            f1(preds, targets.long())
            
            self.safe_log(f'{split}/acc', acc, batch_size=batch_size, on_epoch=True)
            self.safe_log(f'{split}/f1', f1, batch_size=batch_size, on_epoch=True)
        
        elif self.ml_form == 'multi_label_cls':
            acc = getattr(self, f'{split}_acc')
            f1 = getattr(self, f'{split}_f1')
            
            acc(preds, targets.int())
            f1(preds, targets.int())
            
            self.safe_log(f'{split}/acc', acc, batch_size=batch_size, on_epoch=True)
            self.safe_log(f'{split}/f1', f1, batch_size=batch_size, on_epoch=True)
        
        elif self.ml_form == 'regression':
            mse = getattr(self, f'{split}_mse')
            mae = getattr(self, f'{split}_mae')
            r2 = getattr(self, f'{split}_r2')
            
            mse(preds, targets.float())
            mae(preds, targets.float())
            r2(preds, targets.float())
            
            self.safe_log(f'{split}/mse', mse, batch_size=batch_size, on_epoch=True)
            self.safe_log(f'{split}/mae', mae, batch_size=batch_size, on_epoch=True)
            self.safe_log(f'{split}/r2', r2, batch_size=batch_size, on_epoch=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._step(batch, 'test')
    
    def configure_optimizers(self):
        """Configure optimizer - only train unfrozen parameters (linear head)."""
        opt_config = self.cfg.optimization
        
        # Only train parameters that require gradients (the linear head)
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        
        optimizer = AdamW(
            trainable_params,
            lr=opt_config.lr,
            weight_decay=opt_config.weight_decay,
            fused=opt_config.fused_optimizer,
        )
        
        return {'optimizer': optimizer}


def get_metrics_for_task(ml_form: str, num_classes: int = None, num_labels: int = None):
    """
    Get appropriate metrics for a task type.
    
    Returns dict of metric name -> metric instance.
    """
    if ml_form == 'binary_cls':
        return {
            'acc': BinaryAccuracy(),
            'f1': BinaryF1Score(),
            'auroc': BinaryAUROC(),
        }
    
    elif ml_form == 'multi_cls':
        return {
            'acc': MulticlassAccuracy(num_classes=num_classes),
            'f1_macro': MulticlassF1Score(num_classes=num_classes, average='macro'),
            'f1_micro': MulticlassF1Score(num_classes=num_classes, average='micro'),
        }
    
    elif ml_form == 'multi_label_cls':
        return {
            'acc': MultilabelAccuracy(num_labels=num_labels),
            'f1_macro': MultilabelF1Score(num_labels=num_labels, average='macro'),
            'f1_micro': MultilabelF1Score(num_labels=num_labels, average='micro'),
        }
    
    elif ml_form == 'regression':
        return {
            'mse': MeanSquaredError(),
            'mae': MeanAbsoluteError(),
            'r2': R2Score(),
        }
    
    else:
        raise ValueError(f"Unknown ml_form: {ml_form}")
