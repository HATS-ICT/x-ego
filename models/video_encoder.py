import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, SiglipModel, Dinov2Model, VivitModel, VideoMAEModel, VJEPA2Model
from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2AttentivePooler
from torch import Tensor
from typing import Dict, Any


# Mapping from model_type to HuggingFace pretrained model identifier
MODEL_TYPE_TO_PRETRAINED = {
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-base-patch16-224",
    "dinov2": "facebook/dinov2-base",
    "vivit": "google/vivit-b-16x2-kinetics400",
    "videomae": "MCG-NJU/videomae-base",
    "vjepa2": "facebook/vjepa2-vitl-fpc16-256-ssv2",
}


def temporal_sampling(frames: torch.Tensor, target_frames: int) -> torch.Tensor:
    """
    Temporal sampling of video frames to match target frame count.
    
    Args:
        frames: Input tensor of shape [batch_size, num_frames, channels, height, width]
        target_frames: Target number of frames
        
    Returns:
        Resampled frames of shape [batch_size, target_frames, channels, height, width]
    """
    batch_size, num_frames, channels, height, width = frames.shape
    
    if num_frames == target_frames:
        return frames
    
    # Reshape to [batch_size, channels, num_frames, height, width] for interpolation
    frames_reshaped = frames.permute(0, 2, 1, 3, 4)
    
    # Use 3D interpolation along the temporal dimension
    resampled = F.interpolate(
        frames_reshaped, 
        size=(target_frames, height, width), 
        mode='trilinear', 
        align_corners=False
    )
    
    # Reshape back to [batch_size, target_frames, channels, height, width]
    resampled = resampled.permute(0, 2, 1, 3, 4)
    
    return resampled


class VideoEncoderClip(nn.Module):
    """
    CLIP-based video encoder for video classification.
    
    Processes video frames through CLIP vision model and pools across time dimension.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        
        self.model_type = cfg.model_type
        self.from_pretrained = MODEL_TYPE_TO_PRETRAINED[self.model_type]
        
        self.vision_model = CLIPModel.from_pretrained(self.from_pretrained).vision_model
        
        self.embed_dim = self.vision_model.config.hidden_size
        
        if cfg.freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze vision model parameters."""
        for param in self.vision_model.parameters():
            param.requires_grad = False
    
    def forward(self, pixel_values: torch.Tensor, return_last_hidden_state: bool = False) -> Tensor:
        """
        Forward pass through the CLIP video encoder.
        
        Args:
            pixel_values: Video tensor of shape [batch_size, num_frames, channels, height, width]
            
        Returns:
            Video features - either pooled [batch_size, hidden_size] or unpooled [batch_size, seq_len, hidden_size]
        """
        batch_size, num_frames, channels, height, width = pixel_values.shape
        frames = pixel_values.view(-1, channels, height, width)
        
        vision_outputs = self.vision_model(pixel_values=frames)
        sequence_output = vision_outputs.last_hidden_state  # [batch_size * num_frames, seq_len, hidden_size]

        # Original pooled behavior
        # following https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L1170-L1251
        frame_features = torch.mean(sequence_output[:, 1:, :], dim=1)  # [batch_size * num_frames, hidden_size]
        frame_features = frame_features.view(batch_size, num_frames, -1)  # [batch_size, num_frames, hidden_size]
        video_features = torch.mean(frame_features, dim=1)  # [batch_size, hidden_size]
        return video_features


class VideoEncoderSigLIP(nn.Module):
    """
    SigLIP-based video encoder for video classification.
    
    Processes video frames through SigLIP vision model and pools across time dimension.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        
        self.model_type = cfg.model_type
        self.from_pretrained = MODEL_TYPE_TO_PRETRAINED[self.model_type]
        
        
        self.vision_model = SiglipModel.from_pretrained(self.from_pretrained).vision_model
        self.embed_dim = self.vision_model.config.hidden_size
        
        if cfg.freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze vision model parameters."""
        for param in self.vision_model.parameters():
            param.requires_grad = False
    
    def forward(self, pixel_values: torch.Tensor, return_last_hidden_state: bool = False) -> Tensor:
        """
        Forward pass through the SigLIP video encoder.
        
        Args:
            pixel_values: Video tensor of shape [batch_size, num_frames, channels, height, width]
            
        Returns:
            Video features with pooled video features
        """
        batch_size, num_frames, channels, height, width = pixel_values.shape
        
        frames = pixel_values.view(-1, channels, height, width)
        
        vision_outputs = self.vision_model(pixel_values=frames)
        sequence_output = vision_outputs.last_hidden_state  # [batch_size * num_frames, seq_len, hidden_size]
        
        # following https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L1007-L1109
        frame_features = torch.mean(sequence_output, dim=1)  # [batch_size * num_frames, hidden_size]
        frame_features = frame_features.view(batch_size, num_frames, -1)  # [batch_size, num_frames, hidden_size]
        video_features = torch.mean(frame_features, dim=1)  # [batch_size, hidden_size]
        
        return video_features


class VideoEncoderDinov2(nn.Module):
    """
    DINOv2-based video encoder for video classification.
    
    Processes video frames through DINOv2 vision model and pools across time dimension.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        
        self.model_type = cfg.model_type
        self.from_pretrained = MODEL_TYPE_TO_PRETRAINED[self.model_type]
        
        
        self.vision_model = Dinov2Model.from_pretrained(self.from_pretrained)
        self.embed_dim = self.vision_model.config.hidden_size * 2
        
        if cfg.freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze DINOv2 model parameters."""
        for param in self.vision_model.parameters():
            param.requires_grad = False
    
    def forward(self, pixel_values: torch.Tensor, return_last_hidden_state: bool = False) -> Tensor:
        """
        Forward pass through the DINOv2 video encoder.
        
        Args:
            pixel_values: Video tensor of shape [batch_size, num_frames, channels, height, width]
            
        Returns:
            Video features with pooled video features
        """
        batch_size, num_frames, channels, height, width = pixel_values.shape
        frames = pixel_values.view(-1, channels, height, width)
        outputs = self.vision_model(pixel_values=frames)
        
        # following https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov2/modeling_dinov2.py#L555-L603
        # get frame features
        sequence_output = outputs.last_hidden_state  # [batch_size * num_frames, seq_len, hidden_size]
        cls_tokens = sequence_output[:, 0]  # [batch_size * num_frames, hidden_size]
        patch_tokens = sequence_output[:, 1:]  # [batch_size * num_frames, num_patches, hidden_size]
        patch_features = torch.mean(patch_tokens, dim=1)  # [batch_size * num_frames, hidden_size]
        frame_features = torch.cat([cls_tokens, patch_features], dim=1)  # [batch_size * num_frames, hidden_size * 2]
        
        # pool to get video features
        frame_features = frame_features.view(batch_size, num_frames, -1)  # [batch_size, num_frames, hidden_size * 2]
        video_features = torch.mean(frame_features, dim=1)  # [batch_size, hidden_size * 2]
        
        return video_features


class VideoEncoderVivit(nn.Module):
    """
    ViViT-based video encoder for video classification.
    
    Processes videos natively through ViViT model without temporal pooling.
    ViViT is designed to handle video sequences directly.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        
        self.model_type = cfg.model_type
        self.from_pretrained = MODEL_TYPE_TO_PRETRAINED[self.model_type]
        
        
        self.vision_model = VivitModel.from_pretrained(self.from_pretrained)
        
        self.expected_num_frames = 32
        self.embed_dim = self.vision_model.config.hidden_size
        
        if cfg.freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze ViViT model parameters."""
        for param in self.vision_model.parameters():
            param.requires_grad = False
    
    def forward(self, pixel_values: torch.Tensor, return_last_hidden_state: bool = False) -> Tensor:
        """
        Forward pass through the ViViT video encoder.
        
        Args:
            pixel_values: Video tensor of shape [batch_size, num_frames, channels, height, width]
            
        Returns:
            Video features from CLS token
        """
        # ViViT expects exactly 32 frames, so we need to resample if necessary
        pixel_values = temporal_sampling(pixel_values, self.expected_num_frames)
        
        outputs = self.vision_model(pixel_values=pixel_values)
        sequence_output = outputs.last_hidden_state  # [batch_size, sequence_length, hidden_size]
        
        # following https://github.com/huggingface/transformers/blob/bb45d3631ec7026db04a77d33a52b31766372160/src/transformers/models/vivit/modeling_vivit.py#L566-L687
        video_features = sequence_output[:, 0, :]  # [batch_size, hidden_size]
        
        return video_features


class VideoEncoderVideoMAE(nn.Module):
    """
    VideoMAE-based video encoder for video classification.
    
    Processes videos natively through VideoMAE model without temporal pooling.
    VideoMAE is designed to handle video sequences directly with masked autoencoding pretraining.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        
        self.model_type = cfg.model_type
        self.from_pretrained = MODEL_TYPE_TO_PRETRAINED[self.model_type]
        
        
        self.vision_model = VideoMAEModel.from_pretrained(self.from_pretrained)
        
        self.expected_num_frames = 16
        self.embed_dim = self.vision_model.config.hidden_size
        self.use_mean_pooling = self.vision_model.config.use_mean_pooling
        
        # Add layer norm if using mean pooling (following the classification head pattern)
        if self.use_mean_pooling:
            self.fc_norm = nn.LayerNorm(self.embed_dim)
        else:
            self.fc_norm = None
        
        if cfg.freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze VideoMAE model parameters."""
        for param in self.vision_model.parameters():
            param.requires_grad = False
    
    def forward(self, pixel_values: torch.Tensor, return_last_hidden_state: bool = False) -> Tensor:
        """
        Forward pass through the VideoMAE video encoder.
        
        Args:
            pixel_values: Video tensor of shape [batch_size, num_frames, channels, height, width]
            
        Returns:
            Video features using either mean pooling or CLS token
        """
        # VideoMAE expects exactly 16 frames, so we need to resample if necessary
        pixel_values = temporal_sampling(pixel_values, self.expected_num_frames)
        
        outputs = self.vision_model(pixel_values=pixel_values)
        sequence_output = outputs.last_hidden_state  # [batch_size, sequence_length, hidden_size]
        
        if self.use_mean_pooling:
            # Use mean pooling over all tokens (following VideoMAE classification head)
            video_features = sequence_output.mean(1)  # [batch_size, hidden_size]
            if self.fc_norm is not None:
                video_features = self.fc_norm(video_features)
        else:
            # Use CLS token (first token) as video representation
            video_features = sequence_output[:, 0]  # [batch_size, hidden_size]
        
        return video_features


class VideoEncoderVJEPA2(nn.Module):
    """
    VJEPA2-based video encoder for video classification.
    
    Processes videos natively through VJEPA2 model with attentive pooling.
    VJEPA2 is designed to handle video sequences directly with joint embedding
    predictive architecture.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        
        self.model_type = cfg.model_type
        self.from_pretrained = MODEL_TYPE_TO_PRETRAINED[self.model_type]
        
        
        self.vision_model = VJEPA2Model.from_pretrained(self.from_pretrained)
        self.pooler = VJEPA2AttentivePooler(self.vision_model.config)
        
        self.embed_dim = self.vision_model.config.hidden_size
        
        if cfg.freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze VJEPA2 model parameters."""
        for param in self.vision_model.parameters():
            param.requires_grad = False
    
    def forward(self, pixel_values: torch.Tensor, return_last_hidden_state: bool = False) -> Tensor:
        """
        Forward pass through the VJEPA2 video encoder.
        
        Args:
            pixel_values: Video tensor of shape [batch_size, num_frames, channels, height, width]
            
        Returns:
            Video features using attentive pooling
        """
        # VJEPA2 expects input of shape [batch_size, num_frames, channels, height, width]
        # which matches our input format
        
        outputs = self.vision_model(
            pixel_values_videos=pixel_values,
            skip_predictor=True,  # Skip predictor for classification tasks
            output_attentions=False,
            output_hidden_states=False,
        )
        
        last_hidden_state = outputs.last_hidden_state  # [batch_size, sequence_length, hidden_size]
        
        # Use attentive pooling to get video-level features
        # following https://github.com/huggingface/transformers/blob/bb45d3631ec7026db04a77d33a52b31766372160/src/transformers/models/vjepa2/modeling_vjepa2.py#L1137-1214
        video_features = self.pooler(last_hidden_state)  # [batch_size, hidden_size]
        return video_features


class VideoEncoder(nn.Module):
    """
    Factory class for video encoders that automatically selects the appropriate encoder
    based on the model_type parameter.
    
    Supports:
    - CLIP models (model_type="clip")
    - SigLIP models (model_type="siglip")
    - DINOv2 models (model_type="dinov2")
    - ViViT models (model_type="vivit")
    - VideoMAE models (model_type="videomae")
    - VJEPA2 models (model_type="vjepa2")
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        
        model_type = cfg.model_type
        
        if model_type == 'clip':
            self.video_encoder = VideoEncoderClip(cfg)
        elif model_type == 'siglip':
            self.video_encoder = VideoEncoderSigLIP(cfg)
        elif model_type == 'dinov2':
            self.video_encoder = VideoEncoderDinov2(cfg)
        elif model_type == 'vivit':
            self.video_encoder = VideoEncoderVivit(cfg)
        elif model_type == 'videomae':
            self.video_encoder = VideoEncoderVideoMAE(cfg)
        elif model_type == 'vjepa2':
            self.video_encoder = VideoEncoderVJEPA2(cfg)
        else:
            raise ValueError(
                f"Unknown model_type: '{model_type}'. "
                f"Supported types: {list(MODEL_TYPE_TO_PRETRAINED.keys())}"
            )
        
        self.embed_dim = self.video_encoder.embed_dim
    
    def forward(self, pixel_values: torch.Tensor, return_last_hidden_state: bool = False):
        """Forward pass through the selected video encoder."""
        return self.video_encoder.forward(pixel_values, return_last_hidden_state)
    
    def get_embeddings(self, pixel_values: torch.Tensor) -> Tensor:
        """Get video embeddings (alias for forward method)."""
        return self.forward(pixel_values)
