import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2AttentivePooler
from torch import Tensor
from typing import Dict, Any, Tuple
import math

from src.models.modules.architecture_utils import TemporalTransformer


# Mapping from short alias to HuggingFace pretrained model identifier
# 'resnet50' is torchvision-only and uses no pretrained weights (trained from scratch)
MODEL_TYPE_TO_PRETRAINED = {
    "clip": "openai/clip-vit-base-patch32",
    "siglip2": "google/siglip2-base-patch16-224",
    "dinov2": "facebook/dinov2-base",
    "dinov3": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vivit": "google/vivit-b-16x2-kinetics400",
    "videomae": "MCG-NJU/videomae-base",
    "vjepa2": "facebook/vjepa2-vitl-fpc16-256-ssv2",
    "resnet50": "torchvision/resnet50",  # no pretrained weights — trains from scratch
}

# Reverse mapping: full HF name → short alias
# Allows configs to specify either "siglip2" or "google/siglip2-base-patch16-224"
PRETRAINED_TO_MODEL_TYPE = {v: k for k, v in MODEL_TYPE_TO_PRETRAINED.items()}


def normalize_model_type(model_type: str) -> str:
    """
    Accept either a short alias ('siglip2') or a full HF name
    ('google/siglip2-base-patch16-224') and return the canonical short alias.
    """
    if model_type in MODEL_TYPE_TO_PRETRAINED:
        return model_type  # already a short alias
    if model_type in PRETRAINED_TO_MODEL_TYPE:
        return PRETRAINED_TO_MODEL_TYPE[model_type]
    raise ValueError(
        f"Unknown model_type: '{model_type}'. "
        f"Supported aliases: {list(MODEL_TYPE_TO_PRETRAINED.keys())} "
        f"or full HF names: {list(PRETRAINED_TO_MODEL_TYPE.keys())}"
    )



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


def configure_finetuning(model: nn.Module, encoder_layers: nn.ModuleList, 
                         finetune_last_k_layers: int, final_norm: nn.Module = None):
    """
    Configure which transformer layers to finetune.
    
    Args:
        model: The full model to freeze/unfreeze
        encoder_layers: The ModuleList of transformer layers
        finetune_last_k_layers: Number of layers to finetune from the end.
            -1: Finetune all layers (no freezing)
             0: Freeze all layers
             k: Finetune last k transformer layers, freeze the rest
        final_norm: Optional final layer norm to unfreeze
    """
    if finetune_last_k_layers == -1:
        # Finetune all - nothing to freeze
        return
    
    # Freeze all first
    for param in model.parameters():
        param.requires_grad = False
    
    if finetune_last_k_layers == 0:
        return
    
    # Unfreeze last k transformer layers
    num_layers = len(encoder_layers)
    layers_to_unfreeze = min(finetune_last_k_layers, num_layers)
    
    for layer in encoder_layers[-layers_to_unfreeze:]:
        for param in layer.parameters():
            param.requires_grad = True
    
    # Also unfreeze the final layer norm if provided
    if final_norm is not None:
        for param in final_norm.parameters():
            param.requires_grad = True


class VideoEncoderClip(nn.Module):
    """
    CLIP-based video encoder for video classification.
    
    Processes video frames through CLIP vision model and pools across time dimension.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        
        self.model_type = cfg.model_type
        self.from_pretrained = MODEL_TYPE_TO_PRETRAINED[self.model_type]
        
        self.vision_model = AutoModel.from_pretrained(self.from_pretrained).vision_model
        
        self.embed_dim = self.vision_model.config.hidden_size
        
        configure_finetuning(
            self.vision_model,
            self.vision_model.encoder.layers,
            cfg.finetune_last_k_layers,
            self.vision_model.post_layernorm
        )

        # Optional temporal transformer (bidirectional across frames, applied before spatial pooling)
        temporal_heads = cfg.temporal_heads
        temporal_depth = cfg.temporal_depth
        self.temporal = TemporalTransformer(
            embed_dim=self.embed_dim,
            num_heads=temporal_heads,
            depth=temporal_depth,
            causal=False,
        ) if temporal_heads is not None else None

    def forward(self, pixel_values: torch.Tensor, return_temporal_features: bool = False) -> Tensor:
        """
        Forward pass through the CLIP video encoder.

        Args:
            pixel_values: Video tensor of shape [batch_size, num_frames, channels, height, width]
            return_temporal_features: If True, return per-frame features [batch_size, num_frames, hidden_size]
                                     If False, return pooled video features [batch_size, hidden_size]

        Returns:
            Video features - either pooled [batch_size, hidden_size] or temporal [batch_size, num_frames, hidden_size]
        """
        batch_size, num_frames, channels, height, width = pixel_values.shape
        frames = pixel_values.view(-1, channels, height, width)

        vision_outputs = self.vision_model(pixel_values=frames)
        # [B*T, seq_len, E]  (seq_len = 1 CLS + n_patches)
        sequence_output = vision_outputs.last_hidden_state

        # Reshape to [B, T, seq_len, E] so temporal transformer can attend across T
        seq_len = sequence_output.shape[1]
        tokens = sequence_output.view(batch_size, num_frames, seq_len, -1)  # [B, T, seq_len, E]

        # Apply bidirectional temporal transformer (no-op when self.temporal is None)
        if self.temporal is not None:
            tokens = self.temporal(tokens)  # [B, T, seq_len, E]

        # following CLIP: mean over patch tokens, skip CLS at index 0
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L1170-L1251
        frame_features = tokens[:, :, 1:, :].mean(dim=2)  # [B, T, E]

        if return_temporal_features:
            return frame_features  # [B, T, E]

        return frame_features.mean(dim=1)  # [B, E]


class VideoEncoderSigLIP2(nn.Module):
    """
    SigLIP2-based video encoder for video classification.
    
    Processes video frames through SigLIP2 vision model and pools across time dimension.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        self.model_type = cfg.model_type
        self.from_pretrained = MODEL_TYPE_TO_PRETRAINED[self.model_type]

        self.vision_model = AutoModel.from_pretrained(self.from_pretrained).vision_model
        self.embed_dim = self.vision_model.config.hidden_size

        configure_finetuning(
            self.vision_model,
            self.vision_model.encoder.layers,
            cfg.finetune_last_k_layers,
            self.vision_model.post_layernorm
        )

        # Optional temporal transformer (bidirectional across frames, applied before spatial pooling)
        temporal_heads = cfg.temporal_heads
        temporal_depth = cfg.temporal_depth
        self.temporal = TemporalTransformer(
            embed_dim=self.embed_dim,
            num_heads=temporal_heads,
            depth=temporal_depth,
            causal=False,
        ) if temporal_heads is not None else None

    def forward(self, pixel_values: torch.Tensor, return_temporal_features: bool = False) -> Tensor:
        """
        Forward pass through the SigLIP2 video encoder.

        Args:
            pixel_values: Video tensor of shape [batch_size, num_frames, channels, height, width]
            return_temporal_features: If True, return per-frame features [batch_size, num_frames, hidden_size]
                                     If False, return pooled video features [batch_size, hidden_size]

        Returns:
            Video features - either pooled [batch_size, hidden_size] or temporal [batch_size, num_frames, hidden_size]
        """
        batch_size, num_frames, channels, height, width = pixel_values.shape
        frames = pixel_values.view(-1, channels, height, width)

        vision_outputs = self.vision_model(pixel_values=frames)
        # [B*T, num_patches, E]  (SigLIP2 has no CLS token)
        sequence_output = vision_outputs.last_hidden_state

        # Reshape to [B, T, num_patches, E] for temporal processing
        num_patches = sequence_output.shape[1]
        tokens = sequence_output.view(batch_size, num_frames, num_patches, -1)  # [B, T, P, E]

        # Apply bidirectional temporal transformer (no-op when self.temporal is None)
        if self.temporal is not None:
            tokens = self.temporal(tokens)  # [B, T, P, E]

        # Mean pooling over all patches (SigLIP2 has no special CLS token)
        frame_features = tokens.mean(dim=2)  # [B, T, E]

        if return_temporal_features:
            return frame_features  # [B, T, E]

        return frame_features.mean(dim=1)  # [B, E]


class VideoEncoderDinov2(nn.Module):
    """
    DINOv2-based video encoder for video classification.
    
    Processes video frames through DINOv2 vision model and pools across time dimension.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        self.model_type = cfg.model_type
        self.from_pretrained = MODEL_TYPE_TO_PRETRAINED[self.model_type]

        self.vision_model = AutoModel.from_pretrained(self.from_pretrained)
        self._hidden_size = self.vision_model.config.hidden_size
        self.embed_dim = self._hidden_size * 2  # CLS + mean patches concatenated

        configure_finetuning(
            self.vision_model,
            self.vision_model.encoder.layer,  # DINOv2 uses .layer not .layers
            cfg.finetune_last_k_layers,
            self.vision_model.layernorm
        )

        # Temporal transformer runs on hidden_size (not *2) since it operates on
        # raw patch tokens before the CLS-concat output projection.
        temporal_heads = cfg.temporal_heads
        temporal_depth = cfg.temporal_depth
        self.temporal = TemporalTransformer(
            embed_dim=self._hidden_size,
            num_heads=temporal_heads,
            depth=temporal_depth,
            causal=False,
        ) if temporal_heads is not None else None
    
    def forward(self, pixel_values: torch.Tensor, return_temporal_features: bool = False) -> Tensor:
        """
        Forward pass through the DINOv2 video encoder.

        Args:
            pixel_values: Video tensor of shape [batch_size, num_frames, channels, height, width]
            return_temporal_features: If True, return per-frame features [batch_size, num_frames, hidden_size * 2]
                                     If False, return pooled video features [batch_size, hidden_size * 2]

        Returns:
            Video features - either pooled [batch_size, hidden_size * 2] or temporal [batch_size, num_frames, hidden_size * 2]
        """
        batch_size, num_frames, channels, height, width = pixel_values.shape
        frames = pixel_values.view(-1, channels, height, width)
        outputs = self.vision_model(pixel_values=frames)

        # following https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov2/modeling_dinov2.py#L555-L603
        # [B*T, seq_len, E]  (index 0 = CLS, 1: = patches)
        sequence_output = outputs.last_hidden_state

        # Reshape to [B, T, seq_len, E] so temporal transformer attends across T
        # for both CLS and patch tokens jointly (Option A).
        seq_len = sequence_output.shape[1]
        tokens = sequence_output.view(batch_size, num_frames, seq_len, -1)  # [B, T, seq_len, E]

        if self.temporal is not None:
            tokens = self.temporal(tokens)  # [B, T, seq_len, E]

        # DINOv2 output: concat CLS and mean of patch tokens per frame
        cls_tokens = tokens[:, :, 0, :]            # [B, T, E]
        patch_features = tokens[:, :, 1:, :].mean(dim=2)  # [B, T, E]
        frame_features = torch.cat([cls_tokens, patch_features], dim=-1)  # [B, T, 2E]

        if return_temporal_features:
            return frame_features  # [B, T, 2E]

        return frame_features.mean(dim=1)  # [B, 2E]


class VideoEncoderDinov3(nn.Module):
    """
    DINOv3-based video encoder for video classification.
    
    Processes video frames through DINOv3 vision model and pools across time dimension.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        self.model_type = cfg.model_type
        self.from_pretrained = MODEL_TYPE_TO_PRETRAINED[self.model_type]

        self.vision_model = AutoModel.from_pretrained(self.from_pretrained)
        self._hidden_size = self.vision_model.config.hidden_size
        self.embed_dim = self._hidden_size * 2  # CLS + mean patches concatenated

        configure_finetuning(
            self.vision_model,
            self.vision_model.layer,
            cfg.finetune_last_k_layers,
            self.vision_model.norm
        )

        # Temporal transformer runs on hidden_size (not *2) — same reasoning as DINOv2.
        temporal_heads = cfg.temporal_heads
        temporal_depth = cfg.temporal_depth
        self.temporal = TemporalTransformer(
            embed_dim=self._hidden_size,
            num_heads=temporal_heads,
            depth=temporal_depth,
            causal=False,
        ) if temporal_heads is not None else None
    
    def forward(self, pixel_values: torch.Tensor, return_temporal_features: bool = False) -> Tensor:
        """
        Forward pass through the DINOv3 video encoder.

        Args:
            pixel_values: Video tensor of shape [batch_size, num_frames, channels, height, width]
            return_temporal_features: If True, return per-frame features [batch_size, num_frames, hidden_size * 2]
                                     If False, return pooled video features [batch_size, hidden_size * 2]

        Returns:
            Video features - either pooled [batch_size, hidden_size * 2] or temporal [batch_size, num_frames, hidden_size * 2]
        """
        batch_size, num_frames, channels, height, width = pixel_values.shape
        frames = pixel_values.view(-1, channels, height, width)
        outputs = self.vision_model(pixel_values=frames)

        # Following DINOv2 pattern: [B*T, seq_len, E]  (index 0 = CLS, 1: = patches)
        sequence_output = outputs.last_hidden_state

        seq_len = sequence_output.shape[1]
        tokens = sequence_output.view(batch_size, num_frames, seq_len, -1)  # [B, T, seq_len, E]

        if self.temporal is not None:
            tokens = self.temporal(tokens)  # [B, T, seq_len, E]

        # DINOv3 output: concat CLS and mean of patch tokens per frame
        cls_tokens = tokens[:, :, 0, :]            # [B, T, E]
        patch_features = tokens[:, :, 1:, :].mean(dim=2)  # [B, T, E]
        frame_features = torch.cat([cls_tokens, patch_features], dim=-1)  # [B, T, 2E]

        if return_temporal_features:
            return frame_features  # [B, T, 2E]

        return frame_features.mean(dim=1)  # [B, 2E]


class VideoEncoderResNet50(nn.Module):
    """
    ResNet-50 video encoder trained from scratch (no pretrained weights).

    Processes each frame independently through ResNet-50 then pools across time.
    Feature maps from layer4 (7x7 spatial grid = 49 "patches" of dim 2048) are
    exposed for temporal attention when temporal_heads is configured.

    finetune_last_k_layers is ignored — all parameters are trainable since the
    model starts from random initialisation.
    """

    embed_dim: int = 2048  # ResNet-50 channel width after layer4

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        from torchvision.models import resnet50

        self.model_type = cfg.model_type

        # Load ResNet-50 with NO pretrained weights
        _resnet = resnet50(weights=None)

        # Backbone up to (and including) layer4, excluding avgpool and fc
        # layer4 output: [B, 2048, H/32, W/32]  e.g. 7x7 for 224x224 input
        self.backbone = nn.Sequential(
            _resnet.conv1, _resnet.bn1, _resnet.relu, _resnet.maxpool,
            _resnet.layer1, _resnet.layer2, _resnet.layer3, _resnet.layer4,
        )
        self.avgpool = _resnet.avgpool  # AdaptiveAvgPool2d(1,1)
        self.embed_dim = 2048

        # Optional temporal transformer (bidirectional, applied on spatial patches before pooling)
        temporal_heads = cfg.temporal_heads
        temporal_depth = cfg.temporal_depth
        self.temporal = TemporalTransformer(
            embed_dim=self.embed_dim,
            num_heads=temporal_heads,
            depth=temporal_depth,
            causal=False,
        ) if temporal_heads is not None else None

    def forward(self, pixel_values: torch.Tensor, return_temporal_features: bool = False) -> Tensor:
        """
        Forward pass through the ResNet-50 video encoder.

        Args:
            pixel_values: Video tensor of shape [batch_size, num_frames, channels, height, width]
            return_temporal_features: If True, return per-frame features [batch_size, num_frames, 2048]
                                     If False, return pooled video features [batch_size, 2048]

        Returns:
            Video features - either pooled [batch_size, 2048] or temporal [batch_size, num_frames, 2048]
        """
        batch_size, num_frames, channels, height, width = pixel_values.shape
        frames = pixel_values.view(-1, channels, height, width)  # [B*T, C, H, W]

        # Extract spatial feature maps from layer4: [B*T, 2048, h, w]  (h=w=7 for 224px)
        feature_maps = self.backbone(frames)  # [B*T, 2048, h, w]
        _, C, h, w = feature_maps.shape

        if self.temporal is not None:
            # Flatten spatial dims to patches: [B*T, h*w, 2048] -> [B, T, h*w, 2048]
            patches = feature_maps.flatten(2).transpose(1, 2)  # [B*T, h*w, C]
            tokens = patches.view(batch_size, num_frames, h * w, C)  # [B, T, P, E]

            tokens = self.temporal(tokens)  # [B, T, P, E]

            frame_features = tokens.mean(dim=2)  # [B, T, E]  — mean over spatial patches
        else:
            # Plain global average pooling, no temporal interaction
            pooled = self.avgpool(feature_maps).squeeze(-1).squeeze(-1)  # [B*T, 2048]
            frame_features = pooled.view(batch_size, num_frames, -1)     # [B, T, 2048]

        if return_temporal_features:
            return frame_features  # [B, T, 2048]

        return frame_features.mean(dim=1)  # [B, 2048]


class VideoEncoderVivit(nn.Module):
    """
    ViViT-based video encoder for video classification.
    
    Processes videos natively through ViViT model without temporal pooling.
    ViViT is designed to handle video sequences directly.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        
        
        self.model_type = cfg.model_type
        self.from_pretrained = MODEL_TYPE_TO_PRETRAINED[self.model_type]
        
        self.vision_model = AutoModel.from_pretrained(self.from_pretrained)
        
        self.expected_num_frames = 32
        self.embed_dim = self.vision_model.config.hidden_size
        
        configure_finetuning(
            self.vision_model,
            self.vision_model.encoder.layer,  # ViViT uses .layer not .layers
            cfg.finetune_last_k_layers,
            self.vision_model.layernorm
        )
    
    def forward(self, pixel_values: torch.Tensor, return_temporal_features: bool = False) -> Tensor:
        """
        Forward pass through the ViViT video encoder.
        
        Args:
            pixel_values: Video tensor of shape [batch_size, num_frames, channels, height, width]
            return_temporal_features: If True, return spatiotemporal tokens [batch_size, seq_len-1, hidden_size]
                                     (excluding CLS token)
                                     If False, return CLS token [batch_size, hidden_size]
            
        Returns:
            Video features - either CLS token [batch_size, hidden_size] or 
            spatiotemporal tokens [batch_size, seq_len-1, hidden_size]
        """
        # ViViT expects exactly 32 frames, so we need to resample if necessary
        pixel_values = temporal_sampling(pixel_values, self.expected_num_frames)
        
        outputs = self.vision_model(pixel_values=pixel_values)
        sequence_output = outputs.last_hidden_state  # [batch_size, sequence_length, hidden_size]
        
        if return_temporal_features:
            # Return all tokens except CLS token (spatiotemporal patch tokens)
            return sequence_output[:, 1:, :]  # [batch_size, seq_len-1, hidden_size]
        
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
        
        
        self.model_type = cfg.model_type
        self.from_pretrained = MODEL_TYPE_TO_PRETRAINED[self.model_type]
        
        self.vision_model = AutoModel.from_pretrained(self.from_pretrained)
        
        self.expected_num_frames = 16
        self.embed_dim = self.vision_model.config.hidden_size
        self.use_mean_pooling = self.vision_model.config.use_mean_pooling
        
        # Add layer norm if using mean pooling (following the classification head pattern)
        if self.use_mean_pooling:
            self.fc_norm = nn.LayerNorm(self.embed_dim)
        else:
            self.fc_norm = None
        
        configure_finetuning(
            self.vision_model,
            self.vision_model.encoder.layer,  # VideoMAE uses .layer
            cfg.finetune_last_k_layers,
            self.vision_model.layernorm
        )
    
    def forward(self, pixel_values: torch.Tensor, return_temporal_features: bool = False) -> Tensor:
        """
        Forward pass through the VideoMAE video encoder.
        
        Args:
            pixel_values: Video tensor of shape [batch_size, num_frames, channels, height, width]
            return_temporal_features: If True, return all spatiotemporal tokens [batch_size, seq_len, hidden_size]
                                     If False, return pooled/CLS video features [batch_size, hidden_size]
            
        Returns:
            Video features - either pooled [batch_size, hidden_size] or 
            spatiotemporal tokens [batch_size, seq_len, hidden_size]
        """
        # VideoMAE expects exactly 16 frames, so we need to resample if necessary
        pixel_values = temporal_sampling(pixel_values, self.expected_num_frames)
        
        outputs = self.vision_model(pixel_values=pixel_values)
        sequence_output = outputs.last_hidden_state  # [batch_size, sequence_length, hidden_size]
        
        if return_temporal_features:
            return sequence_output  # [batch_size, seq_len, hidden_size]
        
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
    
    Processes videos natively through VJEPA2 model.
    Supports returning:
    1. A single vector per video (Attentive Pooling)
    2. A vector per time-step (Spatial Pooling only)
    3. Raw spatiotemporal tokens (No Pooling)
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        
        
        self.model_type = cfg.model_type
        self.from_pretrained = MODEL_TYPE_TO_PRETRAINED[self.model_type]
        
        # Load Model
        self.vision_model = AutoModel.from_pretrained(self.from_pretrained)
        self.pooler = VJEPA2AttentivePooler(self.vision_model.config)
        
        # Store reference to the patch projection layer to calculate grid sizes later
        # VJEPA2Model structure: encoder.embeddings.patch_embeddings.proj
        self.patch_proj = self.vision_model.encoder.embeddings.patch_embeddings.proj
        
        self.embed_dim = self.vision_model.config.hidden_size
        
        configure_finetuning(
            self.vision_model,
            self.vision_model.encoder.layer,  # VJEPA2 uses encoder.layer
            cfg.finetune_last_k_layers,
            self.vision_model.encoder.layernorm
        )
            
    def _get_patch_grid_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, int, int]:
        """
        Calculates the (Depth/Time, Height, Width) of the patch grid 
        based on the input video shape and the model's convolution parameters.
        """
        # input_shape is (Batch, Frames, Channels, Height, Width)
        batch_size, D_in, in_channels, H_in, W_in = input_shape
        
        # Extract Conv3d parameters from the model's patch embedding layer
        K_d, K_h, K_w = self.patch_proj.kernel_size
        S_d, S_h, S_w = self.patch_proj.stride
        P_d, P_h, P_w = self.patch_proj.padding
        Dil_d, Dil_h, Dil_w = self.patch_proj.dilation

        # Apply Conv3d dimension formula: floor((In + 2*P - Dil*(K-1) - 1)/S + 1)
        # Note: VJEPA2 typically uses tubelet_size=2, patch_size=16
        D_out = math.floor((D_in + 2 * P_d - Dil_d * (K_d - 1) - 1) / S_d + 1)
        H_out = math.floor((H_in + 2 * P_h - Dil_h * (K_h - 1) - 1) / S_h + 1)
        W_out = math.floor((W_in + 2 * P_w - Dil_w * (K_w - 1) - 1) / S_w + 1)

        return D_out, H_out, W_out
    
    def get_temporal_embeddings(self, pixel_values: torch.Tensor) -> Tensor:
        """
        Get embeddings with spatial dimensions collapsed but time preserved.
        
        Args:
            pixel_values: (Batch, Frames, Channels, Height, Width)
            
        Returns:
            Tensor: (Batch, Time_Patches, Hidden_Size)
        """
        # 1. Forward pass to get flat tokens
        outputs = self.vision_model(
            pixel_values_videos=pixel_values,
            skip_predictor=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        last_hidden_state = outputs.last_hidden_state # (Batch, Total_Patches, Dim)
        
        # 2. Calculate the 3D grid shape
        D_out, H_out, W_out = self._get_patch_grid_shape(pixel_values.shape)
        
        batch_size, total_patches, dim = last_hidden_state.shape
        
        # 3. Unflatten: (Batch, Total_Patches, Dim) -> (Batch, Time, Height, Width, Dim)
        # VJEPA flattens in T, H, W order
        features_3d = last_hidden_state.view(batch_size, D_out, H_out, W_out, dim)
        
        # 4. Collapse spatial dimensions (Height and Width), keeping Time
        temporal_features = features_3d.mean(dim=(2, 3)) # (Batch, Time, Dim)
        
        return temporal_features

    def forward(self, 
                pixel_values: torch.Tensor, 
                return_temporal_features: bool = False) -> Tensor:
        """
        Forward pass through the VJEPA2 video encoder.
        
        Args:
            pixel_values: Video tensor of shape [batch_size, num_frames, channels, height, width]
            return_all_tokens: Return raw flattened tokens [batch_size, seq_len, hidden_size]
            return_time_series: Return spatially pooled time series [batch_size, time_steps, hidden_size]
            
        Returns:
            Tensor: Video features based on flags. Default is pooled [batch_size, hidden_size]
        """
        
        # If user wants time series specifically, use the optimized path
        if return_temporal_features:
            return self.get_temporal_embeddings(pixel_values)

        # Standard Forward
        outputs = self.vision_model(
            pixel_values_videos=pixel_values,
            skip_predictor=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        
        last_hidden_state = outputs.last_hidden_state  # [batch_size, sequence_length, hidden_size]
        
        video_features = self.pooler(last_hidden_state)  # [batch_size, hidden_size]
        
        return video_features


def get_embed_dim_for_model_type(model_type: str) -> int:
    """
    Get the embedding dimension for a given model type without initializing the full model.
    This is useful when using precomputed embeddings to save memory.

    Args:
        model_type: Short alias ('clip', 'siglip2', ...) or full HF name
                    ('google/siglip2-base-patch16-224', ...)

    Returns:
        The embedding dimension for the specified model type
    """
    model_type = normalize_model_type(model_type)  # accept full HF names
    pretrained_name = MODEL_TYPE_TO_PRETRAINED[model_type]

    # Get config without loading weights
    if model_type == 'clip':
        from transformers import CLIPConfig
        config = CLIPConfig.from_pretrained(pretrained_name)
        return config.vision_config.hidden_size
    elif model_type == 'siglip2':
        from transformers import Siglip2Config
        config = Siglip2Config.from_pretrained(pretrained_name)
        return config.vision_config.hidden_size
    elif model_type == 'dinov2':
        from transformers import Dinov2Config
        config = Dinov2Config.from_pretrained(pretrained_name)
        return config.hidden_size * 2  # DINOv2 uses concatenated CLS and patch tokens
    elif model_type == 'dinov3':
        from transformers import Dinov3Config
        config = Dinov3Config.from_pretrained(pretrained_name)
        return config.hidden_size * 2  # DINOv3 uses concatenated CLS and patch tokens
    elif model_type == 'vivit':
        from transformers import VivitConfig
        config = VivitConfig.from_pretrained(pretrained_name)
        return config.hidden_size
    elif model_type == 'videomae':
        from transformers import VideoMAEConfig
        config = VideoMAEConfig.from_pretrained(pretrained_name)
        return config.hidden_size
    elif model_type == 'vjepa2':
        from transformers import VJEPA2Config
        config = VJEPA2Config.from_pretrained(pretrained_name)
        return config.hidden_size
    elif model_type == 'resnet50':
        return 2048  # fixed — ResNet-50 layer4 output channels
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


class VideoEncoder(nn.Module):
    """
    Factory class for video encoders that automatically selects the appropriate encoder
    based on the model_type parameter.
    
    Supports:
    - CLIP models (model_type="clip")
    - SigLIP2 models (model_type="siglip2")
    - DINOv2 models (model_type="dinov2")
    - DINOv3 models (model_type="dinov3")
    - ViViT models (model_type="vivit")
    - VideoMAE models (model_type="videomae")
    - VJEPA2 models (model_type="vjepa2")
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        # Normalise so both 'siglip2' and 'google/siglip2-base-patch16-224' work
        model_type = normalize_model_type(cfg.model_type)
        cfg.model_type = model_type  # write back so individual encoders see the alias

        if model_type == 'clip':
            self.video_encoder = VideoEncoderClip(cfg)
        elif model_type == 'siglip2':
            self.video_encoder = VideoEncoderSigLIP2(cfg)
        elif model_type == 'dinov2':
            self.video_encoder = VideoEncoderDinov2(cfg)
        elif model_type == 'dinov3':
            self.video_encoder = VideoEncoderDinov3(cfg)
        elif model_type == 'vivit':
            self.video_encoder = VideoEncoderVivit(cfg)
        elif model_type == 'videomae':
            self.video_encoder = VideoEncoderVideoMAE(cfg)
        elif model_type == 'vjepa2':
            self.video_encoder = VideoEncoderVJEPA2(cfg)
        elif model_type == 'resnet50':
            self.video_encoder = VideoEncoderResNet50(cfg)
        else:
            raise ValueError(
                f"Unknown model_type: '{model_type}'. "
                f"Supported types: {list(MODEL_TYPE_TO_PRETRAINED.keys())}"
            )

        self.embed_dim = self.video_encoder.embed_dim
    
    def forward(self, pixel_values: torch.Tensor, return_temporal_features: bool = False) -> Tensor:
        """
        Forward pass through the selected video encoder.
        
        Args:
            pixel_values: Video tensor of shape [batch_size, num_frames, channels, height, width]
            return_temporal_features: If True, return temporal/spatiotemporal features
                                     If False, return pooled video features
        
        Returns:
            Video features - shape depends on return_temporal_features and model type:
            - If return_temporal_features=False: [batch_size, hidden_size]
            - If return_temporal_features=True:
              - CLIP/SigLIP2/DINOv2: [batch_size, num_frames, hidden_size]
              - ViViT/VideoMAE/VJEPA2: [batch_size, seq_len, hidden_size]
        """
        return self.video_encoder.forward(pixel_values, return_temporal_features)
    
    def get_embeddings(self, pixel_values: torch.Tensor, return_temporal_features: bool = False) -> Tensor:
        """Get video embeddings (alias for forward method)."""
        return self.forward(pixel_values, return_temporal_features)
