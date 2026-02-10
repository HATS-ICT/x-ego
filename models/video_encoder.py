import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, SiglipModel, Dinov2Model, VivitModel, VideoMAEModel, VJEPA2Model
from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2AttentivePooler
from torch import Tensor
from typing import Dict, Any, Tuple
import math


# Mapping from model_type to HuggingFace pretrained model identifier
MODEL_TYPE_TO_PRETRAINED = {
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-base-patch16-224",
    "dinov2": "facebook/dinov2-base",
    "vivit": "google/vivit-b-16x2-kinetics400",
    "videomae": "MCG-NJU/videomae-base",
    "vjepa2": "facebook/vjepa2-vitl-fpc16-256-ssv2",
}

def _to_model_device_dtype(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """
    Ensure an input tensor lives on the same device/dtype as a model's parameters.
    Prevents runtime errors like:
      Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
    """
    try:
        p = next(model.parameters())
    except StopIteration:
        return x
    return x.to(device=p.device, dtype=p.dtype)


def _flatten_video_batch(pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Accept video tensors with arbitrary leading batch dimensions.

    Expected trailing dims: (num_frames, channels, height, width)
      - 4D: (T, C, H, W)                 -> treated as no leading batch dims
      - 5D: (B, T, C, H, W)              -> typical
      - ND: (..., T, C, H, W)            -> e.g. (time, env, agent, T, C, H, W)

    Returns:
        flat_pixel_values: (B_flat, T, C, H, W)
        batch_dims: original leading batch dims (...). Empty tuple means "no leading batch dims".
    """
    if pixel_values.ndim < 4:
        raise ValueError(
            f"Expected pixel_values with at least 4 dims (..., T, C, H, W) or (T, C, H, W), "
            f"got shape={tuple(pixel_values.shape)}"
        )

    if pixel_values.ndim == 4:
        # (T, C, H, W) -> (1, T, C, H, W)
        pixel_values = pixel_values.unsqueeze(0)
        batch_dims: Tuple[int, ...] = ()
    else:
        batch_dims = tuple(int(d) for d in pixel_values.shape[:-4])

    t, c, h, w = (int(d) for d in pixel_values.shape[-4:])
    b_flat = int(math.prod(pixel_values.shape[:-4])) if pixel_values.shape[:-4] else 1
    flat = pixel_values.reshape(b_flat, t, c, h, w)
    return flat, batch_dims


def _unflatten_batch(x: torch.Tensor, batch_dims: Tuple[int, ...]) -> torch.Tensor:
    """Undo _flatten_video_batch's leading-batch flattening on a tensor whose first dim is B_flat."""
    if batch_dims:
        return x.reshape(*batch_dims, *x.shape[1:])
    # Drop the artificial leading batch dim (B_flat == 1) when there were no leading batch dims.
    return x.reshape(*x.shape[1:])


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


class VideoEncoderCNN(nn.Module):
    """
    Lightweight trainable CNN encoder for RL visual observations.

    Expects video input with a singleton temporal dimension (T=1), i.e.
    shape [..., 1, C, H, W]. Returns pooled embeddings of shape
    [..., embed_dim], and can optionally return per-frame embeddings
    [..., 1, embed_dim] for compatibility with the shared interface.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.model_type = cfg.model_type

        in_channels = int(getattr(cfg, "in_channels", 3))
        c1 = int(getattr(cfg, "c1", 32))
        c2 = int(getattr(cfg, "c2", 64))
        c3 = int(getattr(cfg, "c3", 64))
        self.embed_dim = int(getattr(cfg, "embed_dim", 256))
        act_name = str(getattr(cfg, "activation", "relu")).lower()

        if act_name == "silu":
            act_cls = nn.SiLU
        elif act_name == "gelu":
            act_cls = nn.GELU
        else:
            act_cls = nn.ReLU

        # Nature-CNN style trunk with GroupNorm for PPO-friendly stability.
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=8, stride=4),
            nn.GroupNorm(num_groups=8, num_channels=c1),
            act_cls(),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2),
            nn.GroupNorm(num_groups=8, num_channels=c2),
            act_cls(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1),
            nn.GroupNorm(num_groups=8, num_channels=c3),
            act_cls(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, self.embed_dim),
            act_cls(),
        )

        if getattr(cfg, "freeze_backbone", False):
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor, return_temporal_features: bool = False) -> Tensor:
        # Defensive sync: in some TorchRL/BenchMARL initialization paths, model
        # modules may still be on CPU when the first collector batch arrives on CUDA.
        # Keep encoder params on the incoming tensor device to avoid conv dtype/device mismatches.
        try:
            p = next(self.parameters())
            if p.device != pixel_values.device:
                self.to(pixel_values.device)
        except StopIteration:
            pass
        
        pixel_values, batch_dims = _flatten_video_batch(pixel_values)  # (B_flat, T, C, H, W)
        batch_size, num_frames, channels, height, width = pixel_values.shape

        if num_frames != 1:
            raise ValueError(
                f"VideoEncoderCNN expects T=1 for now (got T={num_frames}, shape={tuple(pixel_values.shape)}). "
                "Set frame_stack=1 or update VideoEncoderCNN to aggregate multiple frames."
            )

        frames = pixel_values.reshape(batch_size, channels, height, width)
        frames = _to_model_device_dtype(frames, self)
        features = self.trunk(frames)
        embeddings = self.proj(features)  # [B_flat, embed_dim]

        if return_temporal_features:
            temporal = embeddings.unsqueeze(1)  # [B_flat, 1, embed_dim]
            return _unflatten_batch(temporal, batch_dims)

        return _unflatten_batch(embeddings, batch_dims)


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
        pixel_values = _to_model_device_dtype(pixel_values, self.vision_model)
        pixel_values, batch_dims = _flatten_video_batch(pixel_values)  # (B_flat, T, C, H, W)
        batch_size, num_frames, channels, height, width = pixel_values.shape
        frames = pixel_values.reshape(-1, channels, height, width)
        
        vision_outputs = self.vision_model(pixel_values=frames)
        sequence_output = vision_outputs.last_hidden_state  # [batch_size * num_frames, seq_len, hidden_size]

        # following https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L1170-L1251
        frame_features = torch.mean(sequence_output[:, 1:, :], dim=1)  # [batch_size * num_frames, hidden_size]
        frame_features = frame_features.view(batch_size, num_frames, -1)  # [batch_size, num_frames, hidden_size]
        
        if return_temporal_features:
            return _unflatten_batch(frame_features, batch_dims)  # [..., T, hidden_size]
        
        video_features = torch.mean(frame_features, dim=1)  # [B_flat, hidden_size]
        return _unflatten_batch(video_features, batch_dims)  # [..., hidden_size]


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
    
    def forward(self, pixel_values: torch.Tensor, return_temporal_features: bool = False) -> Tensor:
        """
        Forward pass through the SigLIP video encoder.
        
        Args:
            pixel_values: Video tensor of shape [batch_size, num_frames, channels, height, width]
            return_temporal_features: If True, return per-frame features [batch_size, num_frames, hidden_size]
                                     If False, return pooled video features [batch_size, hidden_size]
            
        Returns:
            Video features - either pooled [batch_size, hidden_size] or temporal [batch_size, num_frames, hidden_size]
        """
        pixel_values = _to_model_device_dtype(pixel_values, self.vision_model)
        pixel_values, batch_dims = _flatten_video_batch(pixel_values)  # (B_flat, T, C, H, W)
        batch_size, num_frames, channels, height, width = pixel_values.shape
        frames = pixel_values.reshape(-1, channels, height, width)
        
        vision_outputs = self.vision_model(pixel_values=frames)
        sequence_output = vision_outputs.last_hidden_state  # [batch_size * num_frames, seq_len, hidden_size]
        
        # following https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L1007-L1109
        frame_features = torch.mean(sequence_output, dim=1)  # [batch_size * num_frames, hidden_size]
        frame_features = frame_features.view(batch_size, num_frames, -1)  # [batch_size, num_frames, hidden_size]
        
        if return_temporal_features:
            return _unflatten_batch(frame_features, batch_dims)  # [..., T, hidden_size]
        
        video_features = torch.mean(frame_features, dim=1)  # [B_flat, hidden_size]
        return _unflatten_batch(video_features, batch_dims)  # [..., hidden_size]


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
        pixel_values = _to_model_device_dtype(pixel_values, self.vision_model)
        pixel_values, batch_dims = _flatten_video_batch(pixel_values)  # (B_flat, T, C, H, W)
        batch_size, num_frames, channels, height, width = pixel_values.shape
        frames = pixel_values.reshape(-1, channels, height, width)
        outputs = self.vision_model(pixel_values=frames)
        
        # following https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov2/modeling_dinov2.py#L555-L603
        # get frame features
        sequence_output = outputs.last_hidden_state  # [batch_size * num_frames, seq_len, hidden_size]
        cls_tokens = sequence_output[:, 0]  # [batch_size * num_frames, hidden_size]
        patch_tokens = sequence_output[:, 1:]  # [batch_size * num_frames, num_patches, hidden_size]
        patch_features = torch.mean(patch_tokens, dim=1)  # [batch_size * num_frames, hidden_size]
        frame_features = torch.cat([cls_tokens, patch_features], dim=1)  # [batch_size * num_frames, hidden_size * 2]
        
        # reshape to [batch_size, num_frames, hidden_size * 2]
        frame_features = frame_features.view(batch_size, num_frames, -1)
        
        if return_temporal_features:
            return _unflatten_batch(frame_features, batch_dims)  # [..., T, hidden_size * 2]
        
        video_features = torch.mean(frame_features, dim=1)  # [B_flat, hidden_size * 2]
        return _unflatten_batch(video_features, batch_dims)  # [..., hidden_size * 2]


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
        pixel_values = _to_model_device_dtype(pixel_values, self.vision_model)
        pixel_values, batch_dims = _flatten_video_batch(pixel_values)  # (B_flat, T, C, H, W)
        # ViViT expects exactly 32 frames, so we need to resample if necessary
        pixel_values = temporal_sampling(pixel_values, self.expected_num_frames)
        
        outputs = self.vision_model(pixel_values=pixel_values)
        sequence_output = outputs.last_hidden_state  # [batch_size, sequence_length, hidden_size]
        
        if return_temporal_features:
            # Return all tokens except CLS token (spatiotemporal patch tokens)
            return _unflatten_batch(sequence_output[:, 1:, :], batch_dims)  # [..., seq_len-1, hidden_size]
        
        # following https://github.com/huggingface/transformers/blob/bb45d3631ec7026db04a77d33a52b31766372160/src/transformers/models/vivit/modeling_vivit.py#L566-L687
        video_features = sequence_output[:, 0, :]  # [B_flat, hidden_size]
        return _unflatten_batch(video_features, batch_dims)  # [..., hidden_size]


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
        pixel_values = _to_model_device_dtype(pixel_values, self.vision_model)
        pixel_values, batch_dims = _flatten_video_batch(pixel_values)  # (B_flat, T, C, H, W)
        # VideoMAE expects exactly 16 frames, so we need to resample if necessary
        pixel_values = temporal_sampling(pixel_values, self.expected_num_frames)
        
        outputs = self.vision_model(pixel_values=pixel_values)
        sequence_output = outputs.last_hidden_state  # [batch_size, sequence_length, hidden_size]
        
        if return_temporal_features:
            return _unflatten_batch(sequence_output, batch_dims)  # [..., seq_len, hidden_size]
        
        if self.use_mean_pooling:
            # Use mean pooling over all tokens (following VideoMAE classification head)
            video_features = sequence_output.mean(1)  # [batch_size, hidden_size]
            if self.fc_norm is not None:
                video_features = self.fc_norm(video_features)
        else:
            # Use CLS token (first token) as video representation
            video_features = sequence_output[:, 0]  # [batch_size, hidden_size]
        
        return _unflatten_batch(video_features, batch_dims)


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
        self.cfg = cfg
        
        self.model_type = cfg.model_type
        # Assuming you have this mapping defined globally or pass the string directly
        self.from_pretrained = cfg.get('from_pretrained', 'facebook/vjepa2-vitl-fpc16-256-ssv2')
        
        # Load Model
        self.vision_model = VJEPA2Model.from_pretrained(self.from_pretrained)
        self.pooler = VJEPA2AttentivePooler(self.vision_model.config)
        
        # Store reference to the patch projection layer to calculate grid sizes later
        self.patch_proj = self.vision_model.embeddings.patch_embeddings.projection
        
        self.embed_dim = self.vision_model.config.hidden_size
        
        if getattr(cfg, 'freeze_backbone', False):
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze VJEPA2 model parameters."""
        for param in self.vision_model.parameters():
            param.requires_grad = False
            
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
        pixel_values = _to_model_device_dtype(pixel_values, self.vision_model)
        pixel_values, batch_dims = _flatten_video_batch(pixel_values)  # (B_flat, T, C, H, W)
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
        
        return _unflatten_batch(temporal_features, batch_dims)

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
        
        pixel_values = _to_model_device_dtype(pixel_values, self.vision_model)
        pixel_values, batch_dims = _flatten_video_batch(pixel_values)  # (B_flat, T, C, H, W)

        # If user wants time series specifically, use the optimized path
        if return_temporal_features:
            # Inline the temporal path here so we can preserve original leading batch dims.
            outputs = self.vision_model(
                pixel_values_videos=pixel_values,
                skip_predictor=True,
                output_attentions=False,
                output_hidden_states=False,
            )
            last_hidden_state = outputs.last_hidden_state  # (B_flat, Total_Patches, Dim)
            D_out, H_out, W_out = self._get_patch_grid_shape(pixel_values.shape)
            b_flat, total_patches, dim = last_hidden_state.shape
            features_3d = last_hidden_state.view(b_flat, D_out, H_out, W_out, dim)
            temporal_features = features_3d.mean(dim=(2, 3))  # (B_flat, Time, Dim)
            return _unflatten_batch(temporal_features, batch_dims)

        # Standard Forward
        outputs = self.vision_model(
            pixel_values_videos=pixel_values,
            skip_predictor=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        
        last_hidden_state = outputs.last_hidden_state  # [batch_size, sequence_length, hidden_size]
        
        video_features = self.pooler(last_hidden_state)  # [B_flat, hidden_size]
        return _unflatten_batch(video_features, batch_dims)


def get_embed_dim_for_model_type(model_type: str) -> int:
    """
    Get the embedding dimension for a given model type without initializing the full model.
    This is useful when using precomputed embeddings to save memory.
    
    Args:
        model_type: The model type (clip, siglip, dinov2, vivit, videomae, vjepa2)
        
    Returns:
        The embedding dimension for the specified model type
    """
    if model_type == "cnn":
        # Default embed dim for VideoEncoderCNN when no cfg is available.
        return 256

    if model_type not in MODEL_TYPE_TO_PRETRAINED:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Supported types: {list(MODEL_TYPE_TO_PRETRAINED.keys()) + ['cnn']}"
        )
    
    pretrained_name = MODEL_TYPE_TO_PRETRAINED[model_type]
    
    # Get config without loading weights
    if model_type == 'clip':
        from transformers import CLIPConfig
        config = CLIPConfig.from_pretrained(pretrained_name)
        return config.vision_config.hidden_size
    elif model_type == 'siglip':
        from transformers import SiglipConfig
        config = SiglipConfig.from_pretrained(pretrained_name)
        return config.vision_config.hidden_size
    elif model_type == 'dinov2':
        from transformers import Dinov2Config
        config = Dinov2Config.from_pretrained(pretrained_name)
        return config.hidden_size * 2  # DINOv2 uses concatenated CLS and patch tokens
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
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


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
    - Custom CNN model (model_type="cnn")
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
        elif model_type == 'cnn':
            self.video_encoder = VideoEncoderCNN(cfg)
        else:
            raise ValueError(
                f"Unknown model_type: '{model_type}'. "
                f"Supported types: {list(MODEL_TYPE_TO_PRETRAINED.keys()) + ['cnn']}"
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
              - CLIP/SigLIP/DINOv2: [batch_size, num_frames, hidden_size]
              - ViViT/VideoMAE/VJEPA2: [batch_size, seq_len, hidden_size]
        """
        return self.video_encoder.forward(pixel_values, return_temporal_features)
    
    def get_embeddings(self, pixel_values: torch.Tensor, return_temporal_features: bool = False) -> Tensor:
        """Get video embeddings (alias for forward method)."""
        return self.forward(pixel_values, return_temporal_features)
