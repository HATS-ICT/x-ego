import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from typing import Optional
from omegaconf import OmegaConf, DictConfig
from transformers import VJEPA2VideoProcessor, VJEPA2Model
import math

try:
    from .utils import SelectIndex
except ImportError:
    from utils import SelectIndex
try:
    from .rope import RopeAttention
except ImportError:
    from rope import RopeAttention
 
 
class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with integrated MLP and RoPE attention."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Multi-head attention
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = RopeAttention(config)
        
        # Forward
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attention_outputs = self.attention(hidden_states)
        hidden_states = attention_outputs[0] + residual
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states) + residual
        return (hidden_states,)

    
class Conv3DPatchEmbeddings(nn.Module):
    """
    Construct mask token, position and patch embeddings.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.tubelet_size = config.tubelet_size
        self.hidden_size = config.hidden_size

        self.proj = nn.Conv3d(
            in_channels=config.in_chans,
            out_channels=config.hidden_size,
            kernel_size=(config.tubelet_size, config.patch_size, config.patch_size),
            stride=(config.tubelet_size, config.patch_size, config.patch_size),
        )
    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        pixel_values_videos = pixel_values_videos.permute(0, 2, 1, 3, 4) # (B, F, C, H, W) -> (B, C, F, H, W)
        embeddings = self.proj(pixel_values_videos).flatten(2).transpose(1, 2)
        return embeddings


class CTFMVideoTransformer(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.embeddings = Conv3DPatchEmbeddings(config)
        self.layer = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values_videos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values_videos)
        for layer_module in self.layer:
            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs[0]
        hidden_states = self.layernorm(hidden_states)
        return hidden_states
    
    
class CTFMVideoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.processor = VJEPA2VideoProcessor.from_pretrained(config.processor_model)
        self.transformer = CTFMVideoTransformer(config)
        
        self.embed_dim = config.hidden_size
        self.pooling_type = config.pooling
        self.proj_after_pooling = config.proj_after_pooling
        self.skip_projection = config.skip_projection
        
        self._init_pooling(config)
        if not self.skip_projection:
            self.projection = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.SiLU(),
                nn.Linear(self.embed_dim, config.shared_proj_dim)
            )
    
    def _init_pooling(self, config):
        """Initialize pooling layer based on pooling type"""
        
        pooler_map = {
            'mean': lambda: nn.AdaptiveAvgPool1d(1),
            'max': lambda: nn.AdaptiveMaxPool1d(1),
            'cls': lambda: SelectIndex(0),
            'eos': lambda: SelectIndex(-1),
        }
        
        if self.pooling_type in pooler_map:
            self.pooler = pooler_map[self.pooling_type]()
        else:
            supported_types = list(pooler_map.keys()) + ['attentive']
            raise ValueError(f"Invalid video pooling: {self.pooling_type}. "
                            f"Supported types: {', '.join(supported_types)}")
    
    def _apply_pooling(self, features):
        """Apply pooling to features with correct dimension handling"""
        if self.pooling_type in ['mean', 'max']:
            # For mean/max pooling, we need to pool along the sequence dimension (dim=1)
            # Input: [batch_size, seq_len, feature_dim] -> [batch_size, feature_dim, seq_len]
            features_transposed = features.transpose(1, 2)
            pooled = self.pooler(features_transposed)  # [batch_size, feature_dim, 1]
            pooled = pooled.squeeze(-1)  # [batch_size, feature_dim]
        elif self.pooling_type == 'attentive':
            # Attentive pooler expects [batch_size, seq_len, feature_dim]
            pooled = self.pooler(features)  # [batch_size, feature_dim]
        else:
            # For cls/eos pooling, we select along the sequence dimension
            pooled = self.pooler(features)  # [batch_size, feature_dim]
        return pooled
        
    def forward(self, pixel_values_videos):
        last_hidden_state = self.transformer(pixel_values_videos)
        
        # Apply pooling and projection in the correct order based on config
        if self.skip_projection:
            pooled_features = self._apply_pooling(last_hidden_state)
            final_features = pooled_features
        else:
            if self.proj_after_pooling:
                # Pool first, then project
                pooled_features = self._apply_pooling(last_hidden_state)
                projected_features = self.projection(pooled_features)
                final_features = projected_features
            else:
                # Project first, then pool
                projected_features = self.projection(last_hidden_state)
                pooled_features = self._apply_pooling(projected_features)
                final_features = pooled_features
            
        return final_features
    
    def get_embeddings(self, pixel_values_videos):
        return self.forward(pixel_values_videos)
    
@dataclass
class CTFMVideoEncoderModelOutput(ModelOutput):
    last_hidden_state: torch.Tensor

class CTFMVideoEncoderModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Get model configuration
        from_pretrained = config['from_pretrained']  # 'facebook/vjepa2-vitl-fpc16-256-ssv2'
        freeze_backbone = config['freeze_backbone']
        
        self.processor = VJEPA2VideoProcessor.from_pretrained(from_pretrained)
        vjepa2 = VJEPA2Model.from_pretrained(from_pretrained, attn_implementation="eager")
        
        self.encoder = vjepa2.encoder
        self.conv3d_proj = vjepa2.get_input_embeddings().proj
        
        # Freeze backbone encoder parameters if specified (pooling and projection remain trainable)
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.embed_dim = self.encoder.config.hidden_size
        
        # Get pooling and projection configuration
        self.pooling_type = config['pooling']
        self.proj_after_pooling = config['proj_after_pooling']
        
        # Initialize pooling layer (trainable)
        self._init_pooling(config)
        
        # Initialize projection layer if needed (trainable)
        shared_dim = config['shared_proj_dim']
        self.projection = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, shared_dim)
        )
    
    def _init_pooling(self, config):
        """Initialize pooling layer based on pooling type"""
        from .utils import SelectIndex
        from transformers import VJEPA2Config
        from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2AttentivePooler
        
        pooler_map = {
            'mean': lambda: nn.AdaptiveAvgPool1d(1),
            'max': lambda: nn.AdaptiveMaxPool1d(1),
            'cls': lambda: SelectIndex(0),
            'eos': lambda: SelectIndex(-1),
        }
        
        # Handle special cases
        if self.pooling_type == 'attentive':
            vjepa_config = VJEPA2Config.from_pretrained(config['from_pretrained'], attn_implementation="eager")
            self.pooler = VJEPA2AttentivePooler(vjepa_config)
        elif self.pooling_type in pooler_map:
            self.pooler = pooler_map[self.pooling_type]()
        else:
            supported_types = list(pooler_map.keys()) + ['attentive']
            raise ValueError(f"Invalid video pooling: {self.pooling_type}. "
                            f"Supported types: {', '.join(supported_types)}")
    
    def _apply_pooling(self, features):
        """Apply pooling to features with correct dimension handling"""
        if self.pooling_type in ['mean', 'max']:
            # For mean/max pooling, we need to pool along the sequence dimension (dim=1)
            # Input: [batch_size, seq_len, feature_dim] -> [batch_size, feature_dim, seq_len]
            features_transposed = features.transpose(1, 2)
            pooled = self.pooler(features_transposed)  # [batch_size, feature_dim, 1]
            pooled = pooled.squeeze(-1)  # [batch_size, feature_dim]
        elif self.pooling_type == 'attentive':
            # Attentive pooler expects [batch_size, seq_len, feature_dim]
            pooled = self.pooler(features)  # [batch_size, feature_dim]
        else:
            # For cls/eos pooling, we select along the sequence dimension
            pooled = self.pooler(features)  # [batch_size, feature_dim]
        return pooled
        
    def forward(self, pixel_values_videos, return_patch_embeddings=False):
        encoder_output = self.encoder(pixel_values_videos)
        last_hidden_state = encoder_output.last_hidden_state # (batch_size, total_patches, feature_dim)
        
        # Apply pooling and projection in the correct order based on config
        if self.proj_after_pooling:
            # Pool first, then project
            pooled_features = self._apply_pooling(last_hidden_state)
            projected_features = self.projection(pooled_features)
            final_features = projected_features
        else:
            # Project first, then pool
            projected_features = self.projection(last_hidden_state)
            pooled_features = self._apply_pooling(projected_features)
            final_features = pooled_features
        
        if return_patch_embeddings:
            batch_size, out_channels, D_out, H_out, W_out = self.conv3d_output_shape(pixel_values_videos.shape)
            batch_size, total_patches, feature_dim = last_hidden_state.shape
            patch_features = last_hidden_state.transpose(1, 2).view(batch_size, feature_dim, D_out, H_out, W_out)
        else:
            patch_features = None
            
        output = CTFMVideoEncoderModelOutput(
            last_hidden_state=final_features
        )
        return output
    
    def get_embeddings(self, pixel_values_videos, return_patch_embeddings=False):
        output = self.forward(pixel_values_videos, return_patch_embeddings)
        return output 
        
    def conv3d_output_shape(self, pixel_values_videos_input_shape):
        batch_size, D_in, in_channels, H_in, W_in = pixel_values_videos_input_shape
        
        # Extract Conv3d parameters
        K_d, K_h, K_w = self.conv3d_proj.kernel_size
        S_d, S_h, S_w = self.conv3d_proj.stride
        P_d, P_h, P_w = self.conv3d_proj.padding
        Dil_d, Dil_h, Dil_w = self.conv3d_proj.dilation
        out_channels = self.conv3d_proj.out_channels

        # Apply Conv3d dimension formula
        D_out = math.floor((D_in + 2 * P_d - Dil_d * (K_d - 1) - 1) / S_d + 1)
        H_out = math.floor((H_in + 2 * P_h - Dil_h * (K_h - 1) - 1) / S_h + 1)
        W_out = math.floor((W_in + 2 * P_w - Dil_w * (K_w - 1) - 1) / S_w + 1)

        return torch.Size((batch_size, out_channels, D_out, H_out, W_out))
        
    
if __name__ == "__main__":
    # Configuration dictionary
    config_dict = {
        "patch_size": 16,
        "crop_size": 224,
        "frames_per_clip": 20,
        "tubelet_size": 2,
        "hidden_size": 1024,
        "in_chans": 3,
        "num_attention_heads": 16,
        "num_hidden_layers": 12,
        "mlp_ratio": 4.0,
        "layer_norm_eps": 1e-6,
        "qkv_bias": True,
        "attention_probs_dropout_prob": 0.0,
        "initializer_range": 0.02,
        "attention_dropout": 0.0,
        "from_pretrained": "facebook/vjepa2-vitl-fpc16-256-ssv2",
        "pooling": "mean",
        "proj_after_pooling": True,
        "shared_proj_dim": 1024
    }
    
    # Load configuration with OmegaConf
    config = OmegaConf.create(config_dict)
    print("Configuration loaded:")
    print(OmegaConf.to_yaml(config))
    
    # Initialize the model
    model = CTFMVideoEncoder(config)
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create a random tensor for testing
    # Shape: (batch_size, frames, channels, height, width)
    batch_size = 2
    frames = config.frames_per_clip
    channels = config.in_chans
    height = config.crop_size
    width = config.crop_size
    
    random_tensor = torch.randn(batch_size, frames, channels, height, width)
    print(f"\nInput tensor shape: {random_tensor.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(random_tensor)
        print(f"Output tensor shape: {output.shape}")
        print(f"Output tensor mean: {output.mean().item():.6f}")
        print(f"Output tensor std: {output.std().item():.6f}")
    
    print("\nTest completed successfully!")
