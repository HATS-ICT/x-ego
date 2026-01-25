"""
Video Decoder for Reconstruction Auxiliary Task

Adapted from a video world model architecture. Uses a block-causal transformer
decoder to reconstruct video patches from latent embeddings.

Architecture:
    Latent Embedding -> Up-projection -> Transformer Decoder -> Patch Head -> Video Patches
    
The decoder uses learnable patch queries that attend to the latent representation
to reconstruct the original video frames.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoid_table(n: int, d: int, base: float = 10000.0, device=None) -> torch.Tensor:
    """Generate sinusoidal positional encoding table."""
    pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(d, device=device, dtype=torch.float32).unsqueeze(0)
    k = torch.floor(i / 2.0)
    div = torch.exp(-(2.0 * k) / max(1.0, float(d)) * math.log(base))
    ang = pos * div
    return torch.where((i % 2) == 0, torch.sin(ang), torch.cos(ang))


def add_sinusoidal_positions(tokens_btSd: torch.Tensor) -> torch.Tensor:
    """Add sinusoidal positional encoding to tokens."""
    B, T, S, D = tokens_btSd.shape
    device = tokens_btSd.device
    pos_t = sinusoid_table(T, D, device=device)
    pos_s = sinusoid_table(S, D, device=device)
    pos = (pos_t[None, :, None, :] + pos_s[None, None, :, :]) * (1.0 / math.sqrt(D))
    return tokens_btSd + pos.to(dtype=tokens_btSd.dtype)


def temporal_unpatchify(patches_btnd: torch.Tensor, H: int, W: int, C: int, patch: int) -> torch.Tensor:
    """
    Convert patches back to video frames.
    
    Args:
        patches_btnd: (B, T, Np, Dp) patch tokens
        H, W: Original frame height and width
        C: Number of channels
        patch: Patch size
        
    Returns:
        video: (B, T, C, H, W) reconstructed video
    """
    assert patches_btnd.dim() == 4
    B, T, Np, Dp = patches_btnd.shape
    assert Dp == C * patch * patch
    x = patches_btnd.reshape(B * T, Np, Dp).transpose(1, 2).contiguous()
    out = F.fold(x, output_size=(H, W), kernel_size=patch, stride=patch)
    return out.reshape(B, T, C, H, W)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * (self.scale / torch.sqrt(var + self.eps))


class MLP(nn.Module):
    """MLP with SiLU-gated activation (SwiGLU variant)."""
    
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc_in = nn.Linear(d_model, 2 * hidden)
        self.fc_out = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.fc_in(x).chunk(2, dim=-1)
        h = u * F.silu(v)
        h = self.drop(h)
        y = self.fc_out(h)
        y = self.drop(y)
        return y


class MultiheadSelfAttention(nn.Module):
    """Multi-head self-attention using PyTorch's scaled_dot_product_attention."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = float(dropout)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x_nld: torch.Tensor, *, attn_mask: Optional[torch.Tensor] = None, is_causal: bool = False):
        N, L, D = x_nld.shape
        q, k, v = self.qkv(x_nld).chunk(3, dim=-1)

        q = q.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)

        drop = self.dropout_p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(N, L, D)
        return self.out(y)


class SpaceSelfAttention(nn.Module):
    """
    Spatial self-attention for decoder.
    
    In decoder mode:
    - Latent queries can only attend to other latents
    - Patch queries can attend to latents and same-modality patches
    """
    
    def __init__(self, d_model: int, n_heads: int, n_latents: int, n_patches: int, dropout: float = 0.0):
        super().__init__()
        self.n_latents = n_latents
        self.n_patches = n_patches
        S = n_latents + n_patches
        
        # Build attention mask for decoder
        # Latent queries: can only attend to latents
        # Patch queries: can attend to latents and all patches
        allow = torch.ones((S, S), dtype=torch.bool)
        
        # Latent queries (rows 0:n_latents) can only see latent keys (cols 0:n_latents)
        allow[:n_latents, n_latents:] = False
        
        attn_mask = allow.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)
        self.register_buffer("attn_mask", attn_mask, persistent=False)
        
        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout=dropout)

    def forward(self, x_btSd: torch.Tensor) -> torch.Tensor:
        B, T, S, D = x_btSd.shape
        x = x_btSd.reshape(B * T, S, D)
        mask = self.attn_mask.expand(B * T, 1, S, S)
        y = self.attn(x, attn_mask=mask, is_causal=False)
        return y.reshape(B, T, S, D)


class TimeSelfAttention(nn.Module):
    """Temporal self-attention across time steps (causal)."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, latents_only: bool = True, n_latents: int = 0):
        super().__init__()
        self.latents_only = latents_only
        self.n_latents = n_latents
        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout=dropout)

    def forward(self, x_btSd: torch.Tensor) -> torch.Tensor:
        B, T, S, D = x_btSd.shape
        if self.latents_only and self.n_latents > 0:
            L = self.n_latents
            lat = x_btSd[:, :, :L, :]
            lat_nld = lat.permute(0, 2, 1, 3).contiguous().view(B * L, T, D)
            out = self.attn(lat_nld, is_causal=True)
            out = out.view(B, L, T, D).permute(0, 2, 1, 3).contiguous()
            x = x_btSd.clone()
            x[:, :, :L, :] = out
            return x
        else:
            x_nld = x_btSd.permute(0, 2, 1, 3).contiguous().view(B * S, T, D)
            out = self.attn(x_nld, is_causal=True)
            return out.view(B, S, T, D).permute(0, 2, 1, 3).contiguous()


class DecoderBlock(nn.Module):
    """Single decoder transformer block with space and time attention."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_latents: int,
        n_patches: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        layer_index: int = 0,
        time_every: int = 4,
        latents_only_time: bool = True,
    ):
        super().__init__()
        self.do_time = ((layer_index + 1) % time_every == 0)

        self.norm1 = RMSNorm(d_model)
        self.space = SpaceSelfAttention(d_model, n_heads, n_latents, n_patches, dropout)
        self.drop1 = nn.Dropout(dropout)

        if self.do_time:
            self.norm2 = RMSNorm(d_model)
            self.time = TimeSelfAttention(d_model, n_heads, dropout, latents_only_time, n_latents)
            self.drop2 = nn.Dropout(dropout)

        self.norm3 = RMSNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio=mlp_ratio, dropout=dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.space(self.norm1(x)))
        if self.do_time:
            x = x + self.drop2(self.time(self.norm2(x)))
        x = x + self.drop3(self.mlp(self.norm3(x)))
        return x


class DecoderTransformer(nn.Module):
    """Stack of decoder transformer blocks."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        depth: int,
        n_latents: int,
        n_patches: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        time_every: int = 4,
        latents_only_time: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_latents=n_latents,
                n_patches=n_patches,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                layer_index=i,
                time_every=time_every,
                latents_only_time=latents_only_time,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class VideoDecoder(nn.Module):
    """
    Transformer-based video decoder for reconstruction.
    
    Takes latent embeddings and reconstructs video patches using a transformer
    decoder with learnable patch queries.
    
    Architecture:
        1. Project latent embedding to decoder dimension
        2. Concatenate with learnable patch queries
        3. Add sinusoidal positional encoding
        4. Process through transformer decoder
        5. Project patch tokens to pixel values
        6. Reshape to video format
    
    Args:
        input_dim: Dimension of input latent embeddings
        d_model: Transformer hidden dimension
        n_heads: Number of attention heads
        depth: Number of transformer layers
        n_latents: Number of latent tokens (typically 1 for single embedding)
        target_frame_size: Output frame resolution (H=W)
        num_frames: Number of frames to reconstruct
        patch_size: Size of each patch (default 8)
        dropout: Dropout rate
        mlp_ratio: MLP hidden dimension ratio
        time_every: Apply temporal attention every N layers
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        n_heads: int = 8,
        depth: int = 4,
        n_latents: int = 1,
        target_frame_size: int = 64,
        num_frames: int = 4,
        patch_size: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        time_every: int = 2,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_latents = n_latents
        self.target_frame_size = target_frame_size
        self.num_frames = num_frames
        self.patch_size = patch_size
        
        # Calculate number of patches
        assert target_frame_size % patch_size == 0, \
            f"target_frame_size ({target_frame_size}) must be divisible by patch_size ({patch_size})"
        self.n_patches_per_side = target_frame_size // patch_size
        self.n_patches = self.n_patches_per_side ** 2
        
        # Patch dimension: C * patch_size * patch_size
        self.d_patch = 3 * patch_size * patch_size
        
        # Project input embedding to latent tokens
        self.up_proj = nn.Linear(input_dim, d_model * n_latents)
        
        # Learnable patch queries
        self.patch_queries = nn.Parameter(torch.empty(self.n_patches, d_model))
        nn.init.normal_(self.patch_queries, std=0.02)
        
        # Transformer decoder
        self.transformer = DecoderTransformer(
            d_model=d_model,
            n_heads=n_heads,
            depth=depth,
            n_latents=n_latents,
            n_patches=self.n_patches,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            time_every=time_every,
            latents_only_time=True,
        )
        
        # Project to patch pixels
        self.patch_head = nn.Linear(d_model, self.d_patch)
        
        # Final normalization
        self.final_norm = RMSNorm(d_model)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Decode embeddings to video frames.
        
        Args:
            embeddings: [B, input_dim] latent embeddings
            
        Returns:
            frames: [B, num_frames, 3, H, W] reconstructed video frames
        """
        B = embeddings.shape[0]
        T = self.num_frames
        
        # Project to latent tokens: [B, n_latents * d_model] -> [B, T, n_latents, d_model]
        lat = self.up_proj(embeddings)  # [B, n_latents * d_model]
        lat = lat.view(B, self.n_latents, self.d_model)  # [B, n_latents, d_model]
        lat = torch.tanh(lat)  # Bounded activation
        lat = lat.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, n_latents, d_model]
        
        # Expand patch queries: [n_patches, d_model] -> [B, T, n_patches, d_model]
        qry = self.patch_queries.view(1, 1, self.n_patches, self.d_model)
        qry = qry.expand(B, T, -1, -1)
        
        # Concatenate latents and queries
        tokens = torch.cat([lat, qry], dim=2)  # [B, T, n_latents + n_patches, d_model]
        
        # Add positional encoding
        tokens = add_sinusoidal_positions(tokens)
        
        # Transformer decoder
        x = self.transformer(tokens)
        
        # Extract patch tokens (skip latent tokens)
        x_patches = x[:, :, self.n_latents:, :]  # [B, T, n_patches, d_model]
        
        # Normalize and project to pixels
        x_patches = self.final_norm(x_patches)
        patches = torch.sigmoid(self.patch_head(x_patches))  # [B, T, n_patches, d_patch]
        
        # Unpatchify to video
        video = temporal_unpatchify(
            patches,
            H=self.target_frame_size,
            W=self.target_frame_size,
            C=3,
            patch=self.patch_size
        )  # [B, T, 3, H, W]
        
        return video


