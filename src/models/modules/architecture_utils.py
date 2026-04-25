"""
Architecture utilities for building neural network components.

This module provides common activation functions and building blocks
for constructing neural network architectures.
"""

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
import math

from src.models.modules.norm import AdaptiveNormalizer


class QuickGELU(nn.Module):
    """Quick GELU activation: x * sigmoid(1.702 * x)"""
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


# Activation function mapping
ACT2CLS = {
    'gelu': nn.GELU,
    'leaky_relu': nn.LeakyReLU,
    'relu': nn.ReLU,
    'linear': nn.Identity,
    'silu': nn.SiLU,
    'quick_gelu': QuickGELU
}


def build_mlp(input_dim, output_dim, num_hidden_layers, hidden_dim, activation='gelu'):
    """
    Build a multi-layer perceptron with specified number of hidden layers.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        num_hidden_layers: Number of hidden layers
        hidden_dim: Dimension of hidden layers
        activation: Name of activation function from ACT2CLS
    
    Returns:
        nn.Sequential: MLP model
    """
    layers = []
    current_dim = input_dim
    
    for i in range(num_hidden_layers):
        layers.extend([
            nn.Linear(current_dim, hidden_dim),
            ACT2CLS[activation](),
        ])
        current_dim = hidden_dim
    
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)



class TemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, causal=True, conditioning_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.norm = AdaptiveNormalizer(embed_dim, conditioning_dim)
        self.causal = causal
        
    def forward(self, x, conditioning=None):
        B, T, P, E = x.shape
        
        # project to Q, K, V and split into heads: [B, T, P, E] -> [(B*P), H, T, D] 
        # (4 dims to work with torch compile attention)
        q = rearrange(self.q_proj(x), 'b t p (h d) -> (b p) h t d', h=self.num_heads)
        k = rearrange(self.k_proj(x), 'b t p (h d) -> (b p) h t d', h=self.num_heads)
        v = rearrange(self.v_proj(x), 'b t p (h d) -> (b p) h t d', h=self.num_heads) # [B, P, H, T, D]

        k_t = k.transpose(-2, -1) # [(B*P), H, T, D, T]

        # attention(q, k, v) = softmax(qk^T / sqrt(d)) v
        scores = torch.matmul(q, k_t) / math.sqrt(self.head_dim) # [(B*P), H, T, T]

        # causal mask for each token t in seq, mask out all tokens to the right of t (after t)
        if self.causal:
            mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
            scores = scores.masked_fill(mask, -torch.inf) # [(B*P), H, T, T]

        attn_weights = F.softmax(scores, dim=-1) # [(B*P), H, T, T]
        attn_output = torch.matmul(attn_weights, v) # [(B*P), H, T, D]
        attn_output = rearrange(attn_output, '(b p) h t d -> b t p (h d)', b=B, p=P) # [B, T, P, E]

        # out proj to mix head information
        attn_out = self.out_proj(attn_output)  # [B, T, P, E]

        # residual and optionally conditioned norm
        out = self.norm(x + attn_out, conditioning) # [B, T, P, E]

        return out # [B, T, P, E]

class SwiGLUFFN(nn.Module):
    # swiglu(x) = W3(sigmoid(W1(x) + b1) * (W2(x) + b2)) + b3
    def __init__(self, embed_dim, hidden_dim, conditioning_dim=None):
        super().__init__()
        h = math.floor(2 * hidden_dim / 3)
        self.w_v = nn.Linear(embed_dim, h)
        self.w_g = nn.Linear(embed_dim, h)
        self.w_o = nn.Linear(h, embed_dim)
        self.norm = AdaptiveNormalizer(embed_dim, conditioning_dim)

    def forward(self, x, conditioning=None):
        v = F.silu(self.w_v(x)) # [B, T, P, h]
        g = self.w_g(x) # [B, T, P, h]
        out = self.w_o(v * g) # [B, T, P, E]
        return self.norm(x + out, conditioning) # [B, T, P, E]


class TemporalTransformerBlock(nn.Module):
    """
    Single temporal transformer block: TemporalAttention followed by SwiGLUFFN.

    Input/output shape: [B, T, P, E]
      B = batch, T = frames, P = spatial patches, E = embed_dim
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 causal: bool = False, conditioning_dim=None):
        super().__init__()
        self.attn = TemporalAttention(embed_dim, num_heads, causal=causal,
                                      conditioning_dim=conditioning_dim)
        self.ffn = SwiGLUFFN(embed_dim, int(embed_dim * mlp_ratio),
                             conditioning_dim=conditioning_dim)

    def forward(self, x: torch.Tensor, conditioning=None) -> torch.Tensor:
        x = self.attn(x, conditioning)  # [B, T, P, E]
        x = self.ffn(x, conditioning)   # [B, T, P, E]
        return x


class TemporalTransformer(nn.Module):
    """
    Stack of TemporalTransformerBlocks for post-encoding temporal modelling.

    Designed to sit after a per-frame image encoder (CLIP, SigLIP2, DINOv2, DINOv3).
    Operates on [B, T, P, E] where P is the number of spatial patch tokens kept
    from the backbone before final pooling.
    """
    def __init__(self, embed_dim: int, num_heads: int, depth: int = 1,
                 mlp_ratio: float = 4.0, causal: bool = False,
                 conditioning_dim=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            TemporalTransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio,
                                     causal=causal, conditioning_dim=conditioning_dim)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, conditioning=None) -> torch.Tensor:
        # x: [B, T, P, E]
        for block in self.blocks:
            x = block(x, conditioning)
        return x  # [B, T, P, E]
