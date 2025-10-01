"""
Agent Fusion Module

This module implements multi-agent embedding fusion strategies for combining
information from multiple agent perspectives into a unified representation.
"""

import torch
import torch.nn as nn


class AgentFuser(nn.Module):
    """
    Multi-agent embedding fusion module.
    
    Fuses agent embeddings using specified fusion method:
    1. Apply fusion method (mean, max, attention, concat)
    2. Project to target dimensionality via MLP
    
    Fusion Methods:
    - mean: Average pooling over agent dimension
    - max: Max pooling over agent dimension
    - attention: Multi-head self-attention over agents
    - concat: Concatenate all agent embeddings
    """
    
    def __init__(self, embed_dim, num_agents, fusion_cfg, activation_fn, dropout):
        """
        Initialize agent fusion module.
        
        Args:
            embed_dim: Dimension of each agent's embedding
            num_agents: Number of agents (A)
            fusion_cfg: Agent fusion configuration with fields:
                - method: Fusion method ('mean', 'max', 'attention', 'concat')
                - fused_agent_dim: Output dimension after fusion
                - num_layers: Number of MLP layers for projection
                - num_attn_heads: Number of attention heads (for attention method)
            activation_fn: Activation function class
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_agents = num_agents
        self.fusion_method = fusion_cfg.method
        self.fused_agent_dim = fusion_cfg.fused_agent_dim
        self.activation_fn = activation_fn
        self.dropout = dropout
        
        # Method-specific layers
        if self.fusion_method == 'attention':
            self.agent_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=fusion_cfg.num_attn_heads,
                batch_first=True
            )
        
        # Determine input dimension for MLP based on fusion method
        if self.fusion_method == 'concat':
            mlp_input_dim = embed_dim * num_agents
        else:
            # mean, max, attention all produce [B, embed_dim]
            mlp_input_dim = embed_dim
        
        # Build projection MLP
        self.fusion_mlp = self._build_mlp(
            mlp_input_dim,
            self.fused_agent_dim,
            fusion_cfg.num_layers,
            self.fused_agent_dim,  # Use fused_agent_dim as hidden_dim
            dropout
        )
    
    def _build_mlp(self, input_dim, output_dim, num_layers, hidden_dim, dropout):
        """Build a multi-layer perceptron."""
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                self.activation_fn(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)
    
    def forward(self, agent_embeddings):
        """
        Fuse multi-agent embeddings into a unified representation.
        
        Args:
            agent_embeddings: [B, A, embed_dim] - per-agent embeddings
                B: batch size
                A: number of agents
                embed_dim: embedding dimension
            
        Returns:
            fused_embeddings: [B, fused_agent_dim] - fused embeddings
        """
        B, A, embed_dim = agent_embeddings.shape
        
        # Apply fusion method
        if self.fusion_method == 'mean':
            fused_embeddings = torch.mean(agent_embeddings, dim=1)
        elif self.fusion_method == 'max':
            fused_embeddings, _ = torch.max(agent_embeddings, dim=1)
        elif self.fusion_method == 'attention':
            fused_embeddings, _ = self.agent_attention(
                agent_embeddings, agent_embeddings, agent_embeddings
            )
            fused_embeddings = torch.mean(fused_embeddings, dim=1)
        elif self.fusion_method == 'concat':
            fused_embeddings = agent_embeddings.view(B, -1)  # [B, A*embed_dim]
        else:
            raise ValueError(f"Unknown agent fusion method: {self.fusion_method}")
        
        # Project to target dimensionality
        fused_embeddings = self.fusion_mlp(fused_embeddings)  # [B, fused_agent_dim]
        
        return fused_embeddings

