"""
Multi-Agent Trajectory Decoder

This module implements a transformer-based decoder for predicting joint 
multi-agent location distributions in a permutation-invariant manner.

The decoder uses learnable agent query embeddings and transformer layers
to predict locations for multiple target agents simultaneously.
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    from architecture_utils import ACT2CLS, build_mlp
except ImportError:
    from .architecture_utils import ACT2CLS, build_mlp


class MultiAgentTrajectoryDecoder(nn.Module):
    """
    Transformer-based decoder for multi-agent trajectory prediction.
    
    This decoder predicts joint location distributions for multiple agents
    in a permutation-invariant manner using learnable agent queries and
    transformer decoder layers.
    
    Architecture:
    1. Project input features (team embedding) to hidden dimension
    2. Use learnable agent query embeddings (one per target agent)
    3. Apply transformer decoder layers with cross-attention
    4. Project to output space (coordinates, grid, or density)
    
    The permutation invariance property means the order of predicted agents
    doesn't matter - this is achieved through learnable queries and proper
    loss functions (e.g., Hungarian matching, optimal transport).
    """
    
    def __init__(self, cfg):
        """
        Initialize multi-agent trajectory decoder.
        
        Args:
            cfg: Configuration object with fields:
                - model.traj_decoder.num_target_agents: Number of agents to predict (e.g., 5)
                - model.traj_decoder.hidden_dim: Hidden dimension for transformer
                - model.traj_decoder.num_layers: Number of transformer decoder layers
                - model.traj_decoder.num_heads: Number of attention heads
                - model.traj_decoder.dropout: Dropout rate
                - model.traj_decoder.dim_feedforward: Feedforward dimension in transformer
                - model.traj_decoder.use_learned_queries: Whether to use learned agent queries
                - model.traj_decoder.output_mode: Output type ('coord', 'grid', 'density')
                - model.activation: Activation function name
                
                Input dimension comes from combined_dim (fused_agent_dim + team_embed_dim)
                For coordinate mode, output_dim is automatically set to 3 (x, y, z)
        """
        super().__init__()
        
        self.cfg = cfg
        self.traj_cfg = cfg.model.traj_decoder
        
        # Architecture parameters
        self.num_target_agents = self.traj_cfg.num_target_agents
        self.hidden_dim = self.traj_cfg.hidden_dim
        self.num_layers = self.traj_cfg.num_layers
        self.num_heads = self.traj_cfg.num_heads
        self.dropout = self.traj_cfg.dropout
        self.dim_feedforward = self.traj_cfg.dim_feedforward
        self.use_learned_queries = self.traj_cfg.use_learned_queries
        self.output_mode = self.traj_cfg.output_mode
        
        # Input/output dimensions
        self.combined_dim = cfg.model.agent_fusion.fused_agent_dim + cfg.model.team_embed_dim
        
        # Determine output dimension per agent based on mode
        if self.output_mode == 'coord':
            self.output_dim_per_agent = 3  # (x, y, z) coordinates
        elif self.output_mode == 'grid':
            grid_resolution = cfg.data.grid_resolution
            self.output_dim_per_agent = grid_resolution * grid_resolution
        elif self.output_mode == 'density':
            grid_resolution = cfg.data.grid_resolution
            self.output_dim_per_agent = grid_resolution * grid_resolution
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")
        
        # Input projection: project combined features to hidden_dim
        self.input_projection = nn.Linear(self.combined_dim, self.hidden_dim)
        
        # Learnable agent query embeddings [num_target_agents, hidden_dim]
        # These queries learn to extract information about different agents
        if self.use_learned_queries:
            self.agent_queries = nn.Parameter(
                torch.randn(self.num_target_agents, self.hidden_dim)
            )
        else:
            # Alternative: use positional embeddings for agent indices
            self.agent_position_embedding = nn.Embedding(
                self.num_target_agents, self.hidden_dim
            )
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self._get_activation_name(),
            batch_first=True,
            norm_first=True  # Pre-LN architecture for better training stability
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.num_layers
        )
        
        # Output projection: map each agent query to its output space
        self.output_projection = build_mlp(
            input_dim=self.hidden_dim,
            output_dim=self.output_dim_per_agent,
            num_hidden_layers=2,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            activation=cfg.model.activation
        )
        
        # Layer normalization for stability
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        
        self._init_weights()
    
    def _get_activation_name(self):
        """Get PyTorch activation name for TransformerDecoderLayer."""
        activation_map = {
            'gelu': 'gelu',
            'relu': 'relu',
            'leaky_relu': 'relu',  # TransformerDecoderLayer only supports 'relu' and 'gelu'
            'silu': 'gelu',
            'quick_gelu': 'gelu',
            'linear': 'gelu'
        }
        return activation_map.get(self.cfg.model.activation, 'gelu')
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform initialization."""
        # Initialize agent queries with normal distribution
        if self.use_learned_queries:
            nn.init.normal_(self.agent_queries, std=0.02)
        
        # Initialize linear layers
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
    
    def forward(
        self, 
        combined_features: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through trajectory decoder.
        
        Args:
            combined_features: [B, combined_dim] - Combined team and agent features
                where combined_dim = fused_agent_dim + team_embed_dim
            memory_mask: Optional [B, 1] mask for memory (not typically used)
            
        Returns:
            predictions: Output predictions based on output_mode:
                - 'coord': [B, num_target_agents, 3] - (x, y, z) coordinates
                - 'grid': [B, num_target_agents, grid_resolution^2] - grid heatmaps
                - 'density': [B, num_target_agents, grid_resolution^2] - density distributions
        
        The decoder uses cross-attention where:
        - Query: Learnable agent embeddings [num_target_agents, hidden_dim]
        - Key/Value: Projected input features [1, hidden_dim]
        
        This allows each agent query to attend to the shared context (team state)
        and extract relevant information for predicting that agent's location.
        """
        batch_size = combined_features.shape[0]
        
        # Project input features to hidden dimension
        # [B, combined_dim] -> [B, 1, hidden_dim]
        memory = self.input_projection(combined_features).unsqueeze(1)
        
        # Prepare agent queries
        if self.use_learned_queries:
            # Use learned agent query embeddings
            # [num_target_agents, hidden_dim] -> [B, num_target_agents, hidden_dim]
            queries = self.agent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # Use positional embeddings for agent indices
            agent_indices = torch.arange(
                self.num_target_agents, 
                device=combined_features.device
            )
            # [num_target_agents] -> [B, num_target_agents, hidden_dim]
            queries = self.agent_position_embedding(agent_indices).unsqueeze(0).expand(
                batch_size, -1, -1
            )
        
        # Apply transformer decoder
        # queries: [B, num_target_agents, hidden_dim]
        # memory: [B, 1, hidden_dim]
        # output: [B, num_target_agents, hidden_dim]
        decoded = self.transformer_decoder(
            tgt=queries,
            memory=memory,
            memory_mask=memory_mask
        )
        
        # Apply output normalization
        decoded = self.output_norm(decoded)  # [B, num_target_agents, hidden_dim]
        
        # Project to output space
        # [B, num_target_agents, hidden_dim] -> [B, num_target_agents, output_dim_per_agent]
        predictions = self.output_projection(decoded)
        
        return predictions


class SetPredictionDecoder(nn.Module):
    """
    Simplified set prediction decoder for multi-agent trajectory prediction.
    
    This is a lighter alternative to the full transformer decoder, using
    just cross-attention between agent queries and context.
    """
    
    def __init__(self, cfg):
        """Initialize set prediction decoder."""
        super().__init__()
        
        self.cfg = cfg
        self.traj_cfg = cfg.model.traj_decoder
        
        self.num_target_agents = self.traj_cfg.num_target_agents
        self.hidden_dim = self.traj_cfg.hidden_dim
        self.num_heads = self.traj_cfg.num_heads
        self.dropout = self.traj_cfg.dropout
        self.output_mode = self.traj_cfg.output_mode
        
        # Input dimension
        self.combined_dim = cfg.model.agent_fusion.fused_agent_dim + cfg.model.team_embed_dim
        
        # Output dimension per agent
        if self.output_mode == 'coord':
            self.output_dim_per_agent = 3
        elif self.output_mode in ['grid', 'density']:
            grid_resolution = cfg.data.grid_resolution
            self.output_dim_per_agent = grid_resolution * grid_resolution
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")
        
        # Learnable agent queries
        self.agent_queries = nn.Parameter(
            torch.randn(self.num_target_agents, self.hidden_dim)
        )
        
        # Context projection
        self.context_projection = nn.Linear(self.combined_dim, self.hidden_dim)
        
        # Cross-attention from queries to context
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Self-attention among agent queries
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Feedforward network
        self.ffn = build_mlp(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_hidden_layers=1,
            hidden_dim=self.traj_cfg.dim_feedforward,
            dropout=self.dropout,
            activation=cfg.model.activation
        )
        
        # Output head
        self.output_head = nn.Linear(self.hidden_dim, self.output_dim_per_agent)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.agent_queries, std=0.02)
        nn.init.xavier_uniform_(self.context_projection.weight)
        nn.init.zeros_(self.context_projection.bias)
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
    
    def forward(self, combined_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through set prediction decoder.
        
        Args:
            combined_features: [B, combined_dim] - Combined features
            
        Returns:
            predictions: [B, num_target_agents, output_dim_per_agent]
        """
        batch_size = combined_features.shape[0]
        
        # Project context
        context = self.context_projection(combined_features).unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Expand queries for batch
        queries = self.agent_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, A, hidden_dim]
        
        # Cross-attention: queries attend to context
        attn_output, _ = self.cross_attention(
            query=queries,
            key=context,
            value=context
        )
        queries = self.norm1(queries + attn_output)
        
        # Self-attention: queries interact with each other
        self_attn_output, _ = self.self_attention(
            query=queries,
            key=queries,
            value=queries
        )
        queries = self.norm2(queries + self_attn_output)
        
        # Feedforward
        ffn_output = self.ffn(queries)
        queries = self.norm3(queries + ffn_output)
        
        # Output projection
        predictions = self.output_head(queries)  # [B, num_target_agents, output_dim_per_agent]
        
        return predictions

