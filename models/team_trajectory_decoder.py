import torch
import torch.nn as nn

from models.architecture_utils import ACT2CLS


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        x = self.dropout(x)
        
        return x


class FeedForward(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, dropout, activation):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout, activation):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, mlp_ratio, dropout, activation)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TeamTrajectoryDecoder(nn.Module):
    """
    Team Trajectory Decoder module.
    
    Predicts trajectories for 5 players from a target team based on per-agent embeddings,
    POV team, and target team information. Outputs 60 timepoints (15s at 4Hz) with (X, Y) coordinates.
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        # Get dimensions from config
        self.agent_embed_dim = cfg.model.encoder.proj_dim
        self.num_agents = cfg.data.num_pov_agents
        self.team_embed_dim = cfg.model.team_embed_dim
        
        # Trajectory decoder configuration
        self.num_timepoints = cfg.num_trajectory_timepoints  # 60
        self.num_target_players = cfg.num_target_players  # 5
        self.output_dim_per_player = self.num_timepoints * 2  # 60 timepoints * 2 coords
        
        # Team embeddings
        self.pov_team_emb = nn.Embedding(2, self.team_embed_dim)  # POV team
        self.target_team_emb = nn.Embedding(2, self.team_embed_dim)  # Target team
        
        # Combined dimension: per-agent embeddings + pov team + target team
        per_agent_combined_dim = self.agent_embed_dim + self.team_embed_dim * 2
        activation = ACT2CLS[cfg.model.activation]
        
        # Transformer blocks for trajectory decoding
        self.blocks = nn.ModuleList([
            TransformerBlock(
                per_agent_combined_dim, 
                cfg.model.traj_decoder.num_heads, 
                cfg.model.traj_decoder.mlp_ratio, 
                cfg.model.traj_decoder.dropout, 
                activation
            )
            for _ in range(cfg.model.traj_decoder.num_layers)
        ])
        
        self.norm = nn.LayerNorm(per_agent_combined_dim)
        self.head = nn.Linear(per_agent_combined_dim, self.output_dim_per_player)
    
    def forward(self, agent_embeddings, pov_team_encoded, target_team_encoded):
        """
        Forward pass.
        
        Args:
            agent_embeddings: [B, A, embed_dim] per-agent video embeddings
            pov_team_encoded: [B] POV team (0=T, 1=CT)
            target_team_encoded: [B] Target team (0=T, 1=CT)
                
        Returns:
            predictions: [B, 5, 60, 2] trajectory predictions
        """
        B, A, embed_dim = agent_embeddings.shape
        
        # Encode team information
        pov_team_embedding = self.pov_team_emb(pov_team_encoded)  # [B, team_embed_dim]
        target_team_embedding = self.target_team_emb(target_team_encoded)  # [B, team_embed_dim]
        
        # Expand team embeddings to match agent dimension
        pov_team_embedding = pov_team_embedding.unsqueeze(1).expand(B, A, self.team_embed_dim)  # [B, A, team_embed_dim]
        target_team_embedding = target_team_embedding.unsqueeze(1).expand(B, A, self.team_embed_dim)  # [B, A, team_embed_dim]
        
        # Combine agent embeddings with team information
        x = torch.cat([agent_embeddings, pov_team_embedding, target_team_embedding], dim=-1)  # [B, A, combined_dim]
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)  # [B, A, combined_dim]
        predictions = self.head(x)  # [B, A, 60*2]
        
        predictions = predictions.view(B, A, self.num_timepoints, 2)  # [B, A, 60, 2]
        
        return predictions
