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
    def __init__(self, cfg):
        super().__init__()
        
        self.feat_dim = cfg.traj_decoder.feat_dim
        self.seq_len = cfg.traj_decoder.seq_len
        self.output_dim = cfg.traj_decoder.seq_len * 2
        self.team_emb_dim = cfg.traj_decoder.team_emb_dim
        
        self.self_team_emb = nn.Embedding(cfg.traj_decoder.num_teams, cfg.traj_decoder.team_emb_dim)
        self.target_team_emb = nn.Embedding(cfg.traj_decoder.num_teams, cfg.traj_decoder.team_emb_dim)
        
        combined_dim = cfg.traj_decoder.feat_dim + cfg.traj_decoder.team_emb_dim * 2
        activation = ACT2CLS[cfg.model.activation]
        
        self.blocks = nn.ModuleList([
            TransformerBlock(combined_dim, cfg.traj_decoder.num_heads, cfg.traj_decoder.mlp_ratio, cfg.traj_decoder.dropout, activation)
            for _ in range(cfg.traj_decoder.num_layers)
        ])
        
        self.norm = nn.LayerNorm(combined_dim)
        self.head = nn.Linear(combined_dim, self.output_dim)
        
    def forward(self, x, self_team, target_team):
        B, N, _ = x.shape
        
        self_team_emb = self.self_team_emb(self_team)
        target_team_emb = self.target_team_emb(target_team)
        
        self_team_emb = self_team_emb.unsqueeze(1).expand(B, N, self.team_emb_dim)
        target_team_emb = target_team_emb.unsqueeze(1).expand(B, N, self.team_emb_dim)
        
        x = torch.cat([x, self_team_emb, target_team_emb], dim=-1)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = self.head(x)
        
        return x
