"""
Cross-Ego Contrastive Learning Module

This module implements a contrastive learning objective that aligns agent embeddings
from the same batch while pushing apart embeddings from different batches.

The module uses a sigmoid loss formulation (similar to SigLIP) where agents from the
same batch are treated as positive pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.architecture_utils import build_mlp


class CrossEgoContrastive(nn.Module):
    """
    Contrastive learning module for multi-agent video embeddings.
    
    Given agent embeddings [B, A, embed_dim], this module:
    1. Normalizes embeddings
    2. Computes pairwise cosine similarities
    3. Applies sigmoid loss with block-diagonal label matrix
    
    The label matrix has 1s for agents from the same batch, 0s otherwise.
    For B=2, A=3: agents are ordered as [B1A1, B1A2, B1A3, B2A1, B2A2, B2A3]
    
    Label matrix:
        [[1, 1, 1, 0, 0, 0],
         [1, 1, 1, 0, 0, 0],
         [1, 1, 1, 0, 0, 0],
         [0, 0, 0, 1, 1, 1],
         [0, 0, 0, 1, 1, 1],
         [0, 0, 0, 1, 1, 1]]
    """
    
    def __init__(self, embed_dim, proj_dim, init_logit_scale=1.0, init_logit_bias=0.0, 
                 learnable_temp=True, mlp_hidden_dim=None, mlp_dropout=0.1, mlp_activation='gelu'):
        """
        Initialize the cross-ego contrastive module.
        
        Args:
            embed_dim: Input embedding dimension
            proj_dim: Output dimension after MLP projection
            init_logit_scale: Initial value for logit scale (temperature parameter)
            init_logit_bias: Initial bias for logits
            learnable_temp: Whether logit_scale and logit_bias are learnable
            mlp_hidden_dim: Hidden dimension for MLP projector (defaults to embed_dim)
            mlp_dropout: Dropout probability for MLP
            mlp_activation: Activation function for MLP
        """
        super().__init__()
        
        # MLP projector (2-layer: input -> hidden -> output)
        if mlp_hidden_dim is None:
            mlp_hidden_dim = embed_dim
        
        self.proj_dim = proj_dim
        self.projector = build_mlp(
            input_dim=embed_dim,
            output_dim=proj_dim,
            num_hidden_layers=1,  # 1 hidden layer = 2 layer MLP
            hidden_dim=mlp_hidden_dim,
            dropout=mlp_dropout,
            activation=mlp_activation
        )
        
        # Learnable temperature and bias parameters (following SigLIP)
        self.logit_scale = nn.Parameter(
            torch.tensor(init_logit_scale).log(),
            requires_grad=learnable_temp
        )
        self.logit_bias = nn.Parameter(
            torch.tensor(init_logit_bias),
            requires_grad=learnable_temp
        )
    
    def create_label_matrix(self, batch_size, num_agents, device):
        """
        Create block-diagonal label matrix for contrastive learning.
        
        Args:
            batch_size: Number of batches (B)
            num_agents: Number of agents per batch (A)
            device: Device to create tensor on
            
        Returns:
            Label matrix of shape [B*A, B*A] with block-diagonal structure
        """
        # Create block-diagonal matrix where each block is all ones
        labels = torch.zeros(batch_size * num_agents, batch_size * num_agents, device=device)
        
        for b in range(batch_size):
            start_idx = b * num_agents
            end_idx = (b + 1) * num_agents
            labels[start_idx:end_idx, start_idx:end_idx] = 1.0
        
        return labels
    
    def forward(self, agent_embeddings, return_loss=True):
        """
        Forward pass through contrastive module.
        
        Args:
            agent_embeddings: Agent embeddings of shape [B, A, embed_dim]
            return_loss: Whether to compute and return the contrastive loss
            
        Returns:
            If return_loss=True:
                Dictionary containing:
                    - embeddings: Normalized embeddings [B, A, proj_dim] 
                    - loss: Contrastive loss scalar
                    - logits: Similarity matrix [B*A, B*A]
            If return_loss=False:
                Dictionary containing:
                    - embeddings: Normalized embeddings [B, A, proj_dim]
        """
        B, A, embed_dim = agent_embeddings.shape
        
        # Flatten batch and agent dimensions: [B, A, embed_dim] -> [B*A, embed_dim]
        flat_embeddings = agent_embeddings.view(B * A, embed_dim)
        
        # Project through MLP: [B*A, embed_dim] -> [B*A, proj_dim]
        projected_embeddings = self.projector(flat_embeddings)
        
        # L2 normalize embeddings
        normalized_embeddings = projected_embeddings / projected_embeddings.norm(p=2, dim=-1, keepdim=True)
        
        if not return_loss:
            # Just return normalized embeddings reshaped back
            return {
                'embeddings': normalized_embeddings.view(B, A, self.proj_dim)
            }
        
        # Compute cosine similarity matrix: [B*A, B*A]
        logits = torch.matmul(normalized_embeddings, normalized_embeddings.t())
        
        # Apply learnable temperature and bias
        logit_scale = self.logit_scale.exp()
        logits = logits * logit_scale + self.logit_bias
        
        # Create label matrix: 1 for same batch, 0 for different batch
        labels = self.create_label_matrix(B, A, device=agent_embeddings.device)
        
        # Compute sigmoid loss
        # For label=1 (positive pairs): logsigmoid(logit)
        # For label=0 (negative pairs): logsigmoid(-logit)
        # This can be written as: logsigmoid((2*label - 1) * logit)
        m1_diag1 = 2 * labels - 1  # Converts {0,1} to {-1,1}
        loglik = F.logsigmoid(m1_diag1 * logits)
        
        # Sum over all pairs and average over instances
        nll = -torch.sum(loglik, dim=-1)
        loss = nll.mean()
        
        # Reshape embeddings back to [B, A, proj_dim]
        output = {
            'embeddings': normalized_embeddings.view(B, A, self.proj_dim),
            'loss': loss,
            'logits': logits
        }
        
        return output
    
    def get_temperature(self):
        """Get current temperature value."""
        return self.logit_scale.exp().item()

