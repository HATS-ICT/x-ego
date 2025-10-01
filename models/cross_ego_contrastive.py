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
    
    def __init__(self, embed_dim, init_logit_scale=1.0, init_logit_bias=0.0, 
                 learnable_temp=True, mlp_hidden_dim=None, mlp_dropout=0.1, mlp_activation='gelu'):
        """
        Initialize the cross-ego contrastive module.
        
        Args:
            embed_dim: Input embedding dimension
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
        
        self.projector = build_mlp(
            input_dim=embed_dim,
            output_dim=embed_dim,
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
    
    def compute_retrieval_metrics(self, logits, labels):
        """
        Compute retrieval accuracy metrics from similarity logits.
        
        For each query agent, there are A-1 positive agents (all other agents from same batch).
        Metrics measure whether the top-K retrieved agents include any of these positives.
        
        Example: B=2 batches, A=3 agents per batch
        - Query B1A1 has positives: [B1A2, B1A3] (2 positives)
        - Top-1 accuracy: % of queries where rank-1 retrieval is a same-batch agent
        - Top-K accuracy: % of queries where at least one of top-K is a same-batch agent
        
        Args:
            logits: Similarity matrix [N, N] where N = B*A
            labels: Label matrix [N, N] with 1 for same batch, 0 otherwise (block-diagonal)
            
        Returns:
            Dictionary of retrieval metrics:
                - top1_acc: Top-1 retrieval accuracy
                - top3_acc: Top-3 retrieval accuracy  
                - top5_acc: Top-5 retrieval accuracy
                - mrr: Mean reciprocal rank
        """
        N = logits.shape[0]
        
        # Mask out self-similarity (diagonal) - we don't want to retrieve ourselves
        mask = torch.eye(N, device=logits.device, dtype=torch.bool)
        masked_logits = logits.clone()
        masked_logits[mask] = float('-inf')
        
        # Get top-k indices for each query
        k_max = min(5, N - 1)  # Can't retrieve more than N-1 (excluding self)
        _, top_k_indices = torch.topk(masked_logits, k=k_max, dim=1)
        
        # Get ground truth labels for each query (excluding self)
        # Each query has A-1 positives (all same-batch agents except self)
        gt_labels = labels.clone()
        gt_labels[mask] = 0  # Remove self from positive pairs
        
        # Calculate Top-K accuracies
        metrics = {}
        
        for k in [1, 3, 5]:
            if k > k_max:
                continue
                
            # For each query, check if any of top-k retrievals are positive
            top_k_idx = top_k_indices[:, :k]  # [N, k]
            
            # Gather labels for top-k retrievals
            batch_indices = torch.arange(N, device=logits.device).unsqueeze(1).expand(-1, k)
            top_k_labels = gt_labels[batch_indices, top_k_idx]  # [N, k]
            
            # Check if at least one positive in top-k
            has_positive = (top_k_labels.sum(dim=1) > 0).float()
            metrics[f'top{k}_acc'] = has_positive.mean()
        
        # Calculate Mean Reciprocal Rank (MRR)
        # For each query, find rank of first positive
        reciprocal_ranks = []
        for i in range(N):
            # Get labels for retrieved items in order
            retrieved_labels = gt_labels[i, top_k_indices[i]]
            # Find first positive
            positive_mask = retrieved_labels > 0
            if positive_mask.any():
                first_positive_rank = positive_mask.int().argmax().item() + 1
                reciprocal_ranks.append(1.0 / first_positive_rank)
            else:
                reciprocal_ranks.append(0.0)
        
        metrics['mrr'] = torch.tensor(reciprocal_ranks, device=logits.device).mean()
        
        return metrics
    
    def forward(self, agent_embeddings):
        """
        Forward pass through contrastive module.
        
        Args:
            agent_embeddings: Agent embeddings of shape [B, A, embed_dim]
            
        Returns:
            Dictionary containing:
                - embeddings: Normalized embeddings [B, A, proj_dim] 
                - loss: Contrastive loss scalar
                - logits: Similarity matrix [B*A, B*A]
                - retrieval_metrics: Dict of retrieval accuracy metrics

        """
        B, A, embed_dim = agent_embeddings.shape
        
        # Flatten batch and agent dimensions: [B, A, embed_dim] -> [B*A, embed_dim]
        flat_embeddings = agent_embeddings.view(B * A, embed_dim)
        
        # Project through MLP: [B*A, embed_dim] -> [B*A, proj_dim]
        projected_embeddings = self.projector(flat_embeddings)
        
        # L2 normalize embeddings
        normalized_embeddings = projected_embeddings / projected_embeddings.norm(p=2, dim=-1, keepdim=True)
        
        
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
        
        # Compute retrieval metrics
        retrieval_metrics = self.compute_retrieval_metrics(logits, labels)
        
        # Compute binary classification accuracy
        # Sigmoid loss treats each pair as binary classification: same batch (1) or not (0)
        pred_probs = torch.sigmoid(logits)
        pred_binary = (pred_probs > 0.5).float()
        binary_acc = (pred_binary == labels).float().mean()
        
        # Add to metrics
        retrieval_metrics['binary_acc'] = binary_acc
        
        # Calculate positive and negative pair accuracies separately
        pos_mask = labels == 1
        neg_mask = labels == 0
        if pos_mask.any():
            pos_acc = (pred_binary[pos_mask] == labels[pos_mask]).float().mean()
            retrieval_metrics['pos_pair_acc'] = pos_acc
        if neg_mask.any():
            neg_acc = (pred_binary[neg_mask] == labels[neg_mask]).float().mean()
            retrieval_metrics['neg_pair_acc'] = neg_acc
        
        # Reshape embeddings back to [B, A, proj_dim]
        output = {
            'embeddings': normalized_embeddings.view(B, A, embed_dim),
            'loss': loss,
            'logits': logits,
            'retrieval_metrics': retrieval_metrics,
            'temperature': logit_scale
        }
        
        return output
    
    def get_temperature(self):
        """Get current temperature value."""
        return self.logit_scale.exp().item()

