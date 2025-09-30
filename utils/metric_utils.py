import numpy as np
import torch
import torch.nn.functional as F
import logging
import os


def kl_divergence_histogram(pred_probs, target_counts, n_agents=5):
    """
    Calculate KL divergence between predicted probability distribution and empirical distribution.
    
    Args:
        pred_probs: [B, K] predicted probabilities (should sum to 1.0)
        target_counts: [B, K] target counts (integers that sum to n_agents)
        n_agents: total number of agents (default: 5)
    
    Returns:
        kl_div: KL divergence KL(target_dist || pred_dist)
    """
    # Convert counts to empirical distributions (normalize by n_agents)
    target_probs = target_counts / n_agents  # [B, K]
    
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    pred_probs_safe = np.clip(pred_probs, eps, 1.0)
    target_probs_safe = np.clip(target_probs, eps, 1.0)
    
    # KL(q || p) where q is target distribution, p is predicted distribution
    kl_div = np.mean(np.sum(target_probs_safe * np.log(target_probs_safe / pred_probs_safe), axis=1))
    return kl_div


def multinomial_loss(logits, counts):
    """
    Multinomial loss for histogram prediction.
    
    Args:
        logits: [B, K] model outputs before softmax
        counts: [B, K] integer counts per area, sum across K = N_agents (5)
    
    Returns:
        loss: scalar loss value
    """
    log_probs = F.log_softmax(logits, dim=-1)    # [B, K]
    loss = -(counts * log_probs).sum(dim=-1)     # [B]
    return loss.mean()

def exact_match_accuracy(pred_counts, target_counts):
    """
    Calculate exact match accuracy for histogram predictions.
    
    Args:
        pred_counts: [B, K] predicted counts (continuous values)
        target_counts: [B, K] target counts (integer values)
    
    Returns:
        accuracy: fraction of samples where all place counts are exactly correct
    """
    exact_matches = np.sum(np.round(pred_counts) == target_counts, axis=1) == target_counts.shape[1]
    return np.mean(exact_matches)

def l1_count_error(pred_counts, target_counts):
    """
    Calculate L1 count error between predicted and target histograms.
    
    Args:
        pred_counts: [B, K] predicted counts (continuous values)
        target_counts: [B, K] target counts (integer values)
    
    Returns:
        l1_error: average L1 distance between predicted and actual counts
    """
    return np.mean(np.sum(np.abs(pred_counts - target_counts), axis=1))


def chamfer_distance_batch(X, Y):
    """
    Batched Chamfer Distance between two sets of point clouds.

    X: (B, N, d) tensor  (batch of B point clouds with N points each)
    Y: (B, M, d) tensor  (batch of B point clouds with M points each)
    Returns: (B,) tensor of Chamfer Distances for each batch
    """
    # Compute pairwise distances
    # X: (B, N, d) -> (B, N, 1, d)
    # Y: (B, M, d) -> (B, 1, M, d)
    diff = X[:, :, None, :] - Y[:, None, :, :]   # (B, N, M, d)
    dist = torch.sum(diff ** 2, dim=3)           # (B, N, M)
    # For each x in X, find nearest y in Y
    min_dist_XY = torch.min(dist, dim=2)[0]   # (B, N)
    # For each y in Y, find nearest x in X
    min_dist_YX = torch.min(dist, dim=1)[0]   # (B, M)
    # Mean over points, then sum both directions
    cd = torch.mean(min_dist_XY, dim=1) + torch.mean(min_dist_YX, dim=1)  # (B,)
    return cd



