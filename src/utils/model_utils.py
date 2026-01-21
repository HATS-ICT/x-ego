"""
Model utilities for X-EGO project.
Provides model components and loading functions.
"""

import torch
import torch.nn as nn
from models.ctfm_contrastive import CTFMContrastive


class SelectIndex(nn.Module):
    """Module that selects a specific index from a sequence."""
    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        return x[:, self.index, :]


def load_model_from_checkpoint(cfg, checkpoint_path: str):
    """Load model from checkpoint
    
    Args:
        cfg: Model configuration
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Loaded model in evaluation mode
    """
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Create model with cfg
    model = CTFMContrastive(cfg)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    return model