"""
Coordinate Scaling Utilities

This module provides utilities for unscaling normalized coordinates
back to original coordinate space.
"""

import torch
import pickle
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=2)
def load_scaler(scaler_path):
    """
    Load scaler from file with caching to avoid repeated disk I/O.
    
    Args:
        scaler_path: Path to the pickled scaler file (must be hashable, use str)
        
    Returns:
        Loaded scaler object
    """
    scaler_path = Path(scaler_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler


def unscale_coordinates(scaled_coords, scaler_path):
    """
    Unscale normalized coordinates back to original coordinate space.
    
    Args:
        scaled_coords: Normalized coordinates in [0, 1] range (tensor or numpy array)
        scaler_path: Path to the pickled scaler file
        
    Returns:
        Unscaled coordinates in same format as input
    """
    # Load scaler (cached)
    scaler = load_scaler(str(scaler_path))
    
    # Convert to numpy
    if torch.is_tensor(scaled_coords):
        coords_np = scaled_coords.detach().cpu().float().numpy()
        device = scaled_coords.device
        was_tensor = True
    else:
        coords_np = scaled_coords
        device = None
        was_tensor = False
    
    # Unscale
    original_shape = coords_np.shape
    coords_flat = coords_np.reshape(-1, 3)
    
    # Only unscale non-zero coordinates (padding coordinates remain [0, 0, 0])
    mask = coords_flat.sum(axis=1) != 0
    if mask.any():
        coords_flat[mask] = scaler.inverse_transform(coords_flat[mask])
    
    unscaled_np = coords_flat.reshape(original_shape)
    
    # Convert back to tensor if needed
    if was_tensor:
        return torch.tensor(unscaled_np, dtype=torch.float32, device=device)
    else:
        return unscaled_np
