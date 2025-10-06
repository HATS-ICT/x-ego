"""
Coordinate Scaling Utilities

This module provides utilities for unscaling normalized coordinates
back to original coordinate space.
"""

import torch
import joblib
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
    scaler = joblib.load(scaler_path)
    return scaler


def unscale_coordinates(scaled_coords, scaler_path):
    """
    Unscale normalized coordinates back to original coordinate space.
    
    Args:
        scaled_coords: Normalized coordinates in [0, 1] range (tensor or numpy array)
                      Shape: [..., 2] for X,Y coordinates
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
    
    # Unscale (only X,Y coordinates, ignore Z from scaler)
    original_shape = coords_np.shape
    coords_flat = coords_np.reshape(-1, 2)
    
    # Only unscale non-zero coordinates (padding coordinates remain [0, 0])
    mask = coords_flat.sum(axis=1) != 0
    if mask.any():
        # Add dummy Z coordinate for scaler (which expects 3D), then extract X,Y
        coords_3d = np.zeros((coords_flat.shape[0], 3))
        coords_3d[:, :2] = coords_flat
        coords_3d_unscaled = scaler.inverse_transform(coords_3d)
        coords_flat[mask] = coords_3d_unscaled[mask, :2]  # Only take X,Y
    
    unscaled_np = coords_flat.reshape(original_shape)
    
    # Convert back to tensor if needed
    if was_tensor:
        return torch.tensor(unscaled_np, dtype=torch.float32, device=device)
    else:
        return unscaled_np
