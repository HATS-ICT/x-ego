"""
Serialization utilities for X-EGO project.
Handles conversion of various data types for JSON and logging.
"""

import numpy as np
from omegaconf import OmegaConf


def json_serializable(obj):
    """Convert numpy types and OmegaConf objects to JSON serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '_metadata') and hasattr(obj, '_content'):  # OmegaConf DictConfig/ListConfig
        return OmegaConf.to_container(obj, resolve=True)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def convert_numpy_to_python(obj):
    """
    Recursively convert numpy arrays to Python lists for JSON serialization.
    
    Args:
        obj: Object that may contain numpy arrays
        
    Returns:
        Object with numpy arrays converted to Python lists
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python(item) for item in obj)
    else:
        return obj
