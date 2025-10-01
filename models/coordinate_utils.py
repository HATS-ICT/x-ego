"""
Coordinate Scaling Utilities

This module provides utilities for scaling and unscaling coordinates
in multi-agent location prediction tasks.
"""

import torch


class CoordinateScalerMixin:
    """Mixin class for coordinate scaling operations."""
    
    def set_coordinate_scaler(self, scaler):
        """
        Set the coordinate scaler for coordinate-based tasks.
        
        Args:
            scaler: sklearn scaler object (e.g., MinMaxScaler)
        """
        if hasattr(self, 'task_form') and self.task_form in ['coord-reg', 'generative', 'grid-cls', 'density-cls']:
            self.coordinate_scaler = scaler
            if scaler is not None:
                print(f"Set coordinate scaler with data_min_: {scaler.data_min_}, "
                      f"data_max_: {scaler.data_max_}, scale_: {scaler.scale_}")
            else:
                print("Coordinate scaler set to None")
    
    def unscale_coordinates(self, scaled_coords):
        """
        Unscale coordinates back to original coordinate space.
        
        Args:
            scaled_coords: Scaled coordinates (tensor or numpy array)
            
        Returns:
            Unscaled coordinates in same format as input
        """
        if hasattr(self, 'task_form') and self.task_form in ['coord-reg', 'generative']:
            assert self.coordinate_scaler is not None, \
                "Coordinate scaler must be set for coordinate regression modes"
        
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
        unscaled_flat = self.coordinate_scaler.inverse_transform(coords_flat)
        unscaled_np = unscaled_flat.reshape(original_shape)
        
        # Convert back to tensor if needed
        if was_tensor:
            return torch.tensor(unscaled_np, dtype=torch.float32, device=device)
        else:
            return unscaled_np
