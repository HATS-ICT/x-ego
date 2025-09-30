"""
Label creation utilities for multi-agent enemy location prediction.

Supports 6 different label formulations:
1. multi-label-cls: Binary vector of occupied locations
2. multi-output-reg: Integer count of agents per location  
3. grid-cls: Binary vector of occupied grid cells
4. density-cls: Smoothed density distribution over grid cells
5. coord-reg: Direct 3D coordinate regression
6. generative: Same as coord-reg but used with VAE/generative models
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler


class LabelCreatorBase:
    """Base class for label creators."""
    
    def __init__(self, config: Dict):
        """
        Initialize label creator with configuration.
        
        Args:
            config: Configuration dictionary containing task parameters
        """
        self.config = config
        self.task_form = config['data']['task_form']
    
    def create_labels(self, enemy_players: List[Dict]) -> torch.Tensor:
        """
        Create labels for enemy players based on task form.
        
        Args:
            enemy_players: List of enemy player data dictionaries
            
        Returns:
            labels: Tensor containing enemy location labels
        """
        raise NotImplementedError("Subclasses must implement create_labels")


class CoordRegLabelCreator(LabelCreatorBase):
    """
    Coordinate regression label creator.
    
    Directly regresses the (x, y, z) coordinates of each agent.
    Output shape: [5, 3] for 5 agents with X, Y, Z coordinates
    """
    
    def __init__(self, config: Dict, coordinate_scaler: MinMaxScaler = None):
        super().__init__(config)
        self.coordinate_scaler = coordinate_scaler
        self.num_target_agents = config['model']['num_target_agents']
    
    def create_labels(self, enemy_players: List[Dict]) -> torch.Tensor:
        """Create coordinate regression labels."""
        coords = []
        for i in range(self.num_target_agents):
            if i < len(enemy_players):
                player = enemy_players[i]
                coords.append([float(player['X']), float(player['Y']), float(player['Z'])])
            else:
                # Pad with zeros if not enough enemy players
                coords.append([0.0, 0.0, 0.0])
        
        coords = np.array(coords)
        
        # Scale coordinates if scaler is available
        if self.coordinate_scaler is not None:
            coords = self._scale_coordinates(coords)
        
        return torch.tensor(coords, dtype=torch.float32)
    
    def _scale_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """Scale coordinates using the fitted scaler."""
        original_shape = coords.shape
        coords_flat = coords.reshape(-1, 3)
        
        # Only scale non-zero coordinates (padding coordinates remain [0, 0, 0])
        mask = np.any(coords_flat != 0, axis=1)
        if np.any(mask):
            coords_flat[mask] = self.coordinate_scaler.transform(coords_flat[mask])
        
        return coords_flat.reshape(original_shape)


class MultiLabelClsLabelCreator(LabelCreatorBase):
    """
    Multi-label classification label creator.
    
    Predicts which locations (places) are occupied.
    Output is a binary vector over all possible locations, where 1 means 
    at least one agent is present.
    Output shape: [num_places]
    """
    
    def __init__(self, config: Dict, place_to_idx: Dict[str, int], num_places: int):
        super().__init__(config)
        self.place_to_idx = place_to_idx
        self.num_places = num_places
    
    def create_labels(self, enemy_players: List[Dict]) -> torch.Tensor:
        """Create multi-label classification labels."""
        place_binary = torch.zeros(self.num_places, dtype=torch.float32)
        
        for player in enemy_players:
            place = player['place']
            if place in self.place_to_idx:
                place_idx = self.place_to_idx[place]
                place_binary[place_idx] = 1.0  # Mark place as occupied
        
        return place_binary


class MultiOutputRegLabelCreator(LabelCreatorBase):
    """
    Multi-output regression label creator.
    
    Predicts the number of agents in each location.
    Output is an integer-valued vector over locations, giving the agent 
    count per location.
    Output shape: [num_places]
    """
    
    def __init__(self, config: Dict, place_to_idx: Dict[str, int], num_places: int):
        super().__init__(config)
        self.place_to_idx = place_to_idx
        self.num_places = num_places
    
    def create_labels(self, enemy_players: List[Dict]) -> torch.Tensor:
        """Create multi-output regression labels (count per location)."""
        place_counts = torch.zeros(self.num_places, dtype=torch.float32)
        
        for player in enemy_players:
            place = player['place']
            if place in self.place_to_idx:
                place_idx = self.place_to_idx[place]
                place_counts[place_idx] += 1.0  # Increment count
        
        return place_counts


class GridClsLabelCreator(LabelCreatorBase):
    """
    Grid classification label creator.
    
    Predicts which grid cells contain agents.
    Output is a binary vector over grid cells, where multiple cells can be 
    active simultaneously.
    Output shape: [grid_resolution * grid_resolution]
    """
    
    def __init__(self, config: Dict, coordinate_scaler: MinMaxScaler = None):
        super().__init__(config)
        self.grid_resolution = config['data'].get('grid_resolution', 10)
        self.coordinate_scaler = coordinate_scaler
        self.output_dim = self.grid_resolution * self.grid_resolution
    
    def create_labels(self, enemy_players: List[Dict]) -> torch.Tensor:
        """Create grid classification labels."""
        grid = np.zeros((self.grid_resolution, self.grid_resolution), dtype=np.float32)
        
        for player in enemy_players:
            try:
                x, y = float(player['X']), float(player['Y'])
                
                # Normalize coordinates to [0, 1] range using scaler if available
                if self.coordinate_scaler is not None:
                    coords = np.array([[x, y, 0.0]])  # Z doesn't matter for 2D grid
                    scaled_coords = self.coordinate_scaler.transform(coords)
                    x_norm, y_norm = scaled_coords[0, 0], scaled_coords[0, 1]
                else:
                    # Fall back to simple normalization if no scaler
                    # Assuming CS:GO map bounds approximately [-3000, 3000]
                    x_norm = (x + 3000) / 6000
                    y_norm = (y + 3000) / 6000
                
                # Convert to grid indices
                x_idx = int(np.clip(x_norm * self.grid_resolution, 0, self.grid_resolution - 1))
                y_idx = int(np.clip(y_norm * self.grid_resolution, 0, self.grid_resolution - 1))
                
                # Mark grid cell as occupied (binary)
                grid[y_idx, x_idx] = 1.0
                
            except (ValueError, KeyError):
                continue
        
        # Flatten to 1D vector
        return torch.tensor(grid.flatten(), dtype=torch.float32)


class DensityClsLabelCreator(LabelCreatorBase):
    """
    Density classification label creator.
    
    Predicts a smoothed occupancy distribution over grid cells using a Gaussian kernel.
    Output is a real-valued vector over grid cells, approximating a continuous 
    density map.
    Output shape: [grid_resolution * grid_resolution]
    """
    
    def __init__(self, config: Dict, coordinate_scaler: MinMaxScaler = None):
        super().__init__(config)
        self.grid_resolution = config['data'].get('grid_resolution', 10)
        self.gaussian_sigma = config['data'].get('gaussian_sigma', 1.0)
        self.coordinate_scaler = coordinate_scaler
        self.output_dim = self.grid_resolution * self.grid_resolution
    
    def create_labels(self, enemy_players: List[Dict]) -> torch.Tensor:
        """Create density classification labels with Gaussian smoothing."""
        grid = np.zeros((self.grid_resolution, self.grid_resolution), dtype=np.float32)
        
        for player in enemy_players:
            try:
                x, y = float(player['X']), float(player['Y'])
                
                # Normalize coordinates to [0, 1] range using scaler if available
                if self.coordinate_scaler is not None:
                    coords = np.array([[x, y, 0.0]])  # Z doesn't matter for 2D grid
                    scaled_coords = self.coordinate_scaler.transform(coords)
                    x_norm, y_norm = scaled_coords[0, 0], scaled_coords[0, 1]
                else:
                    # Fall back to simple normalization if no scaler
                    x_norm = (x + 3000) / 6000
                    y_norm = (y + 3000) / 6000
                
                # Convert to grid indices
                x_idx = int(np.clip(x_norm * self.grid_resolution, 0, self.grid_resolution - 1))
                y_idx = int(np.clip(y_norm * self.grid_resolution, 0, self.grid_resolution - 1))
                
                # Add point to grid
                grid[y_idx, x_idx] += 1.0
                
            except (ValueError, KeyError):
                continue
        
        # Apply Gaussian smoothing
        if np.any(grid > 0):
            grid = gaussian_filter(grid, sigma=self.gaussian_sigma)
            # Normalize to sum to number of agents (density distribution)
            grid = grid / (grid.sum() + 1e-8) * len(enemy_players)
        
        # Flatten to 1D vector
        return torch.tensor(grid.flatten(), dtype=torch.float32)


def create_label_creator(config: Dict, **kwargs) -> LabelCreatorBase:
    """
    Factory function to create appropriate label creator based on task form.
    
    Args:
        config: Configuration dictionary
        **kwargs: Additional arguments like coordinate_scaler, place_to_idx, num_places
        
    Returns:
        Label creator instance
    """
    task_form = config['data']['task_form']
    
    if task_form == 'coord-reg':
        return CoordRegLabelCreator(
            config, 
            coordinate_scaler=kwargs.get('coordinate_scaler')
        )
    elif task_form == 'generative':
        # Generative uses same labels as coord-reg
        return CoordRegLabelCreator(
            config,
            coordinate_scaler=kwargs.get('coordinate_scaler')
        )
    elif task_form == 'multi-label-cls':
        return MultiLabelClsLabelCreator(
            config,
            place_to_idx=kwargs.get('place_to_idx'),
            num_places=kwargs.get('num_places')
        )
    elif task_form == 'multi-output-reg':
        return MultiOutputRegLabelCreator(
            config,
            place_to_idx=kwargs.get('place_to_idx'),
            num_places=kwargs.get('num_places')
        )
    elif task_form == 'grid-cls':
        return GridClsLabelCreator(
            config,
            coordinate_scaler=kwargs.get('coordinate_scaler')
        )
    elif task_form == 'density-cls':
        return DensityClsLabelCreator(
            config,
            coordinate_scaler=kwargs.get('coordinate_scaler')
        )
    else:
        raise ValueError(f"Unknown task_form: {task_form}")
