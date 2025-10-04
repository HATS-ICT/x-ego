"""
Label creation utilities for multi-agent enemy location prediction.

Supports 6 different label formulations:
1. multi-label-cls: Binary vector of occupied locations
2. multi-output-reg: Integer count of agents per location  
3. grid-cls: Binary vector of occupied grid cells
4. density-cls: Smoothed density distribution over grid cells
5. coord-reg: Direct 3D coordinate regression
6. coord-gen: Same as coord-reg but used with VAE/coord-gen models
"""

import torch
import numpy as np
from typing import List, Dict
from scipy.ndimage import gaussian_filter


class LabelCreatorBase:
    """Base class for label creators."""
    
    def __init__(self, cfg: Dict):
        """
        Initialize label creator with configuration.
        
        Args:
            cfg: Configuration dictionary containing task parameters
        """
        self.cfg = cfg
        self.task_form = cfg.data.task_form
    
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
    
    Directly regresses the (x, y, z) normalized coordinates of each agent.
    Output shape: [5, 3] for 5 agents with X_norm, Y_norm, Z_norm coordinates
    """
    
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.num_target_agents = cfg.model.num_target_agents
    
    def create_labels(self, enemy_players: List[Dict]) -> torch.Tensor:
        """Create coordinate regression labels using normalized coordinates."""
        coords = []
        for i in range(self.num_target_agents):
            if i < len(enemy_players):
                player = enemy_players[i]
                coords.append([float(player['X_norm']), float(player['Y_norm']), float(player['Z_norm'])])
            else:
                # Pad with zeros if not enough enemy players
                coords.append([0.0, 0.0, 0.0])
        
        coords = np.array(coords)
        return torch.tensor(coords, dtype=torch.float32)


class MultiLabelClsLabelCreator(LabelCreatorBase):
    """
    Multi-label classification label creator.
    
    Predicts which locations (places) are occupied.
    Output is a binary vector over all possible locations, where 1 means 
    at least one agent is present.
    Output shape: [num_places]
    """
    
    def __init__(self, cfg: Dict, place_to_idx: Dict[str, int], num_places: int):
        super().__init__(cfg)
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
    
    def __init__(self, cfg: Dict, place_to_idx: Dict[str, int], num_places: int):
        super().__init__(cfg)
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
    
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.grid_resolution = cfg.data.grid_resolution
        self.output_dim = self.grid_resolution * self.grid_resolution
    
    def create_labels(self, enemy_players: List[Dict]) -> torch.Tensor:
        """Create grid classification labels using normalized coordinates."""
        grid = np.zeros((self.grid_resolution, self.grid_resolution), dtype=np.float32)
        
        for player in enemy_players:
            try:
                # Use pre-normalized coordinates directly (already in [0, 1] range)
                x_norm = float(player['X_norm'])
                y_norm = float(player['Y_norm'])
                
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
    
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.grid_resolution = cfg.data.grid_resolution
        self.gaussian_sigma = cfg.data.gaussian_sigma
        self.output_dim = self.grid_resolution * self.grid_resolution
    
    def create_labels(self, enemy_players: List[Dict]) -> torch.Tensor:
        """Create density classification labels with Gaussian smoothing using normalized coordinates."""
        grid = np.zeros((self.grid_resolution, self.grid_resolution), dtype=np.float32)
        
        for player in enemy_players:
            try:
                # Use pre-normalized coordinates directly (already in [0, 1] range)
                x_norm = float(player['X_norm'])
                y_norm = float(player['Y_norm'])
                
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


def create_label_creator(cfg: Dict, **kwargs) -> LabelCreatorBase:
    """
    Factory function to create appropriate label creator based on task form.
    
    Args:
        cfg: Configuration dictionary
        **kwargs: Additional arguments like place_to_idx, num_places
        
    Returns:
        Label creator instance
    """
    task_form = cfg.data.task_form
    
    if task_form == 'coord-reg':
        return CoordRegLabelCreator(cfg)
    elif task_form == 'coord-gen':
        # coord-gen uses same labels as coord-reg
        return CoordRegLabelCreator(cfg)
    elif task_form == 'multi-label-cls':
        return MultiLabelClsLabelCreator(
            cfg,
            place_to_idx=kwargs.get('place_to_idx'),
            num_places=kwargs.get('num_places')
        )
    elif task_form == 'multi-output-reg':
        return MultiOutputRegLabelCreator(
            cfg,
            place_to_idx=kwargs.get('place_to_idx'),
            num_places=kwargs.get('num_places')
        )
    elif task_form == 'grid-cls':
        return GridClsLabelCreator(cfg)
    elif task_form == 'density-cls':
        return DensityClsLabelCreator(cfg)
    else:
        raise ValueError(f"Unknown task_form: {task_form}")
