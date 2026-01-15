"""
Label creation utilities for multi-agent enemy location prediction.

Supports multi-label classification: Binary vector of occupied locations.
"""

import torch
from typing import List, Dict


class LabelCreatorBase:
    """Base class for label creators."""
    
    def __init__(self, cfg: Dict):
        """
        Initialize label creator with configuration.
        
        Args:
            cfg: Configuration dictionary containing task parameters
        """
        self.cfg = cfg
    
    def create_labels(self, enemy_players: List[Dict]) -> torch.Tensor:
        """
        Create labels for enemy players based on task form.
        
        Args:
            enemy_players: List of enemy player data dictionaries
            
        Returns:
            labels: Tensor containing enemy location labels
        """
        raise NotImplementedError("Subclasses must implement create_labels")


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


def create_label_creator(cfg: Dict, **kwargs) -> LabelCreatorBase:
    """
    Factory function to create appropriate label creator based on task form.
    
    Args:
        cfg: Configuration dictionary
        **kwargs: Additional arguments like place_to_idx, num_places
        
    Returns:
        Label creator instance
    """
    return MultiLabelClsLabelCreator(
        cfg,
        place_to_idx=kwargs['place_to_idx'],
        num_places=kwargs['num_places']
    )
