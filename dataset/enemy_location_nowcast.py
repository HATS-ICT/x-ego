import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
import random

try:
    from .base import BaseVideoDataset
except ImportError:
    from base import BaseVideoDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def enemy_location_nowcast_collate_fn(batch):
    """
    Custom collate function for multi-agent enemy location prediction dataset.
    
    Args:
        batch: List of dictionaries containing:
            - 'video': Video features tensor [A, T, C, H, W] where A is num_agents
            - 'enemy_locations': Enemy location labels (regression or classification)
            - 'pov_team_side': String indicating team side
            - 'agent_ids': List of agent IDs used
    
    Returns:
        Dictionary with batched tensors and lists of string values
    """
    collated = {}
    
    # Handle each key appropriately
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        
        if key in ['pov_team_side', 'agent_ids']:
            # Keep string/list values as lists
            collated[key] = values
        else:
            # For tensors (video, enemy_locations, pov_team_side_encoded), use default collate
            collated[key] = torch.utils.data.default_collate(values)
    
    return collated


class EnemyLocationNowcastDataset(BaseVideoDataset, Dataset):
    """
    Multi-agent enemy location prediction dataset.
    
    Given videos from a subset of players from one team, predict the locations
    of all 5 players from the enemy team.
    """
    
    def __init__(self, cfg: Dict):
        """
        Initialize Multi-Agent Enemy Location Prediction Dataset.
        
        Args:
            cfg: Configuration dictionary containing dataset parameters
        """
        # Initialize base class - handles all common initialization
        super().__init__(cfg)
            
        logger.info(f"Dataset initialized with {len(self.df)} samples")
        logger.info(f"Number of agents: {self.num_agents}, Task form: {self.task_form}")
        logger.info(f"Minimap masking enabled: {self.mask_minimap}")
        if self.task_form in ['grid-cls', 'density-cls']:
            grid_res = self.cfg.data.grid_resolution
            logger.info(f"Grid resolution: {grid_res}x{grid_res} = {grid_res*grid_res} cells")
            if self.task_form == 'density-cls':
                logger.info(f"Gaussian sigma: {self.cfg.data.gaussian_sigma}")
        logger.info("Team side will be randomly selected for each sample")
    
    def _extract_unique_places(self) -> List[str]:
        """Extract unique place names from the dataset."""
        places = set()
        
        # Extract places from all player columns
        for i in range(10):  # Assuming 10 players (0-9)
            place_col = f'player_{i}_place'
            if place_col in self.df.columns:
                places.update(self.df[place_col].unique())
        
        # Remove empty/null values
        places = {p for p in places if p and str(p).strip() and str(p) != 'nan'}
        
        return sorted(list(places))
    
    def _get_scaler_path(self) -> Path:
        """Get the path for saving/loading the coordinate scaler."""
        return self.output_dir / "coordinate_minmax_scaler.pkl"
    
    def _get_coordinate_columns_for_scaler(self) -> List[Tuple[str, str, str]]:
        """Get list of (X, Y, Z) column name tuples for fitting the coordinate scaler."""
        coord_columns = []
        # For scaler fitting, we need to consider both teams as potential enemies
        for i in range(10):  # 10 players total
            x_col = f'player_{i}_X'
            y_col = f'player_{i}_Y'
            z_col = f'player_{i}_Z'
            coord_columns.append((x_col, y_col, z_col))
        return coord_columns
    
    def _get_player_data(self, row: pd.Series, player_idx: int) -> Dict:
        """Extract player data from a row."""
        return {
            'id': row[f'player_{player_idx}_id'],
            'X': row[f'player_{player_idx}_X'],
            'Y': row[f'player_{player_idx}_Y'],
            'Z': row[f'player_{player_idx}_Z'],
            'side': row[f'player_{player_idx}_side'],
            'place': row[f'player_{player_idx}_place'],
            'name': row[f'player_{player_idx}_name']
        }
    
    def _get_team_players(self, row: pd.Series, pov_team_side: str) -> List[Dict]:
        """Get all players from a specific team side."""
        team_players = []
        
        for i in range(10):  # Assuming 10 players (0-9)
            try:
                player_data = self._get_player_data(row, i)
                if player_data['side'] == pov_team_side.lower():  # sides are lowercase in data
                    team_players.append(player_data)
            except KeyError:
                # Player column doesn't exist, skip
                continue
        
        return team_players
    
    def _create_enemy_location_labels(self, enemy_players: List[Dict]) -> torch.Tensor:
        """
        Create enemy location labels based on task_form.
        
        Args:
            enemy_players: List of enemy player data dictionaries
            
        Returns:
            labels: Tensor containing enemy location labels
        """
        return self.label_creator.create_labels(enemy_players)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample containing multi-agent video data and enemy location labels.
        
        Returns:
            dict: Dictionary containing:
                - 'video': Multi-agent video features tensor [A, T, C, H, W]
                - 'enemy_locations': Enemy location labels
                - 'pov_team_side': Team side used for input (string)
                - 'pov_team_side_encoded': Team side encoded as boolean (0 for T, 1 for CT)
                - 'agent_ids': List of agent IDs used
        """
        # Get sample information
        row = self.df.iloc[idx]
        start_seconds = row['normalized_start_seconds']
        end_seconds = row['normalized_end_seconds']
        match_id = row['match_id']
        round_num = row['round_num']
        
        # Randomly select team side for this sample
        selected_pov_team_side = random.choice(['CT', 'T'])
        enemy_pov_team_side = 'T' if selected_pov_team_side == 'CT' else 'CT'
        
        # Get team players (input) and enemy players (labels)
        team_players = self._get_team_players(row, selected_pov_team_side)
        enemy_players = self._get_team_players(row, enemy_pov_team_side)
        
        # Select subset of agents from team
        selected_agents = self._select_agents(team_players)
        
        # Load videos for selected agents
        agent_videos = []
        agent_ids = []
        
        for agent in selected_agents:
            # Construct video path dynamically
            video_path = self._construct_video_path(match_id, agent['id'], round_num)
            video_clip = self._load_video_clip_with_decord(video_path, start_seconds, end_seconds)
            video_features = self._transform_video(video_clip)
            agent_videos.append(video_features)
            agent_ids.append(agent['id'])
        
        # Stack agent videos: [A, T, C, H, W]
        multi_agent_video = torch.stack(agent_videos, dim=0)
        
        # Create enemy location labels
        enemy_location_labels = self._create_enemy_location_labels(enemy_players)
        
        # Encode team side as boolean for model input (T=0, CT=1)
        pov_team_side_encoded = 1 if selected_pov_team_side == 'CT' else 0
        
        # Prepare return dictionary
        result = {
            'video': multi_agent_video,
            'enemy_locations': enemy_location_labels,
            'pov_team_side': selected_pov_team_side,
            'pov_team_side_encoded': torch.tensor(pov_team_side_encoded, dtype=torch.long),
            'agent_ids': agent_ids
        }
        
        return result


if __name__ == "__main__":
    # Test the dataset
    print("EnemyLocationNowcastDataset test placeholder")
