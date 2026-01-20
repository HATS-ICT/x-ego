import logging
from typing import Dict, List
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
            - 'agent_places': List of agent places (locations)
            - 'time': Normalized prediction seconds (optional)
    
    Returns:
        Dictionary with batched tensors and lists of string values
    """
    collated = {}
    
    # Handle each key appropriately
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        
        if key in ['pov_team_side', 'agent_ids', 'agent_places']:
            # Keep string/list values as lists
            collated[key] = values
        elif key == 'time':
            # For time, convert to tensor
            collated[key] = torch.tensor(values, dtype=torch.float32)
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
        
        # Check if we should return time information
        self.return_time = cfg.data.get('return_time', False)
            
        logger.info(f"Dataset initialized with {len(self.df)} samples")
        logger.info(f"Number of agents: {self.num_agents}")
        logger.info(f"Minimap masking enabled: {self.mask_minimap}")
        logger.info(f"Return time: {self.return_time}")
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
    
    def _get_player_data(self, row: pd.Series, player_idx: int) -> Dict:
        """Extract player data from a row."""
        return {
            'id': row[f'player_{player_idx}_id'],
            'X_norm': row[f'player_{player_idx}_X'],
            'Y_norm': row[f'player_{player_idx}_Y'],
            'Z_norm': row[f'player_{player_idx}_Z'],
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
        Create enemy location labels (multi-hot classification).
        
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
                - 'time': Normalized prediction seconds (if return_time is enabled)
        """
        # Get sample information
        row = self.df.iloc[idx]
        start_seconds = row['normalized_start_seconds']
        end_seconds = row['normalized_end_seconds']
        match_id = row['match_id']
        round_num = row['round_num']
        
        # Get original CSV index for loading pre-computed embeddings
        original_csv_idx = row.get('original_csv_idx', idx)
        
        # Randomly select team side for this sample
        selected_pov_team_side = random.choice(['CT', 'T'])
        enemy_pov_team_side = 'T' if selected_pov_team_side == 'CT' else 'CT'
        
        # Get team players (input) and enemy players (labels)
        team_players = self._get_team_players(row, selected_pov_team_side)
        enemy_players = self._get_team_players(row, enemy_pov_team_side)
        
        # Select subset of agents from team
        selected_agents = self._select_agents(team_players)
        
        # Load videos or embeddings for selected agents
        agent_videos = []
        agent_ids = []
        agent_places = []
        
        if self.use_precomputed_embeddings:
            # Load pre-computed embeddings using original CSV index
            for agent_idx, agent in enumerate(selected_agents):
                embedding = self._load_embedding(original_csv_idx, agent_idx)
                agent_videos.append(embedding)
                agent_ids.append(agent['id'])
                agent_places.append(agent['place'])
            
            # Stack agent embeddings: [A, embed_dim]
            multi_agent_video = torch.stack(agent_videos, dim=0)
        else:
            # Load and process videos
            for agent in selected_agents:
                # Construct video path dynamically
                video_path = self._construct_video_path(match_id, agent['id'], round_num)
                video_clip = self._load_video_clip_with_decord(video_path, start_seconds, end_seconds)
                video_features = self._transform_video(video_clip)
                agent_videos.append(video_features)
                agent_ids.append(agent['id'])
                agent_places.append(agent['place'])
            
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
            'agent_ids': agent_ids,
            'agent_places': agent_places,
            'original_csv_idx': original_csv_idx  # Always include for embedding extraction
        }
        
        # Optionally include time information
        if self.return_time:
            result['time'] = row['normalized_prediction_seconds']
        
        return result


if __name__ == "__main__":
    # Test the dataset
    print("EnemyLocationNowcastDataset test placeholder")
