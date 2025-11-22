import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from .base import BaseVideoDataset
except ImportError:
    from base import BaseVideoDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def teammate_location_nowcast_collate_fn(batch):
    """
    Custom collate function for multi-agent teammate nowcast dataset.
    
    Args:
        batch: List of dictionaries containing:
            - 'video': Video features tensor [A, T, C, H, W] where A is num_agents
            - 'teammate_locations': Teammate location labels (regression or classification)
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
            # For tensors (video, teammate_locations, pov_team_side_encoded), use default collate
            collated[key] = torch.utils.data.default_collate(values)
    
    return collated


class TeammateLocationNowcastDataset(BaseVideoDataset, Dataset):
    """
    Multi-agent teammate location nowcast dataset.
    
    Given videos from a subset of players from one team, predict the current locations
    of all 5 players from the same team (at the middle point of the video segment).
    """
    
    def __init__(self, cfg: Dict):
        """
        Initialize Multi-Agent Teammate Location Nowcast Dataset.
        
        Args:
            cfg: Configuration dictionary containing dataset parameters
        """
        # Initialize base class - handles all common initialization
        super().__init__(cfg)
        
        # Check if we should return time information
        self.return_time = cfg.data.get('return_time', False)
            
        logger.info(f"Dataset initialized with {len(self.df)} samples")
        logger.info(f"Number of agents: {self.num_agents}, Task form: {self.task_form}")
        logger.info(f"Minimap masking enabled: {self.mask_minimap}")
        logger.info(f"Return time: {self.return_time}")
        if self.task_form in ['grid-cls', 'density-cls']:
            grid_res = self.cfg.data.grid_resolution
            logger.info(f"Grid resolution: {grid_res}x{grid_res} = {grid_res*grid_res} cells")
            if self.task_form == 'density-cls':
                logger.info(f"Gaussian sigma: {self.cfg.data.gaussian_sigma}")
        logger.info("Using single team with current location prediction (nowcast)")
    
    def _extract_unique_places(self) -> List[str]:
        """Extract unique place names from the teammate location data."""
        places = set()
        
        # Extract places from all teammate location columns
        for i in range(5):  # 5 teammates per team
            place_col = f'teammate_{i}_place'
            if place_col in self.df.columns:
                places.update(self.df[place_col].unique())
        
        # Remove empty/null values
        places = {p for p in places if p and str(p).strip() and str(p) != 'nan'}
        
        return sorted(list(places))
    
    def _get_team_player_data(self, row: pd.Series, player_idx: int) -> Dict:
        """Extract team player data from a row."""
        return {
            'id': row[f'teammate_{player_idx}_id'],
            'name': row[f'teammate_{player_idx}_name'],
            'side': row[f'teammate_{player_idx}_side'],
            'X_norm': row[f'teammate_{player_idx}_X'],
            'Y_norm': row[f'teammate_{player_idx}_Y'],
            'Z_norm': row[f'teammate_{player_idx}_Z'],
            'place': row[f'teammate_{player_idx}_place']
        }
    
    def _get_all_team_players(self, row: pd.Series) -> List[Dict]:
        """Get all 5 teammates from the team."""
        team_players = []
        
        for i in range(5):  # 5 teammates per team
            try:
                player_data = self._get_team_player_data(row, i)
                team_players.append(player_data)
            except KeyError:
                # Teammate column doesn't exist, skip
                continue
        
        return team_players
    
    def _create_teammate_location_labels(self, team_players: List[Dict]) -> torch.Tensor:
        """
        Create teammate location labels based on task_form.
        
        Args:
            team_players: List of team player data dictionaries with location info
            
        Returns:
            labels: Tensor containing teammate location labels
        """
        return self.label_creator.create_labels(team_players)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample containing multi-agent video data and teammate location labels.
        
        Returns:
            dict: Dictionary containing:
                - 'video': Multi-agent video features tensor [A, T, C, H, W]
                - 'teammate_locations': Teammate location labels
                - 'pov_team_side': Team side used (string)
                - 'pov_team_side_encoded': Team side encoded as boolean (0 for T, 1 for CT)
                - 'agent_ids': List of agent IDs used
                - 'agent_places': List of agent places (locations)
                - 'time': Normalized prediction seconds (if return_time is enabled)
        """
        # Get sample information
        row = self.df.iloc[idx]
        start_seconds = row['normalized_start_seconds']
        end_seconds = row['normalized_end_seconds']
        match_id = row['match_id']
        round_num = row['round_num']
        pov_team_side = row['pov_team_side'].upper()  # Convert to uppercase (T or CT)
        
        # Get original CSV index for loading pre-computed embeddings
        original_csv_idx = row.get('original_csv_idx', idx)
        
        # Get team players (both input and labels from same team)
        team_players = self._get_all_team_players(row)
        
        # Select subset of agents from team for input
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
        
        # Create teammate location labels (using all 5 team players)
        teammate_location_labels = self._create_teammate_location_labels(team_players)
        
        # Encode team side as boolean for model input (T=0, CT=1)
        pov_team_side_encoded = 1 if pov_team_side == 'CT' else 0
        
        # Prepare return dictionary
        result = {
            'video': multi_agent_video,
            'teammate_locations': teammate_location_labels,
            'pov_team_side': pov_team_side,
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
    print("TeammateLocationNowcastDataset test placeholder")

