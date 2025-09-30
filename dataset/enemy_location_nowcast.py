import sys
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import random
from sklearn.preprocessing import MinMaxScaler

try:
    from .base import BaseVideoDataset
    from .label_creators import create_label_creator
except ImportError:
    from base import BaseVideoDataset
    from label_creators import create_label_creator

try:
    from utils.dataset_utils import apply_minimap_mask
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.dataset_utils import apply_minimap_mask

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
            - 'team_side': String indicating team side
            - 'agent_ids': List of agent IDs used
    
    Returns:
        Dictionary with batched tensors and lists of string values
    """
    collated = {}
    
    # Handle each key appropriately
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        
        if key in ['team_side', 'agent_ids']:
            # Keep string/list values as lists
            collated[key] = values
        else:
            # For tensors (video, enemy_locations, team_side_encoded), use default collate
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
        # Initialize base class first
        super().__init__(cfg)
        
        # Get data config section
        self.cfg = cfg
        
        # Load label CSV file
        self.label_path = cfg.data.label_path
        self.df = pd.read_csv(self.label_path, keep_default_na=False)
        
        # Multi-agent enemy location prediction parameters
        self.num_agents = cfg.data.num_agents
        self.task_form = cfg.data.task_form  # regression or classification
        self.mask_minimap = getattr(cfg.data, 'mask_minimap', False)
        
        # Validate parameters
        if self.num_agents < 1 or self.num_agents > 5:
            raise ValueError(f"num_agents must be between 1 and 5, got {self.num_agents}")
            
        valid_task_forms = ['coord-reg', 'generative', 'multi-label-cls', 'multi-output-reg', 'grid-cls', 'density-cls']
        if self.task_form not in valid_task_forms:
            raise ValueError(f"task_form must be one of {valid_task_forms}, got {self.task_form}")
        
        # Get unique place names for place-based classification tasks
        if self.task_form in ['multi-label-cls', 'multi-output-reg']:
            self.place_names = self._extract_unique_places()
            self.place_to_idx = {place: idx for idx, place in enumerate(self.place_names)}
            self.idx_to_place = {idx: place for place, idx in self.place_to_idx.items()}
            self.num_places = len(self.place_names)
            logger.info(f"Found {self.num_places} unique places: {self.place_names}")
        else:
            self.place_names = None
            self.place_to_idx = None
            self.idx_to_place = None
            self.num_places = None
        
        # Filter by partition if specified
        self.partition = cfg.data.partition
        if self.partition != 'all':
            initial_count = len(self.df)
            self.df = self.df[self.df['partition'] == self.partition].reset_index(drop=True)
            filtered_count = len(self.df)
            logger.info(f"Filtered dataset from {initial_count} to {filtered_count} samples for partition '{self.partition}'")
        
        # Store additional configuration parameters
        
        # Store output directory for saving/loading scaler
        self.output_dir = Path(cfg.path.exp)
        
        # Initialize coordinate scaler for coordinate-based tasks
        self.coordinate_scaler = None
        self.scaler_fitted = False
        if self.task_form in ['coord-reg', 'generative', 'grid-cls', 'density-cls']:
            self._init_coordinate_scaler()
        
        # Initialize label creator
        self._init_label_creator()
            
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
    
    def _init_coordinate_scaler(self):
        """Initialize coordinate scaler for regression mode."""
        self.coordinate_scaler = MinMaxScaler()
        self.scaler_path = self.output_dir / "coordinate_minmax_scaler.pkl"
        
        # Try to load existing scaler first
        if self.scaler_path.exists():
            try:
                with open(self.scaler_path, 'rb') as f:
                    self.coordinate_scaler = pickle.load(f)
                self.scaler_fitted = True
                logger.info(f"Loaded existing coordinate scaler from {self.scaler_path}")
                logger.info(f"Scaler data_min_: {self.coordinate_scaler.data_min_}")
                logger.info(f"Scaler data_max_: {self.coordinate_scaler.data_max_}")
                logger.info(f"Scaler scale_: {self.coordinate_scaler.scale_}")
                return
            except Exception as e:
                logger.warning(f"Failed to load existing scaler: {e}")
                self.coordinate_scaler = MinMaxScaler()
        
        # Fit scaler on all coordinate data
        logger.info("Fitting coordinate scaler on all data...")
        all_coords = []
        
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            # For scaler fitting, we need to consider both teams as potential enemies
            # Get all players and extract coordinates from both teams
            for team_side in ['T', 'CT']:
                team_players = self._get_team_players(row, team_side)
                
                for player in team_players:
                    try:
                        coords = [float(player['X']), float(player['Y']), float(player['Z'])]
                        all_coords.append(coords)
                    except (ValueError, KeyError):
                        # Skip invalid coordinates
                        continue
        
        if all_coords:
            all_coords = np.array(all_coords)
            self.coordinate_scaler.fit(all_coords)
            self.scaler_fitted = True
            
            # Save the fitted scaler
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.coordinate_scaler, f)
            
            logger.info(f"Fitted and saved coordinate scaler to {self.scaler_path}")
            logger.info(f"Scaler data_min_: {self.coordinate_scaler.data_min_}")
            logger.info(f"Scaler data_max_: {self.coordinate_scaler.data_max_}")
            logger.info(f"Scaler scale_: {self.coordinate_scaler.scale_}")
            logger.info(f"Fitted on {len(all_coords)} coordinate samples")
        else:
            logger.warning("No valid coordinates found for fitting scaler!")
    
    def _scale_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """Scale coordinates using the fitted scaler."""
        if self.coordinate_scaler is None or not self.scaler_fitted:
            return coords
        
        # coords shape: [5, 3] or [N, 3]
        original_shape = coords.shape
        coords_flat = coords.reshape(-1, 3)
        
        # Only scale non-zero coordinates (padding coordinates remain [0, 0, 0])
        mask = np.any(coords_flat != 0, axis=1)
        if np.any(mask):
            coords_flat[mask] = self.coordinate_scaler.transform(coords_flat[mask])
        
        return coords_flat.reshape(original_shape)
    
    def _unscale_coordinates(self, scaled_coords: np.ndarray) -> np.ndarray:
        """Unscale coordinates using the fitted scaler."""
        if self.coordinate_scaler is None or not self.scaler_fitted:
            return scaled_coords
        
        # scaled_coords shape: [5, 3] or [N, 3]
        original_shape = scaled_coords.shape
        coords_flat = scaled_coords.reshape(-1, 3)
        
        # Only unscale non-zero coordinates (padding coordinates remain [0, 0, 0])
        mask = np.any(coords_flat != 0, axis=1)
        if np.any(mask):
            coords_flat[mask] = self.coordinate_scaler.inverse_transform(coords_flat[mask])
        
        return coords_flat.reshape(original_shape)
    
    def get_coordinate_scaler(self):
        """Get the fitted coordinate scaler for external use."""
        return self.coordinate_scaler if self.scaler_fitted else None
    
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
    
    def _get_team_players(self, row: pd.Series, team_side: str) -> List[Dict]:
        """Get all players from a specific team side."""
        team_players = []
        
        for i in range(10):  # Assuming 10 players (0-9)
            try:
                player_data = self._get_player_data(row, i)
                if player_data['side'] == team_side.lower():  # sides are lowercase in data
                    team_players.append(player_data)
            except KeyError:
                # Player column doesn't exist, skip
                continue
        
        return team_players
    
    def _select_agents(self, team_players: List[Dict]) -> List[Dict]:
        """Select a subset of agents from team players."""
        if len(team_players) < self.num_agents:
            logger.warning(f"Not enough players in team. Found {len(team_players)}, need {self.num_agents}")
            # Pad with the last player if not enough players
            while len(team_players) < self.num_agents:
                if team_players:
                    team_players.append(team_players[-1])
                else:
                    raise ValueError("No players found in team")
        
        # Take the first num_agents players
        return team_players[:self.num_agents]
    
    def _construct_video_path(self, match_id: str, player_id: str, round_num: int) -> str:
        """Construct video path for a player's round."""
        video_folder = self.cfg.data.video_folder
        video_path = Path('data') / video_folder / str(match_id) / str(player_id) / f"round_{round_num}.mp4"
        return str(video_path)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _init_label_creator(self):
        """Initialize label creator based on task form."""
        kwargs = {}
        
        if self.task_form in ['coord-reg', 'generative', 'grid-cls', 'density-cls']:
            kwargs['coordinate_scaler'] = self.coordinate_scaler if self.scaler_fitted else None
        
        if self.task_form in ['multi-label-cls', 'multi-output-reg']:
            kwargs['place_to_idx'] = self.place_to_idx
            kwargs['num_places'] = self.num_places
        
        self.label_creator = create_label_creator(self.cfg, **kwargs)
        logger.info(f"Initialized label creator: {self.label_creator.__class__.__name__}")
    
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
                - 'team_side': Team side used for input (string)
                - 'team_side_encoded': Team side encoded as boolean (0 for T, 1 for CT)
                - 'agent_ids': List of agent IDs used
        """
        # Get sample information
        row = self.df.iloc[idx]
        start_seconds = row['normalized_start_seconds']
        end_seconds = row['normalized_end_seconds']
        match_id = row['match_id']
        round_num = row['round_num']
        
        # Randomly select team side for this sample
        selected_team_side = random.choice(['CT', 'T'])
        enemy_team_side = 'T' if selected_team_side == 'CT' else 'CT'
        
        # Get team players (input) and enemy players (labels)
        team_players = self._get_team_players(row, selected_team_side)
        enemy_players = self._get_team_players(row, enemy_team_side)
        
        # Select subset of agents from team
        selected_agents = self._select_agents(team_players)
        
        # Load videos for selected agents
        agent_videos = []
        agent_ids = []
        
        for agent in selected_agents:
            # Construct video path dynamically
            video_path = self._construct_video_path(match_id, agent['id'], round_num)
            video_clip = self._load_video_clip_with_torchcodec(video_path, start_seconds, end_seconds)
            video_features = self._transform_video(video_clip)
            agent_videos.append(video_features)
            agent_ids.append(agent['id'])
        
        # Stack agent videos: [A, T, C, H, W]
        multi_agent_video = torch.stack(agent_videos, dim=0)
        
        # Create enemy location labels
        enemy_location_labels = self._create_enemy_location_labels(enemy_players)
        
        # Encode team side as boolean for model input (T=0, CT=1)
        team_side_encoded = 1 if selected_team_side == 'CT' else 0
        
        # Prepare return dictionary
        result = {
            'video': multi_agent_video,
            'enemy_locations': enemy_location_labels,
            'team_side': selected_team_side,
            'team_side_encoded': torch.tensor(team_side_encoded, dtype=torch.long),
            'agent_ids': agent_ids
        }
        
        return result


if __name__ == "__main__":
    # Test the dataset
    print("EnemyLocationNowcastDataset test placeholder")
