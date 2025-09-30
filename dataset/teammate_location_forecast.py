import sys
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

try:
    from .base import BaseVideoDataset
except ImportError:
    from base_video_dataset import BaseVideoDataset


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def teammate_location_forecast_collate_fn(batch):
    """
    Custom collate function for multi-agent future location prediction dataset.
    
    Args:
        batch: List of dictionaries containing:
            - 'video': Video features tensor [A, T, C, H, W] where A is num_agents
            - 'future_locations': Future location labels (regression or classification)
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
            # For tensors (video, future_locations, team_side_encoded), use default collate
            collated[key] = torch.utils.data.default_collate(values)
    
    return collated


class TeammateLocationForecastDataset(BaseVideoDataset, Dataset):
    """
    Multi-agent self-team future location prediction dataset.
    
    Given videos from all 5 players from one team, predict the future locations
    of the same team K seconds into the future.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Multi-Agent Self-Team Future Location Prediction Dataset.
        
        Args:
            config: Configuration dictionary containing dataset parameters
        """
        # Initialize base class first
        super().__init__(config)
        
        # Get data config section
        data_config = config['data']
        self.path_config = config["path"]
        
        # Load label CSV file
        self.label_path = data_config['label_path']
        self.df = pd.read_csv(self.label_path, keep_default_na=False)
        
        # Multi-agent future location prediction parameters
        self.num_agents = data_config['num_agents']  # Should be 5 for full team
        self.task_form = data_config['task_form']  # regression or classification
        
        # Validate parameters
        if self.num_agents < 1 or self.num_agents > 5:
            raise ValueError(f"num_agents must be between 1 and 5, got {self.num_agents}")
            
        if self.task_form not in ['regression', 'classification', 'generative']:
            raise ValueError(f"task_form must be 'regression', 'classification', or 'generative', got {self.task_form}")
        
        # Get unique place names for classification
        if self.task_form == 'classification':
            self.place_names = self._extract_unique_places()
            self.place_to_idx = {place: idx for idx, place in enumerate(self.place_names)}
            self.idx_to_place = {idx: place for place, idx in self.place_to_idx.items()}
            self.num_places = len(self.place_names)
            logger.info(f"Found {self.num_places} unique places: {self.place_names}")
        
        # Filter by partition if specified
        self.partition = data_config['partition']
        if self.partition != 'all':
            initial_count = len(self.df)
            self.df = self.df[self.df['partition'] == self.partition].reset_index(drop=True)
            filtered_count = len(self.df)
            logger.info(f"Filtered dataset from {initial_count} to {filtered_count} samples for partition '{self.partition}'")
        
        # Store additional configuration parameters
        
        # Store output directory for saving/loading scaler
        self.output_dir = Path(config['path']['exp'])
        
        # Initialize coordinate scaler for regression
        self.coordinate_scaler = None
        self.scaler_fitted = False
        if self.task_form in ['regression', 'generative']:
            self._init_coordinate_scaler()
            
        logger.info(f"Dataset initialized with {len(self.df)} samples")
        logger.info(f"Number of agents: {self.num_agents}, Location form: {self.task_form}")
        logger.info("Using single team with future location prediction")
    
    def _extract_unique_places(self) -> List[str]:
        """Extract unique place names from the future location data."""
        places = set()
        
        # Extract places from all player future location columns
        for i in range(5):  # 5 players per team
            place_col = f'player_{i}_future_place'
            if place_col in self.df.columns:
                places.update(self.df[place_col].unique())
        
        # Remove empty/null values
        places = {p for p in places if p and str(p).strip() and str(p) != 'nan'}
        
        return sorted(list(places))
    
    def _init_coordinate_scaler(self):
        """Initialize coordinate scaler for regression mode."""
        self.coordinate_scaler = MinMaxScaler()
        self.scaler_path = self.output_dir / "future_coordinate_minmax_scaler.pkl"
        
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
        
        # Fit scaler on all future coordinate data
        logger.info("Fitting coordinate scaler on all future location data...")
        all_coords = []
        
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            # Get future coordinates for all 5 team players
            for i in range(5):
                try:
                    x_col = f'player_{i}_future_X'
                    y_col = f'player_{i}_future_Y'
                    z_col = f'player_{i}_future_Z'
                    
                    if all(col in row.index for col in [x_col, y_col, z_col]):
                        coords = [float(row[x_col]), float(row[y_col]), float(row[z_col])]
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
            logger.warning("No valid future coordinates found for fitting scaler!")
    
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
    
    def _get_team_player_data(self, row: pd.Series, player_idx: int) -> Dict:
        """Extract team player data from a row."""
        return {
            'id': row[f'player_{player_idx}_id'],
            'video_path': row[f'player_{player_idx}_video_path'],
            'name': row[f'player_{player_idx}_name'],
            'future_X': row[f'player_{player_idx}_future_X'],
            'future_Y': row[f'player_{player_idx}_future_Y'],
            'future_Z': row[f'player_{player_idx}_future_Z'],
            'future_place': row[f'player_{player_idx}_future_place']
        }
    
    def _get_all_team_players(self, row: pd.Series) -> List[Dict]:
        """Get all 5 players from the team."""
        team_players = []
        
        for i in range(5):  # 5 players per team in the future location dataset
            try:
                player_data = self._get_team_player_data(row, i)
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
    
    def __len__(self) -> int:
        return len(self.df)
    
    
    
    def _create_future_location_labels(self, team_players: List[Dict]) -> torch.Tensor:
        """
        Create future location labels based on task_form.
        
        Args:
            team_players: List of team player data dictionaries with future location info
            
        Returns:
            labels: Tensor containing future location labels
        """
        if self.task_form in ['regression', 'generative']:
            # Return XYZ coordinates for all 5 team players' future locations
            # Shape: [5, 3] for 5 players with X, Y, Z coordinates
            coords = []
            for i in range(5):
                if i < len(team_players):
                    player = team_players[i]
                    coords.append([float(player['future_X']), float(player['future_Y']), float(player['future_Z'])])
                else:
                    # Pad with zeros if not enough team players
                    coords.append([0.0, 0.0, 0.0])
            
            coords = np.array(coords)
            
            # Scale coordinates if scaler is available
            if self.coordinate_scaler is not None and self.scaler_fitted:
                coords = self._scale_coordinates(coords)
            
            return torch.tensor(coords, dtype=torch.float32)
        
        elif self.task_form == 'classification':
            # Return histogram of team player counts per place for multinomial loss
            # Shape: [num_places] where each value is count of team players at that future place (0-5)
            place_counts = torch.zeros(self.num_places, dtype=torch.float32)
            
            for player in team_players:
                place = player['future_place']
                if place in self.place_to_idx:
                    place_idx = self.place_to_idx[place]
                    place_counts[place_idx] += 1.0
            
            return place_counts
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample containing multi-agent video data and future location labels.
        
        Returns:
            dict: Dictionary containing:
                - 'video': Multi-agent video features tensor [A, T, C, H, W]
                - 'future_locations': Future location labels
                - 'team_side': Team side used (string)
                - 'team_side_encoded': Team side encoded as boolean (0 for T, 1 for CT)
                - 'agent_ids': List of agent IDs used
        """
        # Get sample information
        row = self.df.iloc[idx]
        start_seconds = row['normalized_start_seconds']
        end_seconds = row['normalized_end_seconds']
        team_side = row['team_side'].upper()  # Convert to uppercase (T or CT)
        
        # Get team players (both input and labels from same team)
        team_players = self._get_all_team_players(row)
        
        # Select subset of agents from team for input
        selected_agents = self._select_agents(team_players)
        
        # Load videos for selected agents
        agent_videos = []
        agent_ids = []
        
        for agent in selected_agents:
            video_clip = self._load_video_clip_with_torchcodec(agent['video_path'], start_seconds, end_seconds)
            video_features = self._transform_video(video_clip)
            agent_videos.append(video_features)
            agent_ids.append(agent['id'])
        
        # Stack agent videos: [A, T, C, H, W]
        multi_agent_video = torch.stack(agent_videos, dim=0)
        
        # Create future location labels (using all 5 team players)
        future_location_labels = self._create_future_location_labels(team_players)
        
        # Encode team side as boolean for model input (T=0, CT=1)
        team_side_encoded = 1 if team_side == 'CT' else 0
        
        # Prepare return dictionary
        result = {
            'video': multi_agent_video,
            'future_locations': future_location_labels,
            'team_side': team_side,
            'team_side_encoded': torch.tensor(team_side_encoded, dtype=torch.long),
            'agent_ids': agent_ids
        }
        
        return result


if __name__ == "__main__":
    # Test the dataset
    print("TeammateLocationForecastDataset test placeholder")