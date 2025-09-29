import sys
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from labeler.base import LocationPredictionBase


class EnemyLocationForecastCreator(LocationPredictionBase):
    """
    Creates labeled segments for enemy location forecast task.
    
    Takes a POV player's video segment and predicts enemy locations at a future moment
    (segment_duration + forecast_interval seconds later). The task is to predict where 
    enemy team players will be located in the future.
    """
    
    def _extract_segments_from_round(self, match_id: str, round_num: int, 
                                   config: Dict[str, Any]) -> List[Dict]:
        """
        Extract segments for enemy location forecast from a specific round.
        
        Args:
            match_id: Match identifier
            round_num: Round number
            config: Configuration dictionary containing all parameters
        
        Returns:
            List of segment dictionaries with all players' data
        """
        segment_length_sec = config['segment_length_sec']
        forecast_interval_sec = config.get('forecast_interval_sec', 10)
        
        # Load all player trajectories for this round
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        
        if len(player_trajectories) != 10:
            return []  # Must have exactly 10 players
        
        segments = []
        segment_ticks = segment_length_sec * self.tick_rate
        forecast_ticks = forecast_interval_sec * self.tick_rate
        total_window_ticks = segment_ticks + forecast_ticks
        
        # Find valid tick range where all players are alive
        min_tick, max_tick_alive = self._get_valid_tick_range(player_trajectories)
        
        if max_tick_alive - min_tick < total_window_ticks:
            return []  # Not enough ticks for segment + forecast
        
        # Generate segments with configurable stride
        # Convert stride from seconds to ticks
        stride_ticks = int(self.stride_sec * self.tick_rate)
        stride_ticks = max(1, stride_ticks)  # Ensure at least 1 tick step
        current_tick = min_tick
        
        while current_tick + total_window_ticks <= max_tick_alive:
            end_tick = current_tick + segment_ticks
            forecast_tick = current_tick + total_window_ticks  # Future prediction point
            
            # Check if all players are alive throughout the entire window (segment + forecast)
            all_players_alive = True
            for df in player_trajectories.values():
                window_data = df[(df['tick'] >= current_tick) & (df['tick'] <= forecast_tick)]
                if window_data.empty or (window_data['health'] <= 0).any():
                    all_players_alive = False
                    break
            
            if not all_players_alive:
                current_tick += stride_ticks
                continue
            
            # Get all players' future locations at forecast_tick
            all_players_data = []
            for steamid, df in player_trajectories.items():
                player_future_data = self._extract_player_data_at_tick(df, forecast_tick)
                if player_future_data:
                    all_players_data.append(player_future_data)
            
            if len(all_players_data) == 10:  # All 10 players have valid future data
                # Sort players by steamid for consistent ordering
                all_players_data.sort(key=lambda x: x['steamid'])
                
                # Get map name from first available player
                map_name = 'unknown'
                for df in player_trajectories.values():
                    if 'map_name' in df.columns and not df.empty:
                        map_name = df.iloc[0]['map_name']
                        break
                
                segment_info = {
                    'start_tick': current_tick,
                    'end_tick': end_tick,
                    'forecast_tick': forecast_tick,
                    'start_seconds': current_tick / self.tick_rate,
                    'end_seconds': end_tick / self.tick_rate,
                    'forecast_seconds': forecast_tick / self.tick_rate,
                    'normalized_start_seconds': (current_tick - min_tick) / self.tick_rate,
                    'normalized_end_seconds': (end_tick - min_tick) / self.tick_rate,
                    'normalized_forecast_seconds': (forecast_tick - min_tick) / self.tick_rate,
                    'duration_seconds': segment_length_sec,
                    'forecast_interval_seconds': forecast_interval_sec,
                    'map_name': map_name,
                    'all_players_future_data': all_players_data
                }
                segments.append(segment_info)
            
            # Move to next step based on stride
            current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create the final CSV output for enemy location forecast data."""
        output_rows = []
        idx = 0
        
        for segment in all_segments:
            row = {
                'idx': idx,
                'partition': segment['partition'],
                'seg_duration_sec': segment['duration_seconds'],
                'forecast_interval_sec': segment['forecast_interval_seconds'],
                'start_tick': segment['start_tick'],
                'end_tick': segment['end_tick'],
                'forecast_tick': segment['forecast_tick'],
                'start_seconds': segment['start_seconds'],
                'end_seconds': segment['end_seconds'],
                'forecast_seconds': segment['forecast_seconds'],
                'normalized_start_seconds': segment['normalized_start_seconds'],
                'normalized_end_seconds': segment['normalized_end_seconds'],
                'normalized_forecast_seconds': segment['normalized_forecast_seconds'],
                'match_id': segment['match_id'],
                'round_num': segment['round_num'],
                'map_name': segment['map_name']
            }
            
            # Add all players' future location data (10 players total)
            for i, player in enumerate(segment['all_players_future_data']):
                row[f'player_{i}_id'] = player['steamid']
                row[f'player_{i}_name'] = player['name']
                row[f'player_{i}_side'] = player['side']
                row[f'player_{i}_future_X'] = player['X']
                row[f'player_{i}_future_Y'] = player['Y']
                row[f'player_{i}_future_Z'] = player['Z']
                row[f'player_{i}_future_place'] = player['place']
            
            output_rows.append(row)
            idx += 1
        
        # Create DataFrame and sort
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'], 
                               ascending=[True, True, True])
            df = df.reset_index(drop=True)
            # Update idx after sorting
            df['idx'] = range(len(df))
        
        return df


def main():
    """Main function for testing enemy location forecast creation."""
    # Load paths from environment variables
    DATA_BASE_PATH = os.getenv('DATA_BASE_PATH')
    if not DATA_BASE_PATH:
        raise ValueError("DATA_BASE_PATH environment variable not set. Please check your .env file.")
    
    DATA_DIR = DATA_BASE_PATH
    OUTPUT_DIR = os.path.join(DATA_BASE_PATH, 'labels')
    PARTITION_CSV_PATH = os.path.join(DATA_BASE_PATH, 'match_round_partitioned.csv')
    
    # Create enemy location forecast creator
    creator = EnemyLocationForecastCreator(
        DATA_DIR,
        OUTPUT_DIR, 
        PARTITION_CSV_PATH,
        cpu_usage=0.9,
        stride_sec=1.0  # 1 second stride by default
    )
    
    # Process segments
    creator.process_segments({
        'output_file_name': 'enemy_location_forecast_5s_10s.csv',
        'segment_length_sec': 5,
        'forecast_interval_sec': 10,
        'partition': ['train', 'val', 'test']
    })


if __name__ == "__main__":
    main()
