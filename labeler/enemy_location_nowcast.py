import sys
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

from labeler.base import LocationPredictionBase

load_dotenv()


class EnemyLocationNowcastCreator(LocationPredictionBase):
    """
    Creates labeled segments for enemy location nowcast task.
    
    Takes a POV player's video segment and predicts enemy locations at the current moment
    (middle point of the segment). The task is to predict where enemy team players are
    located at the same time as the observation.
    """
    
    def _extract_segments_from_round(self, match_id: str, round_num: int, 
                                   config: Dict[str, Any]) -> List[Dict]:
        """
        Extract segments for enemy location nowcast from a specific round.
        
        Args:
            match_id: Match identifier
            round_num: Round number
            config: Configuration dictionary containing all parameters
        
        Returns:
            List of segment dictionaries with all players' data
        """
        segment_length_sec = config['segment_length_sec']
        # Load all player trajectories for this round
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        
        if len(player_trajectories) != 10:
            return []  # Must have exactly 10 players
        
        segments = []
        segment_ticks = segment_length_sec * self.tick_rate
        
        # Find valid tick range where all players are alive
        min_tick, max_tick_alive = self._get_valid_tick_range(player_trajectories)
        
        if max_tick_alive - min_tick < segment_ticks:
            return []  # Not enough ticks for even one segment
        
        # Generate segments with configurable stride
        # Convert stride from seconds to ticks
        stride_ticks = int(self.stride_sec * self.tick_rate)
        stride_ticks = max(1, stride_ticks)  # Ensure at least 1 tick step
        current_tick = min_tick
        
        while current_tick + segment_ticks <= max_tick_alive:
            end_tick = current_tick + segment_ticks
            middle_tick = current_tick + segment_ticks // 2  # Current moment (middle of segment)
            
            # Check if all players are alive throughout the segment
            all_players_alive = True
            for df in player_trajectories.values():
                segment_data = df[(df['tick'] >= current_tick) & (df['tick'] <= end_tick)]
                if segment_data.empty or (segment_data['health'] <= 0).any():
                    all_players_alive = False
                    break
            
            if not all_players_alive:
                current_tick += stride_ticks
                continue
            
            # Get all players' locations at the middle tick
            all_players_data = []
            for steamid, df in player_trajectories.items():
                player_data = self._extract_player_data_at_tick(df, middle_tick)
                if player_data:
                    all_players_data.append(player_data)
            
            if len(all_players_data) == 10:  # All 10 players have valid data
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
                    'prediction_tick': middle_tick,
                    'start_seconds': current_tick / self.tick_rate,
                    'end_seconds': end_tick / self.tick_rate,
                    'prediction_seconds': middle_tick / self.tick_rate,
                    'normalized_start_seconds': (current_tick - min_tick) / self.tick_rate,
                    'normalized_end_seconds': (end_tick - min_tick) / self.tick_rate,
                    'normalized_prediction_seconds': (middle_tick - min_tick) / self.tick_rate,
                    'duration_seconds': segment_length_sec,
                    'map_name': map_name,
                    'all_players_data': all_players_data
                }
                segments.append(segment_info)
            
            # Move to next step based on stride
            current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create the final CSV output for enemy location nowcast data."""
        output_rows = []
        idx = 0
        
        for segment in all_segments:
            row = {
                'idx': idx,
                'partition': segment['partition'],
                'seg_duration_sec': segment['duration_seconds'],
                'start_tick': segment['start_tick'],
                'end_tick': segment['end_tick'],
                'prediction_tick': segment['prediction_tick'],
                'start_seconds': segment['start_seconds'],
                'end_seconds': segment['end_seconds'],
                'prediction_seconds': segment['prediction_seconds'],
                'normalized_start_seconds': segment['normalized_start_seconds'],
                'normalized_end_seconds': segment['normalized_end_seconds'],
                'normalized_prediction_seconds': segment['normalized_prediction_seconds'],
                'match_id': segment['match_id'],
                'round_num': segment['round_num'],
                'map_name': segment['map_name']
            }
            
            # Add all players' location data (10 players total)
            for i, player in enumerate(segment['all_players_data']):
                row[f'player_{i}_id'] = player['steamid']
                row[f'player_{i}_name'] = player['name']
                row[f'player_{i}_side'] = player['side']
                row[f'player_{i}_X'] = player['X_norm']
                row[f'player_{i}_Y'] = player['Y_norm']
                row[f'player_{i}_Z'] = player['Z_norm']
                row[f'player_{i}_place'] = player['place']
            
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
    """Main function for testing enemy location nowcast creation."""
    # Load paths from environment variables
    DATA_BASE_PATH = os.getenv('DATA_BASE_PATH')
    if not DATA_BASE_PATH:
        raise ValueError("DATA_BASE_PATH environment variable not set. Please check your .env file.")
    
    DATA_DIR = DATA_BASE_PATH
    OUTPUT_DIR = os.path.join(DATA_BASE_PATH, 'labels')
    PARTITION_CSV_PATH = os.path.join(DATA_BASE_PATH, 'match_round_partitioned.csv')
    
    # Create enemy location nowcast creator
    creator = EnemyLocationNowcastCreator(
        DATA_DIR,
        OUTPUT_DIR, 
        PARTITION_CSV_PATH,
        cpu_usage=0.9,
        stride_sec=5.0  # 1 second stride by default
    )
    
    # Process segments
    creator.process_segments({
        'output_file_name': 'enemy_location_nowcast_5s.csv',
        'segment_length_sec': 5,
        'partition': ['train', 'val', 'test']
    })


if __name__ == "__main__":
    main()
