import sys
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

from labeler.base import LocationPredictionBase

load_dotenv()


class TeammateLocationNowcastCreator(LocationPredictionBase):
    """
    Creates labeled segments for teammate location nowcast task.
    
    Takes a POV player's video segment and predicts teammate locations at the current moment
    (middle point of the segment). The task is to predict where same team players are
    located at the same time as the observation.
    """
    
    def _extract_segments_from_round(self, match_id: str, round_num: int, 
                                   cfg: Dict[str, Any]) -> List[Dict]:
        """
        Extract segments for teammate location nowcast from a specific round.
        
        Args:
            match_id: Match identifier
            round_num: Round number
            cfg: Configuration dictionary containing all parameters
        
        Returns:
            List of segment dictionaries with all players' data by side
        """
        segment_length_sec = cfg['segment_length_sec']
        
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
            
            # Separate players by side
            t_players = self._get_players_by_side(player_trajectories, 't')
            ct_players = self._get_players_by_side(player_trajectories, 'ct')
            
            if len(t_players) != 5 or len(ct_players) != 5:
                current_tick += 1
                continue
            
            # Get T team current locations at middle_tick
            t_team_current_locations = []
            for steamid, df in t_players.items():
                player_data = self._extract_player_data_at_tick(df, middle_tick)
                if player_data:
                    t_team_current_locations.append(player_data)
            
            # Get CT team current locations at middle_tick
            ct_team_current_locations = []
            for steamid, df in ct_players.items():
                player_data = self._extract_player_data_at_tick(df, middle_tick)
                if player_data:
                    ct_team_current_locations.append(player_data)
            
            if len(t_team_current_locations) == 5 and len(ct_team_current_locations) == 5:
                # Sort players by steamid for consistent ordering
                t_team_current_locations.sort(key=lambda x: x['steamid'])
                ct_team_current_locations.sort(key=lambda x: x['steamid'])
                
                # Get map name from first available player
                map_name = 'unknown'
                for df in player_trajectories.values():
                    if 'map_name' in df.columns and not df.empty:
                        map_name = df.iloc[0]['map_name']
                        break
                
                # Create segment for T side
                t_segment_info = {
                    'pov_team_side': 't',
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
                    'team_current_locations': t_team_current_locations
                }
                segments.append(t_segment_info)
                
                # Create segment for CT side
                ct_segment_info = {
                    'pov_team_side': 'ct',
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
                    'team_current_locations': ct_team_current_locations
                }
                segments.append(ct_segment_info)
            
            # Move to next step based on stride
            current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], cfg: Dict[str, Any]) -> pd.DataFrame:
        """Create the final CSV output for teammate location nowcast data."""
        output_rows = []
        idx = 0
        
        for segment in all_segments:
            row = {
                'idx': idx,
                'partition': segment['partition'],
                'pov_team_side': segment['pov_team_side'],
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
            
            # Add team's current location data (5 teammates)
            for i, teammate in enumerate(segment['team_current_locations']):
                row[f'teammate_{i}_id'] = teammate['steamid']
                row[f'teammate_{i}_name'] = teammate['name']
                row[f'teammate_{i}_side'] = teammate['side']
                row[f'teammate_{i}_X'] = teammate['X_norm']
                row[f'teammate_{i}_Y'] = teammate['Y_norm']
                row[f'teammate_{i}_Z'] = teammate['Z_norm']
                row[f'teammate_{i}_place'] = teammate['place']
            
            output_rows.append(row)
            idx += 1
        
        # Create DataFrame and sort
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'pov_team_side', 'match_id', 'round_num'], 
                               ascending=[True, True, True, True])
            df = df.reset_index(drop=True)
            # Update idx after sorting
            df['idx'] = range(len(df))
        
        return df


def main():
    """Main function for testing teammate location nowcast creation."""
    # Load paths from environment variables
    DATA_BASE_PATH = os.getenv('DATA_BASE_PATH')
    if not DATA_BASE_PATH:
        raise ValueError("DATA_BASE_PATH environment variable not set. Please check your .env file.")
    
    DATA_DIR = DATA_BASE_PATH
    OUTPUT_DIR = os.path.join(DATA_BASE_PATH, 'labels')
    PARTITION_CSV_PATH = os.path.join(DATA_BASE_PATH, 'match_round_partitioned.csv')
    
    # Create teammate location nowcast creator
    creator = TeammateLocationNowcastCreator(
        DATA_DIR,
        OUTPUT_DIR, 
        PARTITION_CSV_PATH,
        cpu_usage=0.9,
        stride_sec=1.0  # 1 second stride by default
    )
    
    # Process segments
    creator.process_segments({
        'output_file_name': 'teammate_location_nowcast_s1s_l5s.csv',
        'segment_length_sec': 5,
        'partition': ['train', 'val', 'test']
    })


if __name__ == "__main__":
    main()

