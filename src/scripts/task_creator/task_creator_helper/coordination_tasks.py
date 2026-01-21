"""
Team coordination task creators.

Creates labels for:
- Teammate/enemy alive count

Label column naming convention:
- multi_cls: label (single column with class index)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

from .base_task_creator import TaskCreatorBase


class AliveCountCreator(TaskCreatorBase):
    """
    Creates labeled segments for alive count tasks.
    
    Predicts number of alive teammates (4 excluding POV) or enemies (5).
    Output: Multi-class classification (0-4 for teammates, 0-5 for enemies).
    
    Label column: label (class index)
    """
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for alive count."""
        segment_length_sec = config['segment_length_sec']
        target_type = config.get('target_type', 'teammate')  # 'teammate' or 'enemy'
        
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        
        if len(player_trajectories) != 10:
            return []
        
        segments = []
        segment_ticks = segment_length_sec * self.tick_rate
        
        # For alive count, we want to include segments after deaths
        global_min_tick = min(df['tick'].min() for df in player_trajectories.values() if not df.empty)
        max_tick = max(df['tick'].max() for df in player_trajectories.values() if not df.empty)
        
        if max_tick - global_min_tick < segment_ticks:
            return []
        
        stride_ticks = int(self.stride_sec * self.tick_rate)
        current_tick = global_min_tick
        
        while current_tick + segment_ticks <= max_tick:
            end_tick = current_tick + segment_ticks
            middle_tick = current_tick + segment_ticks // 2
            
            all_players_data = []
            for steamid, df in player_trajectories.items():
                player_data = self._extract_player_data_at_tick(df, middle_tick)
                if player_data:
                    all_players_data.append(player_data)
            
            if len(all_players_data) < 2:  # Need at least some players
                current_tick += stride_ticks
                continue
            
            # Get unique sides present
            sides = set(p['side'] for p in all_players_data)
            
            for side in sides:
                # Find a POV player from this side who is alive
                pov_candidates = [p for p in all_players_data 
                                 if p['side'] == side and p.get('health', 0) > 0]
                
                if not pov_candidates:
                    continue
                
                pov_data = pov_candidates[0]
                pov_steamid = pov_data['steamid']
                pov_side = side
                
                if target_type == 'teammate':
                    # Count alive teammates (excluding POV) - 4 teammates max
                    targets = [p for p in all_players_data 
                              if p['side'] == pov_side and p['steamid'] != pov_steamid]
                    alive_count = sum(1 for p in targets if p.get('health', 0) > 0)
                else:  # enemy
                    # Count alive enemies - 5 enemies max
                    enemy_side = self._get_opposite_side(pov_side)
                    targets = [p for p in all_players_data if p['side'] == enemy_side]
                    alive_count = sum(1 for p in targets if p.get('health', 0) > 0)
                
                segment_info = {
                    'start_tick': current_tick - global_min_tick,  # Relative to round start
                    'end_tick': end_tick - global_min_tick,
                    'prediction_tick': middle_tick - global_min_tick,
                    'duration_seconds': segment_length_sec,
                    'pov_steamid': pov_steamid,
                    'pov_side': pov_side,
                    'alive_count': alive_count
                }
                segments.append(segment_info)
            
            current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for alive count."""
        output_rows = []
        idx = 0
        
        for segment in all_segments:
            row = {
                'idx': idx,
                'partition': segment['partition'],
                'pov_steamid': segment['pov_steamid'],
                'pov_side': segment['pov_side'],
                'seg_duration_sec': segment['duration_seconds'],
                'start_tick': segment['start_tick'],
                'end_tick': segment['end_tick'],
                'prediction_tick': segment['prediction_tick'],
                'match_id': segment['match_id'],
                'round_num': segment['round_num'],
                'label': segment['alive_count']
            }
            output_rows.append(row)
            idx += 1
        
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'])
            df = df.reset_index(drop=True)
            df['idx'] = range(len(df))
        
        return df

