"""
Additional team coordination task creators for teammate-specific predictions.

Creates labels for:
- Teammate movement direction (per teammate)
- Teammate speed (per teammate)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

from .base_task_creator import TaskCreatorBase


class TeammateMovementDirectionCreator(TaskCreatorBase):
    """
    Creates labeled segments for teammate movement direction task.
    
    Predicts the movement direction of each of the 4 teammates.
    Output: Multi-label classification (9 classes per teammate: 8 directions + stationary).
    """
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for teammate movement direction."""
        segment_length_sec = config['segment_length_sec']
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        
        if len(player_trajectories) < 2:
            return []
        
        segments = []
        segment_ticks = segment_length_sec * self.tick_rate
        stride_ticks = int(self.stride_sec * self.tick_rate)
        
        # Get global tick range
        all_min_ticks = []
        for df in player_trajectories.values():
            if not df.empty:
                all_min_ticks.append(df['tick'].min())
        
        if not all_min_ticks:
            return []
        
        global_min_tick = min(all_min_ticks)
        
        # Get map name
        map_name = 'de_mirage'
        for df in player_trajectories.values():
            if 'map_name' in df.columns and not df.empty:
                map_name = df.iloc[0]['map_name']
                break
        
        # Need some history for movement computation
        lookback_ticks = int(0.5 * self.tick_rate)  # 0.5 second lookback
        
        # For each player as POV
        for pov_steamid, pov_df in player_trajectories.items():
            if pov_df.empty:
                continue
            
            death_tick = self._find_player_death_tick(pov_df)
            if death_tick is None:
                death_tick = pov_df['tick'].max()
            
            pov_min_tick = pov_df['tick'].min()
            pov_side = pov_df.iloc[0]['side']
            
            current_tick = pov_min_tick
            
            while current_tick + segment_ticks <= death_tick:
                end_tick = current_tick + segment_ticks
                middle_tick = current_tick + segment_ticks // 2
                prev_tick = max(global_min_tick, middle_tick - lookback_ticks)
                
                # Verify POV player is alive
                segment_data = pov_df[(pov_df['tick'] >= current_tick) & (pov_df['tick'] <= end_tick)]
                if segment_data.empty or (segment_data['health'] <= 0).any():
                    current_tick += stride_ticks
                    continue
                
                # Get POV player data
                pov_data = self._extract_player_data_at_tick(pov_df, middle_tick)
                if not pov_data:
                    current_tick += stride_ticks
                    continue
                
                # Get teammate movement directions
                teammate_directions = []
                teammate_steamids = []
                for steamid, df in player_trajectories.items():
                    if steamid == pov_steamid:
                        continue
                    
                    player_curr = self._extract_player_data_at_tick(df, middle_tick)
                    player_prev = self._extract_player_data_at_tick(df, prev_tick)
                    
                    if player_curr and player_prev and \
                       player_curr['side'] == pov_side and \
                       player_curr.get('health', 0) > 0:
                        direction_idx = self._compute_movement_direction(player_prev, player_curr)
                        teammate_directions.append(direction_idx)
                        teammate_steamids.append(steamid)
                
                # Need exactly 4 teammates
                if len(teammate_directions) != 4:
                    current_tick += stride_ticks
                    continue
                
                segment_info = {
                    'start_tick': current_tick - global_min_tick,
                    'end_tick': end_tick - global_min_tick,
                    'prediction_tick': middle_tick - global_min_tick,
                    'duration_seconds': segment_length_sec,
                    'map_name': map_name,
                    'pov_steamid': pov_steamid,
                    'pov_side': pov_side,
                    'teammate_steamids': teammate_steamids,
                    'teammate_directions': teammate_directions
                }
                segments.append(segment_info)
                
                current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for teammate movement direction."""
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
                'map_name': segment['map_name'],
            }
            # Add teammate steamids and directions
            for i, (steamid, direction) in enumerate(zip(segment['teammate_steamids'], segment['teammate_directions'])):
                row[f'teammate_{i}_steamid'] = steamid
                row[f'label_direction_{i}'] = direction
            
            output_rows.append(row)
            idx += 1
        
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'])
            df = df.reset_index(drop=True)
            df['idx'] = range(len(df))
        
        return df


class TeammateSpeedCreator(TaskCreatorBase):
    """
    Creates labeled segments for teammate speed estimation task.
    
    Predicts the movement speed of each of the 4 teammates.
    Output: Regression with 4 outputs (one speed per teammate).
    """
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for teammate speed estimation."""
        segment_length_sec = config['segment_length_sec']
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        
        if len(player_trajectories) < 2:
            return []
        
        segments = []
        segment_ticks = segment_length_sec * self.tick_rate
        stride_ticks = int(self.stride_sec * self.tick_rate)
        
        # Get global tick range
        all_min_ticks = []
        for df in player_trajectories.values():
            if not df.empty:
                all_min_ticks.append(df['tick'].min())
        
        if not all_min_ticks:
            return []
        
        global_min_tick = min(all_min_ticks)
        
        # Get map name
        map_name = 'de_mirage'
        for df in player_trajectories.values():
            if 'map_name' in df.columns and not df.empty:
                map_name = df.iloc[0]['map_name']
                break
        
        # Need some history for speed computation
        lookback_ticks = int(0.5 * self.tick_rate)  # 0.5 second lookback
        lookback_sec = 0.5
        
        # For each player as POV
        for pov_steamid, pov_df in player_trajectories.items():
            if pov_df.empty:
                continue
            
            death_tick = self._find_player_death_tick(pov_df)
            if death_tick is None:
                death_tick = pov_df['tick'].max()
            
            pov_min_tick = pov_df['tick'].min()
            pov_side = pov_df.iloc[0]['side']
            
            current_tick = pov_min_tick
            
            while current_tick + segment_ticks <= death_tick:
                end_tick = current_tick + segment_ticks
                middle_tick = current_tick + segment_ticks // 2
                prev_tick = max(global_min_tick, middle_tick - lookback_ticks)
                
                # Verify POV player is alive
                segment_data = pov_df[(pov_df['tick'] >= current_tick) & (pov_df['tick'] <= end_tick)]
                if segment_data.empty or (segment_data['health'] <= 0).any():
                    current_tick += stride_ticks
                    continue
                
                # Get POV player data
                pov_data = self._extract_player_data_at_tick(pov_df, middle_tick)
                if not pov_data:
                    current_tick += stride_ticks
                    continue
                
                # Get teammate speeds
                teammate_speeds = []
                teammate_steamids = []
                for steamid, df in player_trajectories.items():
                    if steamid == pov_steamid:
                        continue
                    
                    player_curr = self._extract_player_data_at_tick(df, middle_tick)
                    player_prev = self._extract_player_data_at_tick(df, prev_tick)
                    
                    if player_curr and player_prev and \
                       player_curr['side'] == pov_side and \
                       player_curr.get('health', 0) > 0:
                        # Compute speed
                        dx = player_curr.get('X_norm', 0) - player_prev.get('X_norm', 0)
                        dy = player_curr.get('Y_norm', 0) - player_prev.get('Y_norm', 0)
                        dz = player_curr.get('Z_norm', 0) - player_prev.get('Z_norm', 0)
                        
                        distance = np.sqrt(dx**2 + dy**2 + dz**2)
                        speed = distance / lookback_sec
                        
                        teammate_speeds.append(speed)
                        teammate_steamids.append(steamid)
                
                # Need exactly 4 teammates
                if len(teammate_speeds) != 4:
                    current_tick += stride_ticks
                    continue
                
                segment_info = {
                    'start_tick': current_tick - global_min_tick,
                    'end_tick': end_tick - global_min_tick,
                    'prediction_tick': middle_tick - global_min_tick,
                    'duration_seconds': segment_length_sec,
                    'map_name': map_name,
                    'pov_steamid': pov_steamid,
                    'pov_side': pov_side,
                    'teammate_steamids': teammate_steamids,
                    'teammate_speeds': teammate_speeds
                }
                segments.append(segment_info)
                
                current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for teammate speed estimation."""
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
                'map_name': segment['map_name'],
            }
            # Add teammate steamids and speeds
            for i, (steamid, speed) in enumerate(zip(segment['teammate_steamids'], segment['teammate_speeds'])):
                row[f'teammate_{i}_steamid'] = steamid
                row[f'label_speed_{i}'] = speed
            
            output_rows.append(row)
            idx += 1
        
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'])
            df = df.reset_index(drop=True)
            df['idx'] = range(len(df))
        
        return df
