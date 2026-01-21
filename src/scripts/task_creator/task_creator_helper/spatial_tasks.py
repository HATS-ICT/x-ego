"""
Spatial/POV task creators.

Creates labels for:
- POV movement direction
- POV speed estimation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any

from .base_task_creator import TaskCreatorBase


class POVMovementDirectionCreator(TaskCreatorBase):
    """
    Creates labeled segments for POV movement direction task.
    
    Predicts the movement direction of the POV player.
    Output: Multi-class classification (9 classes: 8 directions + stationary).
    """
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for POV movement direction."""
        segment_length_sec = config['segment_length_sec']
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        
        if len(player_trajectories) < 1:
            return []
        
        segments = []
        segment_ticks = int(segment_length_sec * self.tick_rate)
        stride_ticks = int(self.stride_sec * self.tick_rate)
        stride_ticks = max(1, stride_ticks)
        
        # Get global tick range
        all_min_ticks = []
        for df in player_trajectories.values():
            if not df.empty:
                all_min_ticks.append(int(df['tick'].min()))
        
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
                death_tick = int(pov_df['tick'].max())
            
            pov_min_tick = int(pov_df['tick'].min())
            pov_side = pov_df.iloc[0]['side']
            
            current_tick = pov_min_tick
            
            while current_tick + segment_ticks <= death_tick:
                end_tick = current_tick + segment_ticks
                middle_tick = current_tick + segment_ticks // 2
                
                # Calculate prev_tick for movement computation
                prev_tick = max(pov_min_tick, middle_tick - lookback_ticks)
                
                # Verify POV player is alive
                segment_data = pov_df[(pov_df['tick'] >= current_tick) & (pov_df['tick'] <= end_tick)]
                if segment_data.empty or (segment_data['health'] <= 0).any():
                    current_tick += stride_ticks
                    continue
                
                # Get POV current and previous data
                pov_curr = self._extract_player_data_at_tick(pov_df, middle_tick)
                pov_prev = self._extract_player_data_at_tick(pov_df, prev_tick)
                if not pov_curr or not pov_prev:
                    current_tick += stride_ticks
                    continue
                
                # Compute POV movement direction
                direction_idx = self._compute_movement_direction(pov_prev, pov_curr)
                
                segment_info = {
                    'start_tick': current_tick - global_min_tick,
                    'end_tick': end_tick - global_min_tick,
                    'prediction_tick': middle_tick - global_min_tick,
                    'duration_seconds': segment_length_sec,
                    'map_name': map_name,
                    'pov_steamid': pov_steamid,
                    'pov_side': pov_side,
                    'direction_idx': direction_idx
                }
                segments.append(segment_info)
                
                current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for POV movement direction."""
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
                'label': segment['direction_idx']
            }
            output_rows.append(row)
            idx += 1
        
        # Create DataFrame with proper columns even if empty
        columns = ['idx', 'partition', 'pov_steamid', 'pov_side', 'seg_duration_sec', 
                  'start_tick', 'end_tick', 'prediction_tick', 'match_id', 'round_num', 
                  'map_name', 'label']
        df = pd.DataFrame(output_rows, columns=columns)
        
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'])
            df = df.reset_index(drop=True)
            df['idx'] = range(len(df))
        
        return df


class POVSpeedCreator(TaskCreatorBase):
    """
    Creates labeled segments for POV speed estimation task.
    
    Predicts the movement speed of the POV player.
    Output: Regression with 1 output (speed in normalized coords per second).
    """
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for POV speed estimation."""
        segment_length_sec = config['segment_length_sec']
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        
        if len(player_trajectories) < 1:
            return []
        
        segments = []
        segment_ticks = int(segment_length_sec * self.tick_rate)
        stride_ticks = int(self.stride_sec * self.tick_rate)
        stride_ticks = max(1, stride_ticks)
        
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
                
                # Calculate prev_tick with minimum gap for meaningful movement
                min_gap_ticks = max(1, int(0.1 * self.tick_rate))  # At least 0.1 seconds
                prev_tick = max(pov_min_tick, middle_tick - lookback_ticks)
                
                # Ensure prev_tick is at least min_gap_ticks before middle_tick
                if prev_tick >= middle_tick - min_gap_ticks:
                    prev_tick = max(pov_min_tick, middle_tick - min_gap_ticks)
                    # If we can't get enough history, skip this segment
                    if prev_tick >= middle_tick:
                        current_tick += stride_ticks
                        continue
                
                # Recalculate lookback_sec based on actual time difference
                actual_lookback_sec = (middle_tick - prev_tick) / self.tick_rate
                if actual_lookback_sec <= 0:
                    current_tick += stride_ticks
                    continue
                
                # Verify POV player is alive
                segment_data = pov_df[(pov_df['tick'] >= current_tick) & (pov_df['tick'] <= end_tick)]
                if segment_data.empty or (segment_data['health'] <= 0).any():
                    current_tick += stride_ticks
                    continue
                
                # Get POV current and previous data
                pov_curr = self._extract_player_data_at_tick(pov_df, middle_tick)
                pov_prev = self._extract_player_data_at_tick(pov_df, prev_tick)
                if not pov_curr or not pov_prev:
                    current_tick += stride_ticks
                    continue
                
                # Compute POV speed (distance / time)
                dx = pov_curr.get('X_norm', 0) - pov_prev.get('X_norm', 0)
                dy = pov_curr.get('Y_norm', 0) - pov_prev.get('Y_norm', 0)
                dz = pov_curr.get('Z_norm', 0) - pov_prev.get('Z_norm', 0)
                
                distance = np.sqrt(dx**2 + dy**2 + dz**2)
                speed = distance / actual_lookback_sec  # normalized coords per second
                
                segment_info = {
                    'start_tick': current_tick - global_min_tick,
                    'end_tick': end_tick - global_min_tick,
                    'prediction_tick': middle_tick - global_min_tick,
                    'duration_seconds': segment_length_sec,
                    'map_name': map_name,
                    'pov_steamid': pov_steamid,
                    'pov_side': pov_side,
                    'speed': speed
                }
                segments.append(segment_info)
                
                current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for POV speed estimation."""
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
                'label': segment['speed']
            }
            output_rows.append(row)
            idx += 1
        
        # Create DataFrame with proper columns even if empty
        columns = ['idx', 'partition', 'pov_steamid', 'pov_side', 'seg_duration_sec', 
                  'start_tick', 'end_tick', 'prediction_tick', 'match_id', 'round_num', 
                  'map_name', 'label']
        df = pd.DataFrame(output_rows, columns=columns)
        
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'])
            df = df.reset_index(drop=True)
            df['idx'] = range(len(df))
        
        return df


# Aliases for renamed tasks (self instead of pov)
SelfMovementDirectionCreator = POVMovementDirectionCreator
SelfSpeedCreator = POVSpeedCreator
