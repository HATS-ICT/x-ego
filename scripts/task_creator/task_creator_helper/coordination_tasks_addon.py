"""
Additional team coordination task creators for teammate-specific predictions.

Creates labels for:
- Teammate movement direction (per teammate)
- Teammate speed (per teammate)
- Teammate weapon (per teammate)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

from .base_task_creator import TaskCreatorBase
from ..task_definitions import WEAPON_TO_IDX, NUM_WEAPONS


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


class TeammateWeaponCreator(TaskCreatorBase):
    """
    Creates labeled segments for teammate weapon classification task.
    
    Classifies the weapon each of the 4 teammates is currently using.
    Output: Multi-label classification (NUM_WEAPONS classes per teammate).
    """
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for teammate weapon classification."""
        segment_length_sec = config['segment_length_sec']
        lookback_sec = config.get('lookback_sec', 2.0)  # Look back 2 seconds for weapon info
        
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        shots_df = self._load_shots(match_id, round_num)
        damages_df = self._load_damages(match_id, round_num)
        
        if len(player_trajectories) < 2:
            return []
        
        segments = []
        segment_ticks = segment_length_sec * self.tick_rate
        lookback_ticks = int(lookback_sec * self.tick_rate)
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
                
                # Look for weapon info in recent window
                window_start = middle_tick - lookback_ticks
                window_end = middle_tick + lookback_ticks
                
                # Get teammate weapons
                teammate_weapons = []
                teammate_weapon_names = []
                teammate_steamids = []
                for steamid, df in player_trajectories.items():
                    if steamid == pov_steamid:
                        continue
                    
                    player_data = self._extract_player_data_at_tick(df, middle_tick)
                    
                    if player_data and \
                       player_data['side'] == pov_side and \
                       player_data.get('health', 0) > 0:
                        weapon = self._get_player_weapon(
                            steamid, shots_df, damages_df, window_start, window_end
                        )
                        
                        # Skip if no weapon info found
                        if weapon is None or weapon not in WEAPON_TO_IDX:
                            continue
                        
                        weapon_idx = WEAPON_TO_IDX[weapon]
                        teammate_weapons.append(weapon_idx)
                        teammate_weapon_names.append(weapon)
                        teammate_steamids.append(steamid)
                
                # Need exactly 4 teammates with weapon info
                if len(teammate_weapons) != 4:
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
                    'teammate_weapons': teammate_weapons,
                    'teammate_weapon_names': teammate_weapon_names
                }
                segments.append(segment_info)
                
                current_tick += stride_ticks
        
        return segments
    
    def _get_player_weapon(self, player_steamid: str, shots_df: pd.DataFrame, 
                          damages_df: pd.DataFrame, start_tick: int, end_tick: int) -> Optional[str]:
        """Get the weapon used by a player in the given tick window."""
        # First try shots (more reliable for weapon detection)
        if not shots_df.empty:
            player_shots = shots_df[
                (shots_df['player_steamid'].astype(str) == str(player_steamid)) &
                (shots_df['tick'] >= start_tick) &
                (shots_df['tick'] <= end_tick)
            ]
            if not player_shots.empty:
                # Get most recent shot's weapon
                most_recent = player_shots.sort_values('tick', ascending=False).iloc[0]
                weapon = most_recent.get('weapon', None)
                if weapon and pd.notna(weapon):
                    return str(weapon).lower()
        
        # Fall back to damages (attacker weapon)
        if not damages_df.empty:
            player_damages = damages_df[
                (damages_df['attacker_steamid'].astype(str) == str(player_steamid)) &
                (damages_df['tick'] >= start_tick) &
                (damages_df['tick'] <= end_tick)
            ]
            if not player_damages.empty:
                most_recent = player_damages.sort_values('tick', ascending=False).iloc[0]
                weapon = most_recent.get('weapon', None)
                if weapon and pd.notna(weapon):
                    return str(weapon).lower()
        
        return None
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for teammate weapon classification."""
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
            # Add teammate steamids, weapon names, and weapon indices
            for i, (steamid, weapon_name, weapon_idx) in enumerate(zip(
                segment['teammate_steamids'], 
                segment['teammate_weapon_names'],
                segment['teammate_weapons']
            )):
                row[f'teammate_{i}_steamid'] = steamid
                row[f'weapon_name_{i}'] = weapon_name
                row[f'label_weapon_{i}'] = weapon_idx
            
            output_rows.append(row)
            idx += 1
        
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'])
            df = df.reset_index(drop=True)
            df['idx'] = range(len(df))
        
        return df
