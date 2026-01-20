"""
Action/equipment task creators.

Creates labels for:
- Weapon in use classification
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional

from .base_task_creator import TaskCreatorBase
from ..task_definitions import WEAPON_TO_IDX, NUM_WEAPONS


class WeaponInUseCreator(TaskCreatorBase):
    """
    Creates labeled segments for weapon in use classification task.
    
    Classifies the weapon the POV player is currently using based on
    recent shots or damage events.
    Output: Multi-class classification (NUM_WEAPONS classes).
    """
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for weapon in use classification."""
        segment_length_sec = config['segment_length_sec']
        lookback_sec = config.get('lookback_sec', 2.0)  # Look back 2 seconds for weapon info
        
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        shots_df = self._load_shots(match_id, round_num)
        damages_df = self._load_damages(match_id, round_num)
        
        if len(player_trajectories) < 1:
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
                
                # Look for weapon info in recent window
                window_start = middle_tick - lookback_ticks
                window_end = middle_tick + lookback_ticks
                
                weapon = self._get_pov_weapon(
                    pov_steamid, shots_df, damages_df, window_start, window_end
                )
                
                # Skip if no weapon info found
                if weapon is None or weapon not in WEAPON_TO_IDX:
                    current_tick += stride_ticks
                    continue
                
                weapon_idx = WEAPON_TO_IDX[weapon]
                
                segment_info = {
                    'start_tick': current_tick - global_min_tick,
                    'end_tick': end_tick - global_min_tick,
                    'prediction_tick': middle_tick - global_min_tick,
                    'duration_seconds': segment_length_sec,
                    'map_name': map_name,
                    'pov_steamid': pov_steamid,
                    'pov_side': pov_side,
                    'weapon': weapon,
                    'weapon_idx': weapon_idx
                }
                segments.append(segment_info)
                
                current_tick += stride_ticks
        
        return segments
    
    def _get_pov_weapon(self, pov_steamid: str, shots_df: pd.DataFrame, 
                        damages_df: pd.DataFrame, start_tick: int, end_tick: int) -> Optional[str]:
        """Get the weapon used by POV player in the given tick window."""
        # First try shots (more reliable for weapon detection)
        if not shots_df.empty:
            pov_shots = shots_df[
                (shots_df['player_steamid'].astype(str) == str(pov_steamid)) &
                (shots_df['tick'] >= start_tick) &
                (shots_df['tick'] <= end_tick)
            ]
            if not pov_shots.empty:
                # Get most recent shot's weapon
                most_recent = pov_shots.sort_values('tick', ascending=False).iloc[0]
                weapon = most_recent.get('weapon', None)
                if weapon and pd.notna(weapon):
                    return str(weapon).lower()
        
        # Fall back to damages (attacker weapon)
        if not damages_df.empty:
            pov_damages = damages_df[
                (damages_df['attacker_steamid'].astype(str) == str(pov_steamid)) &
                (damages_df['tick'] >= start_tick) &
                (damages_df['tick'] <= end_tick)
            ]
            if not pov_damages.empty:
                most_recent = pov_damages.sort_values('tick', ascending=False).iloc[0]
                weapon = most_recent.get('weapon', None)
                if weapon and pd.notna(weapon):
                    return str(weapon).lower()
        
        return None
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for weapon in use classification."""
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
                'weapon_name': segment['weapon'],
                'label_weapon': segment['weapon_idx']
            }
            output_rows.append(row)
            idx += 1
        
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'])
            df = df.reset_index(drop=True)
            df['idx'] = range(len(df))
        
        return df
