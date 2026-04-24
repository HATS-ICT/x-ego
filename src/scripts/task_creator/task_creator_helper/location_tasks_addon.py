"""
Additional location prediction task creator.

Creates labels for:
- Self location nowcast

Label column naming convention:
- multi_cls: label (single column with class index)
"""

import pandas as pd
from typing import Dict, List, Any

from .base_task_creator import TaskCreatorBase
from ..task_definitions import get_place_to_idx_for_map


class SelfLocationNowcastCreator(TaskCreatorBase):
    """
    Creates labeled segments for self location nowcast task.
    
    For each POV player, predicts their own current location (place).
    Output: Multi-class classification over NUM_PLACES places.
    
    Label column: label (class index 0-22)
    """

    def _get_place_mapping(self, player_trajectories: Dict[str, pd.DataFrame]) -> dict:
        for df in player_trajectories.values():
            if not df.empty and 'map_name' in df.columns:
                return get_place_to_idx_for_map(df.iloc[0]['map_name'])
        return get_place_to_idx_for_map(None)
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for self location nowcast."""
        segment_length_sec = config['segment_length_sec']
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        
        if len(player_trajectories) < 1:
            return []

        place_to_idx = self._get_place_mapping(player_trajectories)
        
        segments = []
        segment_ticks = segment_length_sec * self.tick_rate
        stride_ticks = int(self.stride_sec * self.tick_rate)
        stride_ticks = max(1, stride_ticks)
        
        # Load metadata to get freeze_end_tick
        metadata = self._load_metadata(match_id)
        if not metadata or 'rounds' not in metadata:
            return []
            
        round_info = next((r for r in metadata['rounds'] if r['round_number'] == round_num), None)
        if not round_info or 'freeze_end_tick' not in round_info:
            return []
            
        global_min_tick = round_info['freeze_end_tick']
        
        # For each player as POV, create segments while they are alive
        for pov_steamid, pov_df in player_trajectories.items():
            if pov_df.empty:
                continue
            
            # Find death tick for POV player
            death_tick = self._find_player_death_tick(metadata, round_num, pov_steamid)
            global_max_tick = round_info.get('end_tick', pov_df['tick'].max())
            if death_tick is None:
                death_tick = global_max_tick
            else:
                death_tick = min(death_tick, global_max_tick)
            
            pov_min_tick = pov_df['tick'].min()
            pov_side = pov_df.iloc[0]['side']
            
            current_tick = pov_min_tick
            
            while current_tick + segment_ticks <= death_tick:
                end_tick = current_tick + segment_ticks
                middle_tick = current_tick + segment_ticks // 2
                
                # Verify POV player is alive throughout segment
                segment_data = pov_df[(pov_df['tick'] >= current_tick) & (pov_df['tick'] <= end_tick)]
                if segment_data.empty or (segment_data['health'] <= 0).any():
                    current_tick += stride_ticks
                    continue
                
                # Get POV player data at middle tick
                pov_data = self._extract_player_data_at_tick(pov_df, middle_tick)
                if not pov_data:
                    current_tick += stride_ticks
                    continue
                
                # Get POV player's place
                place = pov_data.get('place', None)
                if not place or place not in place_to_idx:
                    current_tick += stride_ticks
                    continue
                
                place_idx = place_to_idx[place]
                
                segment_info = {
                    'start_tick': current_tick,
                    'end_tick': end_tick,
                    'prediction_tick': middle_tick,
                    'start_tick_norm': current_tick - global_min_tick,
                    'end_tick_norm': end_tick - global_min_tick,
                    'prediction_tick_norm': middle_tick - global_min_tick,
                    'duration_seconds': segment_length_sec,
                    'pov_steamid': pov_steamid,
                    'pov_side': pov_side,
                    'place_idx': place_idx
                }
                segments.append(segment_info)
                
                current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for self location nowcast."""
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
                'start_tick_norm': segment['start_tick_norm'],
                'end_tick_norm': segment['end_tick_norm'],
                'prediction_tick_norm': segment['prediction_tick_norm'],
                'match_id': segment['match_id'],
                'round_num': segment['round_num'],
                'label': segment['place_idx']
            }
            output_rows.append(row)
            idx += 1
        
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'])
            df = df.reset_index(drop=True)
            df['idx'] = range(len(df))
        
        return df
