"""
Additional bomb-related task creator.

Creates labels for:
- Will plant prediction (binary classification)
"""

import pandas as pd
from typing import Dict, List, Any, Optional

from .base_task_creator import TaskCreatorBase


class WillPlantPredictionCreator(TaskCreatorBase):
    """
    Creates labeled segments for will plant prediction task.
    
    Predicts whether the bomb will be planted in the round.
    Only creates segments for T-side before bomb is planted.
    Output: Binary classification (0=no plant, 1=will plant).
    """

    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for will plant prediction."""
        segment_length_sec = config['segment_length_sec']
        
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        bomb_df = self._load_bomb(match_id, round_num)
        
        if len(player_trajectories) < 1:
            return []
        
        # Check if bomb was planted in this round
        plant_events = bomb_df[bomb_df['event'] == 'plant']
        will_plant = not plant_events.empty
        plant_label = 1 if will_plant else 0
        
        # If plant happened, get plant tick to exclude segments after plant
        plant_tick = None
        if will_plant:
            plant_tick = plant_events.iloc[0]['tick']
        
        segments = []
        segment_ticks = segment_length_sec * self.tick_rate
        stride_ticks = int(self.stride_sec * self.tick_rate)
        
        # Load metadata to get freeze_end_tick
        metadata = self._load_metadata(match_id)
        if not metadata or 'rounds' not in metadata:
            return []
            
        round_info = next((r for r in metadata['rounds'] if r['round_number'] == round_num), None)
        if not round_info or 'freeze_end_tick' not in round_info:
            return []
            
        global_min_tick = round_info['freeze_end_tick']
        
        # Get map name
        map_name = 'de_mirage'
        for df in player_trajectories.values():
            if 'map_name' in df.columns and not df.empty:
                map_name = df.iloc[0]['map_name']
                break
        
        # For each T-side player as POV (only T-side predicts plant)
        for pov_steamid, pov_df in player_trajectories.items():
            if pov_df.empty:
                continue
            
            pov_side = pov_df.iloc[0]['side']
            if pov_side != 't':
                continue  # Skip CT players
            
            death_tick = self._find_player_death_tick(metadata, round_num, pov_steamid)
            global_max_tick = round_info.get('end_tick', pov_df['tick'].max())
            if death_tick is None:
                death_tick = global_max_tick
            else:
                death_tick = min(death_tick, global_max_tick)
            
            # Only use segments before plant (if plant happened)
            if plant_tick is not None:
                max_valid_tick = min(death_tick, plant_tick - segment_ticks)
            else:
                max_valid_tick = death_tick
            
            pov_min_tick = pov_df['tick'].min()
            
            current_tick = pov_min_tick
            
            while current_tick + segment_ticks <= max_valid_tick:
                end_tick = current_tick + segment_ticks
                middle_tick = current_tick + segment_ticks // 2
                
                # Verify POV player is alive
                segment_data = pov_df[(pov_df['tick'] >= current_tick) & (pov_df['tick'] <= end_tick)]
                if segment_data.empty or (segment_data['health'] <= 0).any():
                    current_tick += stride_ticks
                    continue
                
                segment_info = {
                    'start_tick': current_tick,
                    'end_tick': end_tick,
                    'prediction_tick': middle_tick,
                    'start_tick_norm': current_tick - global_min_tick,
                    'end_tick_norm': end_tick - global_min_tick,
                    'prediction_tick_norm': middle_tick - global_min_tick,
                    'start_seconds': current_tick / self.tick_rate,
                    'end_seconds': end_tick / self.tick_rate,
                    'prediction_seconds': middle_tick / self.tick_rate,
                    'normalized_start_seconds': (current_tick - global_min_tick) / self.tick_rate,
                    'normalized_end_seconds': (end_tick - global_min_tick) / self.tick_rate,
                    'normalized_prediction_seconds': (middle_tick - global_min_tick) / self.tick_rate,
                    'duration_seconds': segment_length_sec,
                    'map_name': map_name,
                    'pov_steamid': pov_steamid,
                    'pov_side': pov_side,
                    'will_plant': plant_label
                }
                segments.append(segment_info)
                
                current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for will plant prediction."""
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
                'map_name': segment['map_name'],
                'label_will_plant': segment['will_plant']
            }
            output_rows.append(row)
            idx += 1
        
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'])
            df = df.reset_index(drop=True)
            df['idx'] = range(len(df))
        
        return df
