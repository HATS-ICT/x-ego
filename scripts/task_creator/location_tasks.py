"""
Location prediction task creators.

Creates labels for:
- Teammate location nowcast/forecast
- Enemy location nowcast/forecast
- Self location forecast
- Coordinate prediction (regression)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.task_creator.base_task_creator import TaskCreatorBase
from scripts.task_creator.task_definitions import PLACE_TO_IDX, NUM_PLACES


class TeammateLocationNowcastCreator(TaskCreatorBase):
    """
    Creates labeled segments for teammate location nowcast task.
    
    For each POV player, predicts the 4 teammates' current locations (places).
    Output: Multi-label classification over NUM_PLACES places.
    """
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for teammate location nowcast."""
        segment_length_sec = config['segment_length_sec']
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        
        if len(player_trajectories) < 1:
            return []
        
        segments = []
        segment_ticks = segment_length_sec * self.tick_rate
        stride_ticks = int(self.stride_sec * self.tick_rate)
        stride_ticks = max(1, stride_ticks)
        
        # Get global tick range
        all_min_ticks = []
        all_max_ticks = []
        for df in player_trajectories.values():
            if not df.empty:
                all_min_ticks.append(df['tick'].min())
                all_max_ticks.append(df['tick'].max())
        
        if not all_min_ticks:
            return []
        
        global_min_tick = min(all_min_ticks)
        global_max_tick = max(all_max_ticks)
        
        # Get map name
        map_name = 'de_mirage'
        for df in player_trajectories.values():
            if 'map_name' in df.columns and not df.empty:
                map_name = df.iloc[0]['map_name']
                break
        
        # For each player as POV, create segments while they are alive
        for pov_steamid, pov_df in player_trajectories.items():
            if pov_df.empty:
                continue
            
            # Find death tick for POV player
            death_tick = self._find_player_death_tick(pov_df)
            if death_tick is None:
                death_tick = pov_df['tick'].max()
            
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
                
                # Get POV player data
                pov_data = self._extract_player_data_at_tick(pov_df, middle_tick)
                if not pov_data:
                    current_tick += stride_ticks
                    continue
                
                # Get alive teammates at this tick
                teammates = []
                for steamid, df in player_trajectories.items():
                    if steamid == pov_steamid:
                        continue
                    player_data = self._extract_player_data_at_tick(df, middle_tick)
                    if player_data and player_data['side'] == pov_side and player_data.get('health', 0) > 0:
                        teammates.append(player_data)
                
                # Need at least some teammates (but not necessarily all 4)
                if len(teammates) == 0:
                    current_tick += stride_ticks
                    continue
                
                # Sort teammates by steamid for consistency
                teammates.sort(key=lambda x: x['steamid'])
                
                # Create multi-label target (which places are occupied by alive teammates)
                place_labels = np.zeros(NUM_PLACES, dtype=np.float32)
                for tm in teammates:
                    place = tm.get('place', '')
                    if place in PLACE_TO_IDX:
                        place_labels[PLACE_TO_IDX[place]] = 1.0
                
                segment_info = {
                    'start_tick': current_tick - global_min_tick,  # Relative to round start
                    'end_tick': end_tick - global_min_tick,
                    'prediction_tick': middle_tick - global_min_tick,
                    'duration_seconds': segment_length_sec,
                    'map_name': map_name,
                    'pov_steamid': pov_steamid,
                    'pov_side': pov_side,
                    'teammates_data': teammates,
                    'place_labels': place_labels.tolist(),
                    'num_alive_teammates': len(teammates)
                }
                segments.append(segment_info)
                
                current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for teammate location nowcast."""
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
                'map_name': segment['map_name']
            }
            
            # Add teammate data
            for i, tm in enumerate(segment['teammates_data']):
                row[f'teammate_{i}_id'] = tm['steamid']
                row[f'teammate_{i}_name'] = tm['name']
                row[f'teammate_{i}_X'] = tm['X_norm']
                row[f'teammate_{i}_Y'] = tm['Y_norm']
                row[f'teammate_{i}_Z'] = tm['Z_norm']
                row[f'teammate_{i}_place'] = tm['place']
            
            # Add multi-label targets
            for i, label in enumerate(segment['place_labels']):
                row[f'label_place_{i}'] = label
            
            output_rows.append(row)
            idx += 1
        
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'])
            df = df.reset_index(drop=True)
            df['idx'] = range(len(df))
        
        return df


class EnemyLocationNowcastCreator(TaskCreatorBase):
    """
    Creates labeled segments for enemy location nowcast task.
    
    For each POV player, predicts the 5 enemies' current locations (places).
    Output: Multi-label classification over NUM_PLACES places.
    """
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for enemy location nowcast."""
        segment_length_sec = config['segment_length_sec']
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        
        if len(player_trajectories) < 1:
            return []
        
        segments = []
        segment_ticks = segment_length_sec * self.tick_rate
        stride_ticks = int(self.stride_sec * self.tick_rate)
        stride_ticks = max(1, stride_ticks)
        
        # Get global tick range
        all_min_ticks = []
        all_max_ticks = []
        for df in player_trajectories.values():
            if not df.empty:
                all_min_ticks.append(df['tick'].min())
                all_max_ticks.append(df['tick'].max())
        
        if not all_min_ticks:
            return []
        
        global_min_tick = min(all_min_ticks)
        
        # Get map name
        map_name = 'de_mirage'
        for df in player_trajectories.values():
            if 'map_name' in df.columns and not df.empty:
                map_name = df.iloc[0]['map_name']
                break
        
        # For each player as POV, create segments while they are alive
        for pov_steamid, pov_df in player_trajectories.items():
            if pov_df.empty:
                continue
            
            death_tick = self._find_player_death_tick(pov_df)
            if death_tick is None:
                death_tick = pov_df['tick'].max()
            
            pov_min_tick = pov_df['tick'].min()
            pov_side = pov_df.iloc[0]['side']
            enemy_side = self._get_opposite_side(pov_side)
            
            current_tick = pov_min_tick
            
            while current_tick + segment_ticks <= death_tick:
                end_tick = current_tick + segment_ticks
                middle_tick = current_tick + segment_ticks // 2
                
                # Verify POV player is alive
                segment_data = pov_df[(pov_df['tick'] >= current_tick) & (pov_df['tick'] <= end_tick)]
                if segment_data.empty or (segment_data['health'] <= 0).any():
                    current_tick += stride_ticks
                    continue
                
                # Get alive enemies at this tick
                enemies = []
                for steamid, df in player_trajectories.items():
                    if steamid == pov_steamid:
                        continue
                    player_data = self._extract_player_data_at_tick(df, middle_tick)
                    if player_data and player_data['side'] == enemy_side and player_data.get('health', 0) > 0:
                        enemies.append(player_data)
                
                # Need at least some enemies
                if len(enemies) == 0:
                    current_tick += stride_ticks
                    continue
                
                enemies.sort(key=lambda x: x['steamid'])
                
                # Create multi-label target for alive enemies
                place_labels = np.zeros(NUM_PLACES, dtype=np.float32)
                for en in enemies:
                    place = en.get('place', '')
                    if place in PLACE_TO_IDX:
                        place_labels[PLACE_TO_IDX[place]] = 1.0
                
                segment_info = {
                    'start_tick': current_tick - global_min_tick,  # Relative to round start
                    'end_tick': end_tick - global_min_tick,
                    'prediction_tick': middle_tick - global_min_tick,
                    'duration_seconds': segment_length_sec,
                    'map_name': map_name,
                    'pov_steamid': pov_steamid,
                    'pov_side': pov_side,
                    'enemies_data': enemies,
                    'place_labels': place_labels.tolist(),
                    'num_alive_enemies': len(enemies)
                }
                segments.append(segment_info)
                
                current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for enemy location nowcast."""
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
                'map_name': segment['map_name']
            }
            
            # Add enemy data
            for i, en in enumerate(segment['enemies_data']):
                row[f'enemy_{i}_id'] = en['steamid']
                row[f'enemy_{i}_name'] = en['name']
                row[f'enemy_{i}_X'] = en['X_norm']
                row[f'enemy_{i}_Y'] = en['Y_norm']
                row[f'enemy_{i}_Z'] = en['Z_norm']
                row[f'enemy_{i}_place'] = en['place']
            
            # Add multi-label targets
            for i, label in enumerate(segment['place_labels']):
                row[f'label_place_{i}'] = label
            
            output_rows.append(row)
            idx += 1
        
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'])
            df = df.reset_index(drop=True)
            df['idx'] = range(len(df))
        
        return df


class TeammateCoordinateNowcastCreator(TaskCreatorBase):
    """
    Creates labeled segments for teammate coordinate nowcast task.
    
    For each POV player, predicts the 4 teammates' normalized XYZ coordinates.
    Output: Regression with 12 outputs (4 teammates x 3 coordinates).
    """
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for teammate coordinate nowcast."""
        segment_length_sec = config['segment_length_sec']
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        
        if len(player_trajectories) < 1:
            return []
        
        segments = []
        segment_ticks = segment_length_sec * self.tick_rate
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
                
                # Get alive teammates at this tick
                teammates = []
                for steamid, df in player_trajectories.items():
                    if steamid == pov_steamid:
                        continue
                    player_data = self._extract_player_data_at_tick(df, middle_tick)
                    if player_data and player_data['side'] == pov_side and player_data.get('health', 0) > 0:
                        teammates.append(player_data)
                
                if len(teammates) == 0:
                    current_tick += stride_ticks
                    continue
                
                teammates.sort(key=lambda x: x['steamid'])
                
                # Create coordinate labels for alive teammates (variable number)
                coord_labels = []
                for tm in teammates:
                    coord_labels.extend([
                        tm.get('X_norm', 0.5),
                        tm.get('Y_norm', 0.5),
                        tm.get('Z_norm', 0.5)
                    ])
                
                segment_info = {
                    'start_tick': current_tick - global_min_tick,  # Relative to round start
                    'end_tick': end_tick - global_min_tick,
                    'prediction_tick': middle_tick - global_min_tick,
                    'duration_seconds': segment_length_sec,
                    'map_name': map_name,
                    'pov_steamid': pov_steamid,
                    'pov_side': pov_side,
                    'teammates_data': teammates,
                    'coord_labels': coord_labels,
                    'num_alive_teammates': len(teammates)
                }
                segments.append(segment_info)
                
                current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for teammate coordinate nowcast."""
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
                'map_name': segment['map_name']
            }
            
            # Add teammate info
            for i, tm in enumerate(segment['teammates_data']):
                row[f'teammate_{i}_id'] = tm['steamid']
                row[f'teammate_{i}_name'] = tm['name']
            
            # Add coordinate labels
            for i, coord in enumerate(segment['coord_labels']):
                row[f'label_coord_{i}'] = coord
            
            output_rows.append(row)
            idx += 1
        
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'])
            df = df.reset_index(drop=True)
            df['idx'] = range(len(df))
        
        return df


class LocationForecastCreator(TaskCreatorBase):
    """
    Creates labeled segments for location forecast tasks.
    
    Predicts future locations (places) at specified horizon.
    Can be configured for self, teammate, or enemy prediction.
    """
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for location forecast."""
        segment_length_sec = config['segment_length_sec']
        forecast_horizon_sec = config['forecast_horizon_sec']
        target_type = config.get('target_type', 'teammate')  # 'self', 'teammate', or 'enemy'
        
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        
        if len(player_trajectories) < 1:
            return []
        
        segments = []
        segment_ticks = segment_length_sec * self.tick_rate
        horizon_ticks = int(forecast_horizon_sec * self.tick_rate)
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
        
        # For each player as POV
        for pov_steamid, pov_df in player_trajectories.items():
            if pov_df.empty:
                continue
            
            death_tick = self._find_player_death_tick(pov_df)
            if death_tick is None:
                death_tick = pov_df['tick'].max()
            
            pov_min_tick = pov_df['tick'].min()
            pov_side = pov_df.iloc[0]['side']
            enemy_side = self._get_opposite_side(pov_side)
            
            current_tick = pov_min_tick
            
            # For forecast, POV must be alive through segment and prediction tick
            while current_tick + segment_ticks + horizon_ticks <= death_tick:
                end_tick = current_tick + segment_ticks
                prediction_tick = end_tick + horizon_ticks
                middle_tick = current_tick + segment_ticks // 2
                
                # Verify POV player is alive through prediction tick
                segment_data = pov_df[(pov_df['tick'] >= current_tick) & (pov_df['tick'] <= prediction_tick)]
                if segment_data.empty or (segment_data['health'] <= 0).any():
                    current_tick += stride_ticks
                    continue
                
                # Get alive targets at prediction tick (future)
                if target_type == 'self':
                    pov_future = self._extract_player_data_at_tick(pov_df, prediction_tick)
                    targets = [pov_future] if pov_future and pov_future.get('health', 0) > 0 else []
                elif target_type == 'teammate':
                    targets = []
                    for steamid, df in player_trajectories.items():
                        if steamid == pov_steamid:
                            continue
                        player_data = self._extract_player_data_at_tick(df, prediction_tick)
                        if player_data and player_data['side'] == pov_side and player_data.get('health', 0) > 0:
                            targets.append(player_data)
                else:  # enemy
                    targets = []
                    for steamid, df in player_trajectories.items():
                        player_data = self._extract_player_data_at_tick(df, prediction_tick)
                        if player_data and player_data['side'] == enemy_side and player_data.get('health', 0) > 0:
                            targets.append(player_data)
                
                # Need at least some targets (for self, need exactly 1)
                if target_type == 'self' and len(targets) != 1:
                    current_tick += stride_ticks
                    continue
                if target_type != 'self' and len(targets) == 0:
                    current_tick += stride_ticks
                    continue
                
                targets.sort(key=lambda x: x['steamid'])
                
                # Create labels for alive targets
                place_labels = np.zeros(NUM_PLACES, dtype=np.float32)
                for t in targets:
                    place = t.get('place', '')
                    if place in PLACE_TO_IDX:
                        place_labels[PLACE_TO_IDX[place]] = 1.0
                
                segment_info = {
                    'start_tick': current_tick - global_min_tick,  # Relative to round start
                    'end_tick': end_tick - global_min_tick,
                    'prediction_tick': prediction_tick - global_min_tick,
                    'forecast_horizon_sec': forecast_horizon_sec,
                    'duration_seconds': segment_length_sec,
                    'map_name': map_name,
                    'pov_steamid': pov_steamid,
                    'pov_side': pov_side,
                    'target_type': target_type,
                    'targets_data': targets,
                    'place_labels': place_labels.tolist(),
                    'num_alive_targets': len(targets)
                }
                segments.append(segment_info)
                
                current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for location forecast."""
        output_rows = []
        idx = 0
        
        target_type = config.get('target_type', 'teammate')
        
        for segment in all_segments:
            row = {
                'idx': idx,
                'partition': segment['partition'],
                'pov_steamid': segment['pov_steamid'],
                'pov_side': segment['pov_side'],
                'target_type': segment['target_type'],
                'seg_duration_sec': segment['duration_seconds'],
                'forecast_horizon_sec': segment['forecast_horizon_sec'],
                'start_tick': segment['start_tick'],
                'end_tick': segment['end_tick'],
                'prediction_tick': segment['prediction_tick'],
                'match_id': segment['match_id'],
                'round_num': segment['round_num'],
                'map_name': segment['map_name']
            }
            
            # Add target data
            prefix = 'self' if target_type == 'self' else ('teammate' if target_type == 'teammate' else 'enemy')
            for i, t in enumerate(segment['targets_data']):
                row[f'{prefix}_{i}_id'] = t['steamid']
                row[f'{prefix}_{i}_future_place'] = t['place']
                row[f'{prefix}_{i}_future_X'] = t['X_norm']
                row[f'{prefix}_{i}_future_Y'] = t['Y_norm']
                row[f'{prefix}_{i}_future_Z'] = t['Z_norm']
            
            # Add multi-label targets
            for i, label in enumerate(segment['place_labels']):
                row[f'label_place_{i}'] = label
            
            output_rows.append(row)
            idx += 1
        
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'])
            df = df.reset_index(drop=True)
            df['idx'] = range(len(df))
        
        return df


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    DATA_BASE_PATH = os.getenv('DATA_BASE_PATH')
    if not DATA_BASE_PATH:
        print("DATA_BASE_PATH not set")
        exit(1)
    
    OUTPUT_DIR = os.path.join(DATA_BASE_PATH, 'labels', 'task_creator_test')
    PARTITION_CSV = os.path.join(DATA_BASE_PATH, 'match_round_partitioned.csv')
    
    # Test teammate location nowcast
    print("Testing TeammateLocationNowcastCreator...")
    creator = TeammateLocationNowcastCreator(
        DATA_BASE_PATH, OUTPUT_DIR, PARTITION_CSV,
        stride_sec=5.0  # Larger stride for testing
    )
    
    creator.process_segments({
        'output_file_name': 'test_teammate_loc_now.csv',
        'segment_length_sec': 5,
        'partition': ['test']
    })
