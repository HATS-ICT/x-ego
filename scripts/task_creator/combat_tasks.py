"""
Combat/engagement task creators.

Creates labels for:
- Imminent kill prediction
- Imminent death (self) prediction
- Imminent damage prediction
- In-combat detection (POV and team)
- Headshot prediction
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.task_creator.base_task_creator import TaskCreatorBase


class ImminentKillCreator(TaskCreatorBase):
    """
    Creates labeled segments for imminent kill prediction task.
    
    Predicts whether any kill will happen in the next N seconds.
    Output: Binary classification.
    """
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for imminent kill prediction."""
        segment_length_sec = config['segment_length_sec']
        horizon_sec = config.get('horizon_sec', 3.0)
        
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        kills_df = self._load_kills(match_id, round_num)
        
        if len(player_trajectories) < 1:
            return []
        
        segments = []
        segment_ticks = segment_length_sec * self.tick_rate
        horizon_ticks = int(horizon_sec * self.tick_rate)
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
                
                # Check for kills in forecast window
                forecast_start = end_tick
                forecast_end = end_tick + horizon_ticks
                
                has_kill = False
                if not kills_df.empty:
                    has_kill = self._check_event_in_window(kills_df, forecast_start, forecast_end)
                
                segment_info = {
                    'start_tick': current_tick - global_min_tick,  # Relative to round start
                    'end_tick': end_tick - global_min_tick,
                    'prediction_tick': middle_tick - global_min_tick,
                    'horizon_sec': horizon_sec,
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
                    'has_kill': int(has_kill)
                }
                segments.append(segment_info)
                
                current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for imminent kill prediction."""
        output_rows = []
        idx = 0
        
        for segment in all_segments:
            row = {
                'idx': idx,
                'partition': segment['partition'],
                'pov_steamid': segment['pov_steamid'],
                'pov_side': segment['pov_side'],
                'seg_duration_sec': segment['duration_seconds'],
                'horizon_sec': segment['horizon_sec'],
                'start_tick': segment['start_tick'],
                'end_tick': segment['end_tick'],
                'prediction_tick': segment['prediction_tick'],
                'match_id': segment['match_id'],
                'round_num': segment['round_num'],
                'map_name': segment['map_name'],
                'label_has_kill': segment['has_kill']
            }
            output_rows.append(row)
            idx += 1
        
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'])
            df = df.reset_index(drop=True)
            df['idx'] = range(len(df))
        
        return df


class ImminentDeathSelfCreator(TaskCreatorBase):
    """
    Creates labeled segments for imminent self death prediction task.
    
    Predicts whether the POV player will die in the next N seconds.
    Output: Binary classification.
    """
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for imminent self death prediction."""
        segment_length_sec = config['segment_length_sec']
        horizon_sec = config.get('horizon_sec', 3.0)
        
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        kills_df = self._load_kills(match_id, round_num)
        
        if len(player_trajectories) < 1:
            return []
        
        segments = []
        segment_ticks = segment_length_sec * self.tick_rate
        horizon_ticks = int(horizon_sec * self.tick_rate)
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
                
                # Check if POV player dies in forecast window
                forecast_start = end_tick
                forecast_end = end_tick + horizon_ticks
                
                pov_dies = False
                if not kills_df.empty:
                    window_kills = self._get_events_in_window(kills_df, forecast_start, forecast_end)
                    if not window_kills.empty:
                        pov_dies = (window_kills['victim_steamid'].astype(str) == str(pov_steamid)).any()
                
                segment_info = {
                    'start_tick': current_tick - global_min_tick,  # Relative to round start
                    'end_tick': end_tick - global_min_tick,
                    'prediction_tick': middle_tick - global_min_tick,
                    'horizon_sec': horizon_sec,
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
                    'pov_dies': int(pov_dies)
                }
                segments.append(segment_info)
                
                current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for imminent self death prediction."""
        output_rows = []
        idx = 0
        
        for segment in all_segments:
            row = {
                'idx': idx,
                'partition': segment['partition'],
                'pov_steamid': segment['pov_steamid'],
                'pov_side': segment['pov_side'],
                'seg_duration_sec': segment['duration_seconds'],
                'horizon_sec': segment['horizon_sec'],
                'start_tick': segment['start_tick'],
                'end_tick': segment['end_tick'],
                'prediction_tick': segment['prediction_tick'],
                'match_id': segment['match_id'],
                'round_num': segment['round_num'],
                'map_name': segment['map_name'],
                'label_pov_dies': segment['pov_dies']
            }
            output_rows.append(row)
            idx += 1
        
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'])
            df = df.reset_index(drop=True)
            df['idx'] = range(len(df))
        
        return df


class ImminentDamageCreator(TaskCreatorBase):
    """
    Creates labeled segments for imminent damage prediction task.
    
    Predicts whether any damage will occur in the next N seconds.
    Output: Binary classification.
    """
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for imminent damage prediction."""
        segment_length_sec = config['segment_length_sec']
        horizon_sec = config.get('horizon_sec', 3.0)
        
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        damages_df = self._load_damages(match_id, round_num)
        
        if len(player_trajectories) < 1:
            return []
        
        segments = []
        segment_ticks = segment_length_sec * self.tick_rate
        horizon_ticks = int(horizon_sec * self.tick_rate)
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
                
                # Check for damage in forecast window
                forecast_start = end_tick
                forecast_end = end_tick + horizon_ticks
                
                has_damage = False
                if not damages_df.empty:
                    has_damage = self._check_event_in_window(damages_df, forecast_start, forecast_end)
                
                segment_info = {
                    'start_tick': current_tick - global_min_tick,  # Relative to round start
                    'end_tick': end_tick - global_min_tick,
                    'prediction_tick': middle_tick - global_min_tick,
                    'horizon_sec': horizon_sec,
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
                    'has_damage': int(has_damage)
                }
                segments.append(segment_info)
                
                current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for imminent damage prediction."""
        output_rows = []
        idx = 0
        
        for segment in all_segments:
            row = {
                'idx': idx,
                'partition': segment['partition'],
                'pov_steamid': segment['pov_steamid'],
                'pov_side': segment['pov_side'],
                'seg_duration_sec': segment['duration_seconds'],
                'horizon_sec': segment['horizon_sec'],
                'start_tick': segment['start_tick'],
                'end_tick': segment['end_tick'],
                'prediction_tick': segment['prediction_tick'],
                'match_id': segment['match_id'],
                'round_num': segment['round_num'],
                'map_name': segment['map_name'],
                'label_has_damage': segment['has_damage']
            }
            output_rows.append(row)
            idx += 1
        
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'])
            df = df.reset_index(drop=True)
            df['idx'] = range(len(df))
        
        return df


class InCombatCreator(TaskCreatorBase):
    """
    Creates labeled segments for in-combat detection task.
    
    Detects whether POV player or team is currently in combat
    (has damage event in recent window).
    Output: Binary classification.
    """
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for in-combat detection."""
        segment_length_sec = config['segment_length_sec']
        combat_window_sec = config.get('combat_window_sec', 2.0)
        target_type = config.get('target_type', 'pov')  # 'pov' or 'team'
        
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        damages_df = self._load_damages(match_id, round_num)
        
        if len(player_trajectories) < 1:
            return []
        
        segments = []
        segment_ticks = segment_length_sec * self.tick_rate
        combat_window_ticks = int(combat_window_sec * self.tick_rate)
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
                
                # Get recent damage events
                combat_start = middle_tick - combat_window_ticks
                combat_end = middle_tick + combat_window_ticks
                
                recent_damages = pd.DataFrame()
                if not damages_df.empty:
                    recent_damages = self._get_events_in_window(damages_df, combat_start, combat_end)
                
                if target_type == 'pov':
                    # Check if POV player is involved in combat
                    in_combat = False
                    if not recent_damages.empty:
                        in_combat = (
                            (recent_damages['attacker_steamid'].astype(str) == str(pov_steamid)).any() or
                            (recent_damages['victim_steamid'].astype(str) == str(pov_steamid)).any()
                        )
                else:  # team
                    # Get alive teammates at this tick
                    team_steamids = [str(pov_steamid)]
                    for steamid, df in player_trajectories.items():
                        if steamid == pov_steamid:
                            continue
                        player_data = self._extract_player_data_at_tick(df, middle_tick)
                        if player_data and player_data['side'] == pov_side and player_data.get('health', 0) > 0:
                            team_steamids.append(str(steamid))
                    
                    in_combat = False
                    if not recent_damages.empty:
                        for sid in team_steamids:
                            if ((recent_damages['attacker_steamid'].astype(str) == sid).any() or
                                (recent_damages['victim_steamid'].astype(str) == sid).any()):
                                in_combat = True
                                break
                
                segment_info = {
                    'start_tick': current_tick - global_min_tick,  # Relative to round start
                    'end_tick': end_tick - global_min_tick,
                    'prediction_tick': middle_tick - global_min_tick,
                    'combat_window_sec': combat_window_sec,
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
                    'target_type': target_type,
                    'in_combat': int(in_combat)
                }
                segments.append(segment_info)
                
                current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for in-combat detection."""
        output_rows = []
        idx = 0
        
        for segment in all_segments:
            row = {
                'idx': idx,
                'partition': segment['partition'],
                'pov_steamid': segment['pov_steamid'],
                'pov_side': segment['pov_side'],
                'target_type': segment['target_type'],
                'seg_duration_sec': segment['duration_seconds'],
                'combat_window_sec': segment['combat_window_sec'],
                'start_tick': segment['start_tick'],
                'end_tick': segment['end_tick'],
                'prediction_tick': segment['prediction_tick'],
                'match_id': segment['match_id'],
                'round_num': segment['round_num'],
                'map_name': segment['map_name'],
                'label_in_combat': segment['in_combat']
            }
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
    
    # Test imminent kill
    print("Testing ImminentKillCreator...")
    creator = ImminentKillCreator(
        DATA_BASE_PATH, OUTPUT_DIR, PARTITION_CSV,
        stride_sec=5.0
    )
    
    creator.process_segments({
        'output_file_name': 'test_imminent_kill.csv',
        'segment_length_sec': 5,
        'horizon_sec': 3.0,
        'partition': ['test']
    })
