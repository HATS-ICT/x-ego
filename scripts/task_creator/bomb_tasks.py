"""
Bomb-related task creators.

Creates labels for:
- Bomb planted state detection
- Bomb site prediction
- Time to plant estimation
- Post-plant outcome prediction
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.task_creator.base_task_creator import TaskCreatorBase


class BombPlantedStateCreator(TaskCreatorBase):
    """
    Creates labeled segments for bomb planted state detection.
    
    Detects whether the bomb is currently planted.
    Output: Binary classification.
    """
    
    def _find_death_tick(self, player_df: pd.DataFrame) -> Optional[int]:
        """Find the tick when a player dies, or None if they survive."""
        if player_df.empty or 'health' not in player_df.columns:
            return None
        
        death_mask = player_df['health'] <= 0
        if death_mask.any():
            first_death_idx = death_mask.idxmax()
            return player_df.loc[first_death_idx, 'tick']
        return None
    
    def _get_bomb_state_at_tick(self, bomb_df: pd.DataFrame, tick: int) -> bool:
        """Determine if bomb is planted at given tick."""
        if bomb_df.empty:
            return False
        
        # Filter events before or at the tick
        past_events = bomb_df[bomb_df['tick'] <= tick].copy()
        if past_events.empty:
            return False
        
        # Get the most recent event
        past_events = past_events.sort_values('tick', ascending=False)
        latest_event = past_events.iloc[0]['event']
        
        # Bomb is planted if last event was 'plant' (not defuse/detonate)
        return latest_event == 'plant'
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for bomb planted state detection.
        
        Unlike other tasks, this only requires the POV player to be alive,
        not all 10 players. This is necessary because bombs are typically
        planted after some fighting has occurred.
        """
        segment_length_sec = config['segment_length_sec']
        
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        bomb_df = self._load_bomb(match_id, round_num)
        
        if len(player_trajectories) < 1:
            return []
        
        segments = []
        segment_ticks = segment_length_sec * self.tick_rate
        stride_ticks = int(self.stride_sec * self.tick_rate)
        
        # Get global tick range from all trajectories
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
        
        # For each player, create segments while they are alive
        for steamid, pov_df in player_trajectories.items():
            if pov_df.empty:
                continue
            
            # Find death tick for this player (or use max tick if they survive)
            death_tick = self._find_death_tick(pov_df)
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
                
                # Check bomb state at prediction tick
                bomb_planted = self._get_bomb_state_at_tick(bomb_df, middle_tick)
                
                segment_info = {
                    'start_tick': current_tick - global_min_tick,  # Relative to round start
                    'end_tick': end_tick - global_min_tick,
                    'prediction_tick': middle_tick - global_min_tick,
                    'start_seconds': current_tick / self.tick_rate,
                    'end_seconds': end_tick / self.tick_rate,
                    'prediction_seconds': middle_tick / self.tick_rate,
                    'normalized_start_seconds': (current_tick - global_min_tick) / self.tick_rate,
                    'normalized_end_seconds': (end_tick - global_min_tick) / self.tick_rate,
                    'normalized_prediction_seconds': (middle_tick - global_min_tick) / self.tick_rate,
                    'duration_seconds': segment_length_sec,
                    'map_name': map_name,
                    'pov_steamid': steamid,
                    'pov_side': pov_side,
                    'bomb_planted': int(bomb_planted)
                }
                segments.append(segment_info)
                
                current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for bomb planted state."""
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
                'label_bomb_planted': segment['bomb_planted']
            }
            output_rows.append(row)
            idx += 1
        
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'])
            df = df.reset_index(drop=True)
            df['idx'] = range(len(df))
        
        return df


class BombSitePredictionCreator(TaskCreatorBase):
    """
    Creates labeled segments for bomb site prediction task.
    
    Predicts which bomb site (A or B) the bomb will be planted at.
    Only creates segments for T-side before bomb is planted.
    Output: Binary classification (0=A, 1=B).
    """
    
    def _find_player_death_tick(self, player_df: pd.DataFrame) -> Optional[int]:
        """Find the tick when a player dies, or None if they survive."""
        if player_df.empty or 'health' not in player_df.columns:
            return None
        
        death_mask = player_df['health'] <= 0
        if death_mask.any():
            first_death_idx = death_mask.idxmax()
            return player_df.loc[first_death_idx, 'tick']
        return None
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for bomb site prediction."""
        segment_length_sec = config['segment_length_sec']
        
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        bomb_df = self._load_bomb(match_id, round_num)
        
        if len(player_trajectories) < 1:
            return []
        
        plant_events = bomb_df[bomb_df['event'] == 'plant']
        if plant_events.empty:
            return []  # No plant in this round
        
        plant_event = plant_events.iloc[0]
        plant_tick = plant_event['tick']
        bomb_site = plant_event['bombsite']
        
        if pd.isna(bomb_site) or bomb_site == '':
            return []  # No bomb site info
        
        # Determine site label (0=A, 1=B)
        if 'a' in str(bomb_site).lower():
            site_label = 0
        elif 'b' in str(bomb_site).lower():
            site_label = 1
        else:
            return []  # Unknown site
        
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
        
        # For each T-side player as POV (only T-side predicts bomb site)
        for pov_steamid, pov_df in player_trajectories.items():
            if pov_df.empty:
                continue
            
            pov_side = pov_df.iloc[0]['side']
            if pov_side != 't':
                continue  # Skip CT players
            
            death_tick = self._find_player_death_tick(pov_df)
            if death_tick is None:
                death_tick = pov_df['tick'].max()
            
            # Only use segments before plant
            max_valid_tick = min(death_tick, plant_tick - segment_ticks)
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
                    'start_tick': current_tick - global_min_tick,  # Relative to round start
                    'end_tick': end_tick - global_min_tick,
                    'prediction_tick': middle_tick - global_min_tick,
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
                    'bomb_site': site_label,
                    'bomb_site_name': bomb_site
                }
                segments.append(segment_info)
                
                current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for bomb site prediction."""
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
                'label_bomb_site': segment['bomb_site'],
                'bomb_site_name': segment['bomb_site_name']
            }
            output_rows.append(row)
            idx += 1
        
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'])
            df = df.reset_index(drop=True)
            df['idx'] = range(len(df))
        
        return df


class PostPlantOutcomeCreator(TaskCreatorBase):
    """
    Creates labeled segments for post-plant outcome prediction.
    
    After bomb is planted, predicts whether it will explode or be defused.
    Output: Binary classification (0=defused, 1=exploded).
    """
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for post-plant outcome prediction."""
        segment_length_sec = config['segment_length_sec']
        
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        bomb_df = self._load_bomb(match_id, round_num)
        rounds_df = self._load_rounds(match_id)
        
        if len(player_trajectories) != 10:
            return []
        
        # Check round outcome
        round_info = rounds_df[rounds_df['round_num'] == round_num]
        if round_info.empty:
            return []
        
        reason = round_info.iloc[0].get('reason', '')
        bomb_site = round_info.iloc[0].get('bomb_site', '')
        
        # Only include rounds where bomb was planted
        if bomb_site == 'not_planted' or pd.isna(bomb_site) or bomb_site == '':
            return []
        
        # Determine outcome (0=defused, 1=exploded)
        if reason == 'bomb_defused':
            outcome_label = 0
        elif reason == 'bomb_exploded':
            outcome_label = 1
        else:
            # Round ended by elimination after plant - use whether bomb exploded
            detonate_events = bomb_df[bomb_df['event'] == 'detonate']
            outcome_label = 1 if not detonate_events.empty else 0
        
        # Find plant tick
        plant_events = bomb_df[bomb_df['event'] == 'plant']
        if plant_events.empty:
            return []
        
        plant_tick = plant_events.iloc[0]['tick']
        
        segments = []
        segment_ticks = segment_length_sec * self.tick_rate
        
        # Get tick range after plant
        all_ticks = []
        for df in player_trajectories.values():
            if not df.empty:
                all_ticks.extend(df['tick'].tolist())
        
        if not all_ticks:
            return []
        
        max_tick = max(all_ticks)
        
        # Only use segments after plant
        min_valid_tick = plant_tick + segment_ticks // 2
        
        if max_tick - min_valid_tick < segment_ticks:
            return []
        
        stride_ticks = int(self.stride_sec * self.tick_rate)
        current_tick = min_valid_tick
        
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
            
            # Create segments for alive players
            for pov_data in all_players_data:
                if pov_data.get('health', 0) <= 0:
                    continue
                
                map_name = 'de_mirage'
                for df in player_trajectories.values():
                    if 'map_name' in df.columns and not df.empty:
                        map_name = df.iloc[0]['map_name']
                        break
                
                segment_info = {
                    'start_tick': current_tick - global_min_tick,  # Relative to round start
                    'end_tick': end_tick - global_min_tick,
                    'prediction_tick': middle_tick - global_min_tick,
                    'plant_tick': plant_tick,
                    'start_seconds': current_tick / self.tick_rate,
                    'end_seconds': end_tick / self.tick_rate,
                    'prediction_seconds': middle_tick / self.tick_rate,
                    'time_since_plant': (middle_tick - plant_tick) / self.tick_rate,
                    'duration_seconds': segment_length_sec,
                    'map_name': map_name,
                    'pov_steamid': pov_data['steamid'],
                    'pov_side': pov_data['side'],
                    'outcome': outcome_label
                }
                segments.append(segment_info)
            
            current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for post-plant outcome."""
        output_rows = []
        idx = 0
        
        for segment in all_segments:
            row = {
                'idx': idx,
                'partition': segment['partition'],
                'pov_steamid': segment['pov_steamid'],
                'pov_side': segment['pov_side'],
                'seg_duration_sec': segment['duration_seconds'],
                'time_since_plant': segment['time_since_plant'],
                'start_tick': segment['start_tick'],
                'end_tick': segment['end_tick'],
                'prediction_tick': segment['prediction_tick'],
                'plant_tick': segment['plant_tick'],
                'match_id': segment['match_id'],
                'round_num': segment['round_num'],
                'map_name': segment['map_name'],
                'label_outcome': segment['outcome']  # 0=defused, 1=exploded
            }
            output_rows.append(row)
            idx += 1
        
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'])
            df = df.reset_index(drop=True)
            df['idx'] = range(len(df))
        
        return df


class RoundWinnerCreator(TaskCreatorBase):
    """
    Creates labeled segments for round winner prediction.
    
    Predicts which team will win the round.
    Output: Binary classification (0=CT wins, 1=T wins).
    """
    
    def _find_player_death_tick(self, player_df: pd.DataFrame) -> Optional[int]:
        """Find the tick when a player dies, or None if they survive."""
        if player_df.empty or 'health' not in player_df.columns:
            return None
        
        death_mask = player_df['health'] <= 0
        if death_mask.any():
            first_death_idx = death_mask.idxmax()
            return player_df.loc[first_death_idx, 'tick']
        return None
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for round winner prediction."""
        segment_length_sec = config['segment_length_sec']
        
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        rounds_df = self._load_rounds(match_id)
        
        if len(player_trajectories) < 1:
            return []
        
        # Get round winner
        round_info = rounds_df[rounds_df['round_num'] == round_num]
        if round_info.empty:
            return []
        
        winner = round_info.iloc[0].get('winner', '')
        if winner == 'ct':
            winner_label = 0
        elif winner == 't':
            winner_label = 1
        else:
            return []  # Unknown winner
        
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
                
                segment_info = {
                    'start_tick': current_tick - global_min_tick,  # Relative to round start
                    'end_tick': end_tick - global_min_tick,
                    'prediction_tick': middle_tick - global_min_tick,
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
                    'round_winner': winner_label,
                    'round_winner_name': winner
                }
                segments.append(segment_info)
                
                current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for round winner prediction."""
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
                'label_round_winner': segment['round_winner'],
                'round_winner_name': segment['round_winner_name']
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
    
    # Test bomb planted state
    print("Testing BombPlantedStateCreator...")
    creator = BombPlantedStateCreator(
        DATA_BASE_PATH, OUTPUT_DIR, PARTITION_CSV,
        stride_sec=5.0
    )
    
    creator.process_segments({
        'output_file_name': 'test_bomb_planted.csv',
        'segment_length_sec': 5,
        'partition': ['test']
    })
