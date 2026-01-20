"""
Team coordination task creators.

Creates labels for:
- Team spread (spatial dispersion)
- Team centroid location
- Teammate/enemy alive count
- Nearest teammate distance
- Team movement direction
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.task_creator.base_task_creator import TaskCreatorBase


class TeamSpreadCreator(TaskCreatorBase):
    """
    Creates labeled segments for team spatial spread task.
    
    Predicts the spatial spread (std of positions) of alive team members.
    Output: Regression with 1 output (spread value).
    """
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for team spread."""
        segment_length_sec = config['segment_length_sec']
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        
        if len(player_trajectories) < 1:
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
                
                # Get alive team members (including POV)
                team_members = [pov_data]
                for steamid, df in player_trajectories.items():
                    if steamid == pov_steamid:
                        continue
                    player_data = self._extract_player_data_at_tick(df, middle_tick)
                    if player_data and player_data['side'] == pov_side and player_data.get('health', 0) > 0:
                        team_members.append(player_data)
                
                # Need at least 2 alive team members to compute spread
                if len(team_members) < 2:
                    current_tick += stride_ticks
                    continue
                
                # Compute team spread
                spread = self._compute_team_spread(team_members)
                
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
                    'team_spread': spread,
                    'num_alive_team_members': len(team_members)
                }
                segments.append(segment_info)
                
                current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for team spread."""
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
                'label_team_spread': segment['team_spread']
            }
            output_rows.append(row)
            idx += 1
        
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'])
            df = df.reset_index(drop=True)
            df['idx'] = range(len(df))
        
        return df


class TeamCentroidCreator(TaskCreatorBase):
    """
    Creates labeled segments for team centroid task.
    
    Predicts the centroid location of alive team members.
    Output: Regression with 3 outputs (X, Y, Z normalized coords).
    """
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for team centroid."""
        segment_length_sec = config['segment_length_sec']
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        
        if len(player_trajectories) < 1:
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
                
                # Get alive team members (including POV)
                team_members = [pov_data]
                for steamid, df in player_trajectories.items():
                    if steamid == pov_steamid:
                        continue
                    player_data = self._extract_player_data_at_tick(df, middle_tick)
                    if player_data and player_data['side'] == pov_side and player_data.get('health', 0) > 0:
                        team_members.append(player_data)
                
                # Need at least 1 alive team member
                if len(team_members) < 1:
                    current_tick += stride_ticks
                    continue
                
                # Compute team centroid
                centroid_x, centroid_y, centroid_z = self._compute_team_centroid(team_members)
                
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
                    'centroid_x': centroid_x,
                    'centroid_y': centroid_y,
                    'centroid_z': centroid_z,
                    'num_alive_team_members': len(team_members)
                }
                segments.append(segment_info)
                
                current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for team centroid."""
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
                'label_centroid_x': segment['centroid_x'],
                'label_centroid_y': segment['centroid_y'],
                'label_centroid_z': segment['centroid_z']
            }
            output_rows.append(row)
            idx += 1
        
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'])
            df = df.reset_index(drop=True)
            df['idx'] = range(len(df))
        
        return df


class AliveCountCreator(TaskCreatorBase):
    """
    Creates labeled segments for alive count tasks.
    
    Predicts number of alive teammates or enemies.
    Output: Multi-class classification (0-4 for teammates, 0-5 for enemies).
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
                    # Count alive teammates (excluding POV)
                    targets = [p for p in all_players_data 
                              if p['side'] == pov_side and p['steamid'] != pov_steamid]
                    alive_count = sum(1 for p in targets if p.get('health', 0) > 0)
                else:  # enemy
                    enemy_side = self._get_opposite_side(pov_side)
                    targets = [p for p in all_players_data if p['side'] == enemy_side]
                    alive_count = sum(1 for p in targets if p.get('health', 0) > 0)
                
                map_name = 'de_mirage'
                for df in player_trajectories.values():
                    if 'map_name' in df.columns and not df.empty:
                        map_name = df.iloc[0]['map_name']
                        break
                
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
                    'target_type': target_type,
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
                'target_type': segment['target_type'],
                'seg_duration_sec': segment['duration_seconds'],
                'start_tick': segment['start_tick'],
                'end_tick': segment['end_tick'],
                'prediction_tick': segment['prediction_tick'],
                'match_id': segment['match_id'],
                'round_num': segment['round_num'],
                'map_name': segment['map_name'],
                'label_alive_count': segment['alive_count']
            }
            output_rows.append(row)
            idx += 1
        
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'])
            df = df.reset_index(drop=True)
            df['idx'] = range(len(df))
        
        return df


class NearestTeammateDistanceCreator(TaskCreatorBase):
    """
    Creates labeled segments for nearest teammate distance task.
    
    Predicts the distance to the nearest alive teammate.
    Output: Regression with 1 output (distance in normalized coords).
    """
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for nearest teammate distance."""
        segment_length_sec = config['segment_length_sec']
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        
        if len(player_trajectories) < 1:
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
                
                # Get alive teammates (excluding POV)
                teammates = []
                for steamid, df in player_trajectories.items():
                    if steamid == pov_steamid:
                        continue
                    player_data = self._extract_player_data_at_tick(df, middle_tick)
                    if player_data and player_data['side'] == pov_side and player_data.get('health', 0) > 0:
                        teammates.append(player_data)
                
                # Need at least one alive teammate to compute distance
                if len(teammates) == 0:
                    current_tick += stride_ticks
                    continue
                
                # Find nearest alive teammate
                min_dist = float('inf')
                for tm in teammates:
                    dist = self._compute_distance(pov_data, tm)
                    if dist < min_dist:
                        min_dist = dist
                
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
                    'nearest_distance': min_dist,
                    'num_alive_teammates': len(teammates)
                }
                segments.append(segment_info)
                
                current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for nearest teammate distance."""
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
                'label_nearest_distance': segment['nearest_distance']
            }
            output_rows.append(row)
            idx += 1
        
        df = pd.DataFrame(output_rows)
        if len(df) > 0:
            df = df.sort_values(['partition', 'match_id', 'round_num'])
            df = df.reset_index(drop=True)
            df['idx'] = range(len(df))
        
        return df


class TeamMovementDirectionCreator(TaskCreatorBase):
    """
    Creates labeled segments for team movement direction task.
    
    Predicts the aggregate movement direction of alive team members.
    Output: Multi-class classification (9 classes: 8 directions + stationary).
    """
    
    def _extract_segments_from_round(self, match_id: str, round_num: int,
                                     config: Dict[str, Any]) -> List[Dict]:
        """Extract segments for team movement direction."""
        segment_length_sec = config['segment_length_sec']
        player_trajectories = self._load_player_trajectories(match_id, round_num)
        
        if len(player_trajectories) < 1:
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
                
                # Get alive team members' current and previous positions
                team_curr = [pov_curr]
                team_prev = {pov_steamid: pov_prev}
                
                for steamid, df in player_trajectories.items():
                    if steamid == pov_steamid:
                        continue
                    curr_data = self._extract_player_data_at_tick(df, middle_tick)
                    prev_data = self._extract_player_data_at_tick(df, prev_tick)
                    if curr_data and prev_data and curr_data['side'] == pov_side and curr_data.get('health', 0) > 0:
                        team_curr.append(curr_data)
                        team_prev[steamid] = prev_data
                
                # Need at least 1 alive team member
                if len(team_curr) < 1:
                    current_tick += stride_ticks
                    continue
                
                # Compute aggregate team movement of alive members
                total_dx = 0.0
                total_dy = 0.0
                for curr in team_curr:
                    sid = curr['steamid']
                    if sid in team_prev:
                        prev = team_prev[sid]
                        total_dx += curr.get('X_norm', 0) - prev.get('X_norm', 0)
                        total_dy += curr.get('Y_norm', 0) - prev.get('Y_norm', 0)
                
                # Compute aggregate direction
                agg_prev = {'X_norm': 0, 'Y_norm': 0}
                agg_curr = {'X_norm': total_dx, 'Y_norm': total_dy}
                direction_idx = self._compute_movement_direction(agg_prev, agg_curr)
                
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
                    'direction_idx': direction_idx,
                    'num_alive_team_members': len(team_curr)
                }
                segments.append(segment_info)
                
                current_tick += stride_ticks
        
        return segments
    
    def _create_output_csv(self, all_segments: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
        """Create output CSV for team movement direction."""
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
                'label_direction': segment['direction_idx']
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
    
    # Test team spread
    print("Testing TeamSpreadCreator...")
    creator = TeamSpreadCreator(
        DATA_BASE_PATH, OUTPUT_DIR, PARTITION_CSV,
        stride_sec=5.0
    )
    
    creator.process_segments({
        'output_file_name': 'test_team_spread.csv',
        'segment_length_sec': 5,
        'partition': ['test']
    })
