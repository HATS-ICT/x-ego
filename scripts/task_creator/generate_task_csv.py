#!/usr/bin/env python3
"""
Script to generate task_definitions.csv with all task definitions.

This script creates a comprehensive CSV file defining all downstream
prediction tasks for linear probing experiments evaluating team POV
contrastive learning.
"""

import pandas as pd
from pathlib import Path


def generate_task_definitions() -> pd.DataFrame:
    """
    Generate all task definitions as a DataFrame.
    
    Returns:
        DataFrame with task definitions
    """
    tasks = []
    
    # ========== Category: Location ==========
    
    # Teammate Location Nowcast
    tasks.append({
        'task_id': 'teammate_loc_now',
        'task_name': 'Teammate Location Nowcast',
        'category': 'location',
        'description': 'Predict 4 teammates current locations (places)',
        'ml_form': 'multi_label_cls',
        'num_classes': 25,
        'output_dim': 25,
        'primary_data_source': 'trajectory',
        'label_field': 'place',
        'temporal_type': 'nowcast',
        'horizon_sec': 0,
        'feasibility_notes': 'Uses place field from trajectory; 25 unique places on de_mirage',
        'implemented': 'yes'
    })
    
    # Teammate Coordinate Nowcast
    tasks.append({
        'task_id': 'teammate_coord_now',
        'task_name': 'Teammate Coordinate Nowcast',
        'category': 'location',
        'description': 'Predict 4 teammates normalized XYZ coordinates',
        'ml_form': 'regression',
        'num_classes': None,
        'output_dim': 12,
        'primary_data_source': 'trajectory',
        'label_field': 'X_norm;Y_norm;Z_norm',
        'temporal_type': 'nowcast',
        'horizon_sec': 0,
        'feasibility_notes': '4 teammates x 3 coordinates = 12 output dims',
        'implemented': 'yes'
    })
    
    # Enemy Location Nowcast
    tasks.append({
        'task_id': 'enemy_loc_now',
        'task_name': 'Enemy Location Nowcast',
        'category': 'location',
        'description': 'Predict 5 enemies current locations (places)',
        'ml_form': 'multi_label_cls',
        'num_classes': 25,
        'output_dim': 25,
        'primary_data_source': 'trajectory',
        'label_field': 'place',
        'temporal_type': 'nowcast',
        'horizon_sec': 0,
        'feasibility_notes': 'Enemies not in contrastive set; baseline task',
        'implemented': 'yes'
    })
    
    # Enemy Coordinate Nowcast
    tasks.append({
        'task_id': 'enemy_coord_now',
        'task_name': 'Enemy Coordinate Nowcast',
        'category': 'location',
        'description': 'Predict 5 enemies normalized XYZ coordinates',
        'ml_form': 'regression',
        'num_classes': None,
        'output_dim': 15,
        'primary_data_source': 'trajectory',
        'label_field': 'X_norm;Y_norm;Z_norm',
        'temporal_type': 'nowcast',
        'horizon_sec': 0,
        'feasibility_notes': '5 enemies x 3 coordinates = 15 output dims',
        'implemented': 'no'
    })
    
    # Self Location Forecast 5s
    tasks.append({
        'task_id': 'self_loc_forecast_5s',
        'task_name': 'Self Location Forecast 5s',
        'category': 'location',
        'description': 'Predict own location 5 seconds ahead',
        'ml_form': 'multi_cls',
        'num_classes': 25,
        'output_dim': 25,
        'primary_data_source': 'trajectory',
        'label_field': 'place',
        'temporal_type': 'forecast',
        'horizon_sec': 5,
        'feasibility_notes': 'Own future intent; medium team relevance',
        'implemented': 'yes'
    })
    
    # Self Location Forecast 10s
    tasks.append({
        'task_id': 'self_loc_forecast_10s',
        'task_name': 'Self Location Forecast 10s',
        'category': 'location',
        'description': 'Predict own location 10 seconds ahead',
        'ml_form': 'multi_cls',
        'num_classes': 25,
        'output_dim': 25,
        'primary_data_source': 'trajectory',
        'label_field': 'place',
        'temporal_type': 'forecast',
        'horizon_sec': 10,
        'feasibility_notes': 'Longer horizon self prediction',
        'implemented': 'yes'
    })
    
    # Teammate Location Forecast 5s
    tasks.append({
        'task_id': 'teammate_loc_forecast_5s',
        'task_name': 'Teammate Location Forecast 5s',
        'category': 'location',
        'description': 'Predict 4 teammates locations 5 seconds ahead',
        'ml_form': 'multi_label_cls',
        'num_classes': 25,
        'output_dim': 25,
        'primary_data_source': 'trajectory',
        'label_field': 'place',
        'temporal_type': 'forecast',
        'horizon_sec': 5,
        'feasibility_notes': 'Team coordination prediction',
        'implemented': 'yes'
    })
    
    # Teammate Location Forecast 10s
    tasks.append({
        'task_id': 'teammate_loc_forecast_10s',
        'task_name': 'Teammate Location Forecast 10s',
        'category': 'location',
        'description': 'Predict 4 teammates locations 10 seconds ahead',
        'ml_form': 'multi_label_cls',
        'num_classes': 25,
        'output_dim': 25,
        'primary_data_source': 'trajectory',
        'label_field': 'place',
        'temporal_type': 'forecast',
        'horizon_sec': 10,
        'feasibility_notes': 'Longer horizon team prediction',
        'implemented': 'yes'
    })
    
    # Enemy Location Forecast 5s
    tasks.append({
        'task_id': 'enemy_loc_forecast_5s',
        'task_name': 'Enemy Location Forecast 5s',
        'category': 'location',
        'description': 'Predict 5 enemies locations 5 seconds ahead',
        'ml_form': 'multi_label_cls',
        'num_classes': 25,
        'output_dim': 25,
        'primary_data_source': 'trajectory',
        'label_field': 'place',
        'temporal_type': 'forecast',
        'horizon_sec': 5,
        'feasibility_notes': 'Enemy prediction baseline',
        'implemented': 'yes'
    })
    
    # Enemy Location Forecast 10s
    tasks.append({
        'task_id': 'enemy_loc_forecast_10s',
        'task_name': 'Enemy Location Forecast 10s',
        'category': 'location',
        'description': 'Predict 5 enemies locations 10 seconds ahead',
        'ml_form': 'multi_label_cls',
        'num_classes': 25,
        'output_dim': 25,
        'primary_data_source': 'trajectory',
        'label_field': 'place',
        'temporal_type': 'forecast',
        'horizon_sec': 10,
        'feasibility_notes': 'Longer horizon enemy prediction',
        'implemented': 'yes'
    })
    
    # ========== Category: Coordination ==========
    
    # Team Spread
    tasks.append({
        'task_id': 'team_spread',
        'task_name': 'Team Spatial Spread',
        'category': 'coordination',
        'description': 'Estimate team spatial spread (std of positions)',
        'ml_form': 'regression',
        'num_classes': None,
        'output_dim': 1,
        'primary_data_source': 'trajectory',
        'label_field': 'X_norm;Y_norm',
        'temporal_type': 'nowcast',
        'horizon_sec': 0,
        'feasibility_notes': 'Std dev of teammate positions',
        'implemented': 'yes'
    })
    
    # Team Centroid
    tasks.append({
        'task_id': 'team_centroid',
        'task_name': 'Team Centroid Location',
        'category': 'coordination',
        'description': 'Predict team centroid normalized coordinates',
        'ml_form': 'regression',
        'num_classes': None,
        'output_dim': 3,
        'primary_data_source': 'trajectory',
        'label_field': 'X_norm;Y_norm;Z_norm',
        'temporal_type': 'nowcast',
        'horizon_sec': 0,
        'feasibility_notes': 'Mean of 5 team member positions',
        'implemented': 'yes'
    })
    
    # Teammate Alive Count
    tasks.append({
        'task_id': 'teammate_alive_count',
        'task_name': 'Teammate Alive Count',
        'category': 'coordination',
        'description': 'Count number of alive teammates',
        'ml_form': 'multi_cls',
        'num_classes': 5,
        'output_dim': 5,
        'primary_data_source': 'trajectory',
        'label_field': 'health',
        'temporal_type': 'nowcast',
        'horizon_sec': 0,
        'feasibility_notes': 'Classes 0-4 teammates alive',
        'implemented': 'yes'
    })
    
    # Enemy Alive Count
    tasks.append({
        'task_id': 'enemy_alive_count',
        'task_name': 'Enemy Alive Count',
        'category': 'coordination',
        'description': 'Count number of alive enemies',
        'ml_form': 'multi_cls',
        'num_classes': 6,
        'output_dim': 6,
        'primary_data_source': 'trajectory',
        'label_field': 'health',
        'temporal_type': 'nowcast',
        'horizon_sec': 0,
        'feasibility_notes': 'Classes 0-5 enemies alive',
        'implemented': 'yes'
    })
    
    # Teammate Proximity
    tasks.append({
        'task_id': 'teammate_proximity',
        'task_name': 'Nearest Teammate Distance',
        'category': 'coordination',
        'description': 'Predict distance to nearest teammate',
        'ml_form': 'regression',
        'num_classes': None,
        'output_dim': 1,
        'primary_data_source': 'trajectory',
        'label_field': 'X_norm;Y_norm;Z_norm',
        'temporal_type': 'nowcast',
        'horizon_sec': 0,
        'feasibility_notes': 'Euclidean distance in normalized coords',
        'implemented': 'yes'
    })
    
    # Team Movement Direction
    tasks.append({
        'task_id': 'team_movement_dir',
        'task_name': 'Team Movement Direction',
        'category': 'coordination',
        'description': 'Predict aggregate team movement direction',
        'ml_form': 'multi_cls',
        'num_classes': 9,
        'output_dim': 9,
        'primary_data_source': 'trajectory',
        'label_field': 'X_norm;Y_norm',
        'temporal_type': 'nowcast',
        'horizon_sec': 0,
        'feasibility_notes': '8 directions + stationary; requires delta computation',
        'implemented': 'yes'
    })
    
    # ========== Category: Combat ==========
    
    # Imminent Kill 3s
    tasks.append({
        'task_id': 'imminent_kill_3s',
        'task_name': 'Imminent Kill 3s',
        'category': 'combat',
        'description': 'Will any kill happen in next 3 seconds',
        'ml_form': 'binary_cls',
        'num_classes': 2,
        'output_dim': 1,
        'primary_data_source': 'kills',
        'label_field': 'tick',
        'temporal_type': 'forecast',
        'horizon_sec': 3,
        'feasibility_notes': 'Binary prediction from kills.csv timing',
        'implemented': 'yes'
    })
    
    # Imminent Kill 5s
    tasks.append({
        'task_id': 'imminent_kill_5s',
        'task_name': 'Imminent Kill 5s',
        'category': 'combat',
        'description': 'Will any kill happen in next 5 seconds',
        'ml_form': 'binary_cls',
        'num_classes': 2,
        'output_dim': 1,
        'primary_data_source': 'kills',
        'label_field': 'tick',
        'temporal_type': 'forecast',
        'horizon_sec': 5,
        'feasibility_notes': 'Wider window for kill prediction',
        'implemented': 'yes'
    })
    
    # Imminent Death Self 3s
    tasks.append({
        'task_id': 'imminent_death_self_3s',
        'task_name': 'Self Death 3s',
        'category': 'combat',
        'description': 'Will POV player die in next 3 seconds',
        'ml_form': 'binary_cls',
        'num_classes': 2,
        'output_dim': 1,
        'primary_data_source': 'kills',
        'label_field': 'victim_steamid',
        'temporal_type': 'forecast',
        'horizon_sec': 3,
        'feasibility_notes': 'Individual survival prediction',
        'implemented': 'yes'
    })
    
    # Imminent Death Self 5s
    tasks.append({
        'task_id': 'imminent_death_self_5s',
        'task_name': 'Self Death 5s',
        'category': 'combat',
        'description': 'Will POV player die in next 5 seconds',
        'ml_form': 'binary_cls',
        'num_classes': 2,
        'output_dim': 1,
        'primary_data_source': 'kills',
        'label_field': 'victim_steamid',
        'temporal_type': 'forecast',
        'horizon_sec': 5,
        'feasibility_notes': 'Individual survival wider window',
        'implemented': 'yes'
    })
    
    # Imminent Damage 3s
    tasks.append({
        'task_id': 'imminent_damage_3s',
        'task_name': 'Imminent Damage 3s',
        'category': 'combat',
        'description': 'Will any damage occur in next 3 seconds',
        'ml_form': 'binary_cls',
        'num_classes': 2,
        'output_dim': 1,
        'primary_data_source': 'damages',
        'label_field': 'tick',
        'temporal_type': 'forecast',
        'horizon_sec': 3,
        'feasibility_notes': 'Combat engagement prediction',
        'implemented': 'yes'
    })
    
    # Team In Combat
    tasks.append({
        'task_id': 'team_in_combat',
        'task_name': 'Team In Combat',
        'category': 'combat',
        'description': 'Is any teammate currently in combat',
        'ml_form': 'binary_cls',
        'num_classes': 2,
        'output_dim': 1,
        'primary_data_source': 'damages',
        'label_field': 'attacker_steamid;victim_steamid',
        'temporal_type': 'nowcast',
        'horizon_sec': 0,
        'feasibility_notes': 'Recent damage involving team within 2s window',
        'implemented': 'yes'
    })
    
    # POV In Combat
    tasks.append({
        'task_id': 'pov_in_combat',
        'task_name': 'POV In Combat',
        'category': 'combat',
        'description': 'Is POV player currently in combat',
        'ml_form': 'binary_cls',
        'num_classes': 2,
        'output_dim': 1,
        'primary_data_source': 'damages',
        'label_field': 'attacker_steamid;victim_steamid',
        'temporal_type': 'nowcast',
        'horizon_sec': 0,
        'feasibility_notes': 'Recent damage involving POV player',
        'implemented': 'yes'
    })
    
    # ========== Category: Bomb ==========
    
    # Bomb Planted
    tasks.append({
        'task_id': 'bomb_planted',
        'task_name': 'Bomb Planted State',
        'category': 'bomb',
        'description': 'Is bomb currently planted',
        'ml_form': 'binary_cls',
        'num_classes': 2,
        'output_dim': 1,
        'primary_data_source': 'bomb',
        'label_field': 'event',
        'temporal_type': 'nowcast',
        'horizon_sec': 0,
        'feasibility_notes': 'Game state from bomb plant/defuse/explode events',
        'implemented': 'yes'
    })
    
    # Bomb Site Prediction
    tasks.append({
        'task_id': 'bomb_site_prediction',
        'task_name': 'Bomb Site Prediction',
        'category': 'bomb',
        'description': 'Predict which site bomb will be planted (A vs B)',
        'ml_form': 'binary_cls',
        'num_classes': 2,
        'output_dim': 1,
        'primary_data_source': 'bomb;rounds',
        'label_field': 'bomb_site',
        'temporal_type': 'forecast',
        'horizon_sec': 0,
        'feasibility_notes': 'Strategic prediction; only for T-side pre-plant',
        'implemented': 'yes'
    })
    
    # Post Plant Outcome
    tasks.append({
        'task_id': 'post_plant_outcome',
        'task_name': 'Post Plant Outcome',
        'category': 'bomb',
        'description': 'After plant predict explode vs defuse',
        'ml_form': 'binary_cls',
        'num_classes': 2,
        'output_dim': 1,
        'primary_data_source': 'bomb',
        'label_field': 'event',
        'temporal_type': 'forecast',
        'horizon_sec': 0,
        'feasibility_notes': 'Only valid for post-plant segments',
        'implemented': 'yes'
    })
    
    # ========== Category: Round ==========
    
    # Round Winner
    tasks.append({
        'task_id': 'round_winner',
        'task_name': 'Round Winner Prediction',
        'category': 'round',
        'description': 'Predict which team wins the round',
        'ml_form': 'binary_cls',
        'num_classes': 2,
        'output_dim': 1,
        'primary_data_source': 'rounds',
        'label_field': 'winner',
        'temporal_type': 'forecast',
        'horizon_sec': 0,
        'feasibility_notes': 'Win probability from current state',
        'implemented': 'yes'
    })
    
    # Round Outcome Reason
    tasks.append({
        'task_id': 'round_outcome_reason',
        'task_name': 'Round Outcome Reason',
        'category': 'round',
        'description': 'Predict how round will end',
        'ml_form': 'multi_cls',
        'num_classes': 4,
        'output_dim': 4,
        'primary_data_source': 'rounds',
        'label_field': 'reason',
        'temporal_type': 'forecast',
        'horizon_sec': 0,
        'feasibility_notes': 't_killed;ct_killed;bomb_exploded;bomb_defused',
        'implemented': 'no'
    })
    
    # ========== Category: Spatial ==========
    
    # POV Place Classification
    tasks.append({
        'task_id': 'pov_place_cls',
        'task_name': 'POV Place Classification',
        'category': 'spatial',
        'description': 'Classify POV player current location',
        'ml_form': 'multi_cls',
        'num_classes': 25,
        'output_dim': 25,
        'primary_data_source': 'trajectory',
        'label_field': 'place',
        'temporal_type': 'nowcast',
        'horizon_sec': 0,
        'feasibility_notes': 'Self-localization baseline',
        'implemented': 'no'
    })
    
    # POV Movement Direction
    tasks.append({
        'task_id': 'pov_movement_dir',
        'task_name': 'POV Movement Direction',
        'category': 'spatial',
        'description': 'Predict POV player movement direction',
        'ml_form': 'multi_cls',
        'num_classes': 9,
        'output_dim': 9,
        'primary_data_source': 'trajectory',
        'label_field': 'X_norm;Y_norm',
        'temporal_type': 'nowcast',
        'horizon_sec': 0,
        'feasibility_notes': '8 directions + stationary',
        'implemented': 'no'
    })
    
    # POV Speed
    tasks.append({
        'task_id': 'pov_speed',
        'task_name': 'POV Speed Estimation',
        'category': 'spatial',
        'description': 'Estimate POV player movement speed',
        'ml_form': 'regression',
        'num_classes': None,
        'output_dim': 1,
        'primary_data_source': 'trajectory',
        'label_field': 'X_norm;Y_norm;Z_norm',
        'temporal_type': 'nowcast',
        'horizon_sec': 0,
        'feasibility_notes': 'Magnitude of velocity vector',
        'implemented': 'no'
    })
    
    # ========== Category: Action ==========
    
    # Team Executing
    tasks.append({
        'task_id': 'team_executing',
        'task_name': 'Team Executing',
        'category': 'action',
        'description': 'Is team executing a site take',
        'ml_form': 'binary_cls',
        'num_classes': 2,
        'output_dim': 1,
        'primary_data_source': 'trajectory',
        'label_field': 'X_norm;Y_norm',
        'temporal_type': 'nowcast',
        'horizon_sec': 0,
        'feasibility_notes': 'Inferred from coordinated movement toward site',
        'implemented': 'no'
    })
    
    # Team Rotating
    tasks.append({
        'task_id': 'team_rotating',
        'task_name': 'Team Rotating',
        'category': 'action',
        'description': 'Is team rotating between sites',
        'ml_form': 'binary_cls',
        'num_classes': 2,
        'output_dim': 1,
        'primary_data_source': 'trajectory',
        'label_field': 'X_norm;Y_norm',
        'temporal_type': 'nowcast',
        'horizon_sec': 0,
        'feasibility_notes': 'Inferred from lateral movement pattern',
        'implemented': 'no'
    })
    
    return pd.DataFrame(tasks)


def main():
    """Generate task_definitions.csv."""
    output_path = Path(__file__).parent / "task_definitions.csv"
    
    print("Generating task definitions...")
    df = generate_task_definitions()
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"Saved {len(df)} task definitions to {output_path}")
    
    # Print summary
    print("\nSummary by category:")
    for category in df['category'].unique():
        count = len(df[df['category'] == category])
        print(f"  {category}: {count} tasks")
    
    print("\nSummary by implementation status:")
    for impl in df['implemented'].unique():
        count = len(df[df['implemented'] == impl])
        print(f"  {impl}: {count} tasks")
    
    print("\nSummary by ML form:")
    for form in df['ml_form'].unique():
        count = len(df[df['ml_form'] == form])
        print(f"  {form}: {count} tasks")


if __name__ == "__main__":
    main()
