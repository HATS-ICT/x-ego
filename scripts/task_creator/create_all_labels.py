"""
Master script to create all task labels.

This script instantiates and runs all task creators to generate
label CSV files for linear probing experiments.

Usage:
    python -m scripts.task_creator.create_all_labels [--output_dir OUTPUT_DIR] [--stride STRIDE]
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import argparse
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))

# Import all task creators
from scripts.task_creator.task_creator_helper import (
    SelfLocationNowcastCreator,
    TeammateLocationNowcastCreator,
    EnemyLocationNowcastCreator,
    LocationForecastCreator,
    TeamSpreadCreator,
    TeamCentroidCreator,
    AliveCountCreator,
    NearestTeammateDistanceCreator,
    TeamMovementDirectionCreator,
    TeammateMovementDirectionCreator,
    TeammateSpeedCreator,
    TeammateWeaponCreator,
    ImminentKillCreator,
    ImminentDeathSelfCreator,
    ImminentKillSelfCreator,
    InCombatCreator,
    SelfInCombatCreator,
    BombPlantedStateCreator,
    BombSitePredictionCreator,
    WillPlantPredictionCreator,
    PostPlantOutcomeCreator,
    RoundWinnerCreator,
    RoundOutcomeReasonCreator,
    SelfMovementDirectionCreator,
    SelfSpeedCreator,
    SelfWeaponCreator,
)


# Task configurations: (task_id, creator_class, config_overrides, output_filename)
TASK_CONFIGS: List[Tuple[str, type, Dict[str, Any], str]] = [
    # ============= LOCATION TASKS =============
    # Self location nowcast
    ("self_loc_now", SelfLocationNowcastCreator, {}, "self_loc_now.csv"),
    
    # Teammate location nowcast
    ("teammate_loc_now", TeammateLocationNowcastCreator, {}, "teammate_loc_now.csv"),
    
    # Enemy location nowcast
    ("enemy_loc_now", EnemyLocationNowcastCreator, {}, "enemy_loc_now.csv"),
    
    # Self location forecast (5s, 10s, 20s)
    ("self_loc_forecast_5s", LocationForecastCreator, 
     {"forecast_horizon_sec": 5.0, "target_type": "self"}, "self_loc_forecast_5s.csv"),
    ("self_loc_forecast_10s", LocationForecastCreator, 
     {"forecast_horizon_sec": 10.0, "target_type": "self"}, "self_loc_forecast_10s.csv"),
    ("self_loc_forecast_20s", LocationForecastCreator, 
     {"forecast_horizon_sec": 20.0, "target_type": "self"}, "self_loc_forecast_20s.csv"),
    
    # Teammate location forecast (5s, 10s, 20s)
    ("teammate_loc_forecast_5s", LocationForecastCreator, 
     {"forecast_horizon_sec": 5.0, "target_type": "teammate"}, "teammate_loc_forecast_5s.csv"),
    ("teammate_loc_forecast_10s", LocationForecastCreator, 
     {"forecast_horizon_sec": 10.0, "target_type": "teammate"}, "teammate_loc_forecast_10s.csv"),
    ("teammate_loc_forecast_20s", LocationForecastCreator, 
     {"forecast_horizon_sec": 20.0, "target_type": "teammate"}, "teammate_loc_forecast_20s.csv"),
    
    # Enemy location forecast (5s, 10s, 20s)
    ("enemy_loc_forecast_5s", LocationForecastCreator, 
     {"forecast_horizon_sec": 5.0, "target_type": "enemy"}, "enemy_loc_forecast_5s.csv"),
    ("enemy_loc_forecast_10s", LocationForecastCreator, 
     {"forecast_horizon_sec": 10.0, "target_type": "enemy"}, "enemy_loc_forecast_10s.csv"),
    ("enemy_loc_forecast_20s", LocationForecastCreator, 
     {"forecast_horizon_sec": 20.0, "target_type": "enemy"}, "enemy_loc_forecast_20s.csv"),
    
    # ============= COORDINATION TASKS =============
    # Team spread
    ("team_spread", TeamSpreadCreator, {}, "team_spread.csv"),
    
    # Team centroid
    ("team_centroid", TeamCentroidCreator, {}, "team_centroid.csv"),
    
    # Alive counts
    ("teammate_alive_count", AliveCountCreator, 
     {"count_type": "teammate"}, "teammate_alive_count.csv"),
    ("enemy_alive_count", AliveCountCreator, 
     {"count_type": "enemy"}, "enemy_alive_count.csv"),
    
    # Nearest teammate distance
    ("teammate_proximity", NearestTeammateDistanceCreator, {}, "teammate_proximity.csv"),
    
    # Team movement direction
    ("team_movement_dir", TeamMovementDirectionCreator, {}, "team_movement_dir.csv"),
    
    # Teammate movement direction
    ("teammate_movement_dir", TeammateMovementDirectionCreator, {}, "teammate_movement_dir.csv"),
    
    # Teammate speed
    ("teammate_speed", TeammateSpeedCreator, {}, "teammate_speed.csv"),
    
    # Teammate weapon
    ("teammate_weapon", TeammateWeaponCreator, {}, "teammate_weapon.csv"),
    
    # ============= COMBAT TASKS =============
    # Imminent kill (5s, 10s, 20s)
    ("imminent_kill_5s", ImminentKillCreator, 
     {"horizon_sec": 5.0}, "imminent_kill_5s.csv"),
    ("imminent_kill_10s", ImminentKillCreator, 
     {"horizon_sec": 10.0}, "imminent_kill_10s.csv"),
    ("imminent_kill_20s", ImminentKillCreator, 
     {"horizon_sec": 20.0}, "imminent_kill_20s.csv"),
    
    # Imminent death self (5s, 10s, 20s)
    ("imminent_death_self_5s", ImminentDeathSelfCreator, 
     {"horizon_sec": 5.0}, "imminent_death_self_5s.csv"),
    ("imminent_death_self_10s", ImminentDeathSelfCreator, 
     {"horizon_sec": 10.0}, "imminent_death_self_10s.csv"),
    ("imminent_death_self_20s", ImminentDeathSelfCreator, 
     {"horizon_sec": 20.0}, "imminent_death_self_20s.csv"),
    
    # POV player gets a kill (5s, 10s, 20s)
    ("imminent_kill_self_5s", ImminentKillSelfCreator, 
     {"horizon_sec": 5.0}, "imminent_kill_self_5s.csv"),
    ("imminent_kill_self_10s", ImminentKillSelfCreator, 
     {"horizon_sec": 10.0}, "imminent_kill_self_10s.csv"),
    ("imminent_kill_self_20s", ImminentKillSelfCreator, 
     {"horizon_sec": 20.0}, "imminent_kill_self_20s.csv"),
    
    # In combat detection
    ("team_in_combat", InCombatCreator, 
     {"combat_type": "team"}, "team_in_combat.csv"),
    ("self_in_combat", SelfInCombatCreator, 
     {"combat_type": "pov"}, "self_in_combat.csv"),
    
    # ============= BOMB TASKS =============
    # Bomb planted state
    ("bomb_planted", BombPlantedStateCreator, {}, "bomb_planted.csv"),
    
    # Bomb site prediction
    ("bomb_site_prediction", BombSitePredictionCreator, {}, "bomb_site_prediction.csv"),
    
    # Will plant prediction
    ("will_plant", WillPlantPredictionCreator, {}, "will_plant.csv"),
    
    # Post plant outcome
    ("post_plant_outcome", PostPlantOutcomeCreator, {}, "post_plant_outcome.csv"),
    
    # Round winner
    ("round_winner", RoundWinnerCreator, {}, "round_winner.csv"),
    
    # Round outcome reason (multi-class)
    ("round_outcome_reason", RoundOutcomeReasonCreator, {}, "round_outcome_reason.csv"),
    
    # ============= SPATIAL TASKS =============
    # Self movement direction
    ("self_movement_dir", SelfMovementDirectionCreator, {}, "self_movement_dir.csv"),
    
    # Self speed estimation
    ("self_speed", SelfSpeedCreator, {}, "self_speed.csv"),
    
    # ============= ACTION TASKS =============
    # Self weapon in use
    ("self_weapon", SelfWeaponCreator, {}, "self_weapon.csv"),
]

# Tasks that are defined but not yet implemented
NOT_IMPLEMENTED_TASKS = [
    "headshot_next_kill",    # Needs new creator
    "bomb_carrier_dist",     # Needs new creator
    "pov_place_cls",         # POV self-location classification
    "area_control_mid",      # Area control detection
    "imminent_shot_self_2s", # Shot prediction
    "team_executing",        # Site take detection
    "team_rotating",         # Rotation detection
]


def create_task_labels(
    data_dir: str,
    output_dir: str,
    partition_csv: str,
    task_ids: List[str] = None,
    stride_sec: float = 1.0,
    segment_length_sec: float = 5.0,
    partitions: List[str] = None,
    max_samples: int = None,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Create labels for specified tasks.
    
    Args:
        data_dir: Base data directory
        output_dir: Output directory for CSV files
        partition_csv: Path to partition CSV
        task_ids: List of task IDs to create (None = all)
        stride_sec: Stride between segments in seconds
        segment_length_sec: Segment length in seconds
        partitions: List of partitions to include (default: all)
        max_samples: Maximum samples per task (None = no limit)
        verbose: Print progress
        
    Returns:
        Dict mapping task_id to output file path
    """
    if partitions is None:
        partitions = ["train", "val", "test"]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Filter tasks if specific ones requested
    tasks_to_run = TASK_CONFIGS
    if task_ids:
        tasks_to_run = [t for t in TASK_CONFIGS if t[0] in task_ids]
        
        # Warn about unknown task IDs
        known_ids = {t[0] for t in TASK_CONFIGS}
        for tid in task_ids:
            if tid not in known_ids:
                if tid in NOT_IMPLEMENTED_TASKS:
                    print(f"Warning: Task '{tid}' is not yet implemented")
                else:
                    print(f"Warning: Unknown task ID '{tid}'")
    
    total = len(tasks_to_run)
    
    for i, (task_id, creator_class, config_overrides, filename) in enumerate(tasks_to_run, 1):
        if verbose:
            print(f"\n[{i}/{total}] Creating labels for: {task_id}")
            print(f"  Creator: {creator_class.__name__}")
            print(f"  Output: {filename}")
        
        try:
            # Instantiate creator
            creator = creator_class(
                data_dir, output_dir, partition_csv,
                stride_sec=stride_sec
            )
            
            # Build config
            config = {
                "segment_length_sec": segment_length_sec,
                "output_file_name": filename,
                "partition": partitions,
                "max_samples": max_samples,
                **config_overrides
            }
            
            # Process segments with early stopping
            df = creator.process_segments(config)
            
            # Save
            output_file = output_path / filename
            df.to_csv(output_file, index=False)
            
            results[task_id] = str(output_file)
            
            if verbose:
                print(f"  Generated {len(df)} segments")
                print(f"  Saved to: {output_file}")
                
        except Exception as e:
            print(f"  ERROR: Failed to create {task_id}: {e}")
            import traceback
            traceback.print_exc()
            results[task_id] = None
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Create all task labels for linear probing")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: DATA_BASE_PATH/labels/all_tasks)")
    parser.add_argument("--stride", type=float, default=5.0,
                        help="Stride between segments in seconds (default: 1.0)")
    parser.add_argument("--segment_length", type=float, default=5.0,
                        help="Segment length in seconds (default: 5.0)")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Specific task IDs to create (default: all)")
    parser.add_argument("--partitions", type=str, nargs="+", default=None,
                        help="Partitions to include (default: train val test)")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Maximum samples per task (default: 5000, use 0 for no limit)")
    parser.add_argument("--list", action="store_true",
                        help="List all available tasks and exit")
    args = parser.parse_args()
    
    # List mode
    if args.list:
        print("Implemented tasks:")
        print("-" * 60)
        for task_id, creator_class, config, filename in TASK_CONFIGS:
            print(f"  {task_id:30s} -> {filename}")
        print()
        print("Not yet implemented tasks:")
        print("-" * 60)
        for task_id in NOT_IMPLEMENTED_TASKS:
            print(f"  {task_id}")
        return
    
    # Load environment
    load_dotenv()
    
    DATA_BASE_PATH = os.getenv('DATA_BASE_PATH')
    if not DATA_BASE_PATH:
        print("ERROR: DATA_BASE_PATH environment variable not set")
        sys.exit(1)
    
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(DATA_BASE_PATH, 'labels', 'all_tasks')
    
    partition_csv = os.path.join(DATA_BASE_PATH, 'match_round_partitioned.csv')
    
    print("=" * 60)
    print("Task Label Creation")
    print("=" * 60)
    print(f"Data directory: {DATA_BASE_PATH}")
    print(f"Output directory: {output_dir}")
    print(f"Partition CSV: {partition_csv}")
    print(f"Stride: {args.stride}s")
    print(f"Segment length: {args.segment_length}s")
    
    if args.tasks:
        print(f"Tasks: {args.tasks}")
    else:
        print(f"Tasks: ALL ({len(TASK_CONFIGS)} tasks)")
    
    partitions = args.partitions if args.partitions else ["train", "val", "test"]
    print(f"Partitions: {partitions}")
    
    max_samples = args.max_samples if args.max_samples > 0 else None
    if max_samples:
        print(f"Max samples per task: {max_samples}")
    else:
        print("Max samples per task: unlimited")
    
    print("=" * 60)
    
    results = create_task_labels(
        data_dir=DATA_BASE_PATH,
        output_dir=output_dir,
        partition_csv=partition_csv,
        task_ids=args.tasks,
        stride_sec=args.stride,
        segment_length_sec=args.segment_length,
        partitions=partitions,
        max_samples=max_samples,
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    successful = sum(1 for v in results.values() if v is not None)
    failed = sum(1 for v in results.values() if v is None)
    
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed tasks:")
        for task_id, path in results.items():
            if path is None:
                print(f"  - {task_id}")


if __name__ == "__main__":
    main()
