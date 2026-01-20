"""
Task Creator Module for Linear Probing Experiments.

This module provides tools to create diverse downstream prediction tasks
for evaluating team POV contrastive learning effectiveness.

Task Categories:
- location: Teammate/enemy location prediction (nowcast and forecast)
- coordination: Team spread, centroid, alive count, proximity
- combat: Kill/damage prediction, in-combat detection
- bomb: Bomb state, site prediction, post-plant outcome
- round: Round winner prediction
- spatial: POV location, movement direction, speed
- action: Team executing, team rotating

Usage:
    from scripts.task_creator import (
        load_task_definitions,
        get_implemented_tasks,
        TeammateLocationNowcastCreator,
        TeamSpreadCreator,
        ImminentKillCreator,
        BombPlantedStateCreator,
    )
    
    # Load all task definitions
    tasks = load_task_definitions()
    
    # Get tasks that are implemented
    impl_tasks = get_implemented_tasks(tasks)
    
    # Create labels for a specific task
    creator = TeammateLocationNowcastCreator(
        data_dir, output_dir, partition_csv_path
    )
    creator.process_segments({
        'output_file_name': 'teammate_loc_now.csv',
        'segment_length_sec': 5,
        'partition': ['train', 'val', 'test']
    })
"""

# Task definitions and utilities
from scripts.task_creator.task_definitions import (
    TaskCategory,
    MLForm,
    TemporalType,
    DataSource,
    TaskDefinition,
    MIRAGE_PLACES,
    PLACE_TO_IDX,
    IDX_TO_PLACE,
    NUM_PLACES,
    MOVEMENT_DIRECTIONS,
    DIRECTION_TO_IDX,
    NUM_DIRECTIONS,
    ROUND_OUTCOMES,
    OUTCOME_TO_IDX,
    NUM_OUTCOMES,
    WEAPONS,
    WEAPON_TO_IDX,
    NUM_WEAPONS,
    load_task_definitions,
    get_tasks_by_category,
    get_implemented_tasks,
    get_classification_tasks,
    get_regression_tasks,
)

# Base class
from scripts.task_creator.base_task_creator import TaskCreatorBase

# Location task creators
from scripts.task_creator.location_tasks import (
    TeammateLocationNowcastCreator,
    EnemyLocationNowcastCreator,
    TeammateCoordinateNowcastCreator,
    LocationForecastCreator,
)

# Coordination task creators
from scripts.task_creator.coordination_tasks import (
    TeamSpreadCreator,
    TeamCentroidCreator,
    AliveCountCreator,
    NearestTeammateDistanceCreator,
    TeamMovementDirectionCreator,
)

# Combat task creators
from scripts.task_creator.combat_tasks import (
    ImminentKillCreator,
    ImminentDeathSelfCreator,
    ImminentDamageCreator,
    InCombatCreator,
)

# Bomb task creators
from scripts.task_creator.bomb_tasks import (
    BombPlantedStateCreator,
    BombSitePredictionCreator,
    PostPlantOutcomeCreator,
    RoundWinnerCreator,
)

# Note: create_all_labels and analyze_label_stats are standalone scripts
# Run them directly:
#   python -m scripts.task_creator.create_all_labels
#   python -m scripts.task_creator.analyze_label_stats


__all__ = [
    # Enums and types
    'TaskCategory',
    'MLForm',
    'TemporalType',
    'DataSource',
    'TaskDefinition',
    
    # Constants
    'MIRAGE_PLACES',
    'PLACE_TO_IDX',
    'IDX_TO_PLACE',
    'NUM_PLACES',
    'MOVEMENT_DIRECTIONS',
    'DIRECTION_TO_IDX',
    'NUM_DIRECTIONS',
    'ROUND_OUTCOMES',
    'OUTCOME_TO_IDX',
    'NUM_OUTCOMES',
    'WEAPONS',
    'WEAPON_TO_IDX',
    'NUM_WEAPONS',
    
    # Utility functions
    'load_task_definitions',
    'get_tasks_by_category',
    'get_implemented_tasks',
    'get_classification_tasks',
    'get_regression_tasks',
    
    # Base class
    'TaskCreatorBase',
    
    # Location creators
    'TeammateLocationNowcastCreator',
    'EnemyLocationNowcastCreator',
    'TeammateCoordinateNowcastCreator',
    'LocationForecastCreator',
    
    # Coordination creators
    'TeamSpreadCreator',
    'TeamCentroidCreator',
    'AliveCountCreator',
    'NearestTeammateDistanceCreator',
    'TeamMovementDirectionCreator',
    
    # Combat creators
    'ImminentKillCreator',
    'ImminentDeathSelfCreator',
    'ImminentDamageCreator',
    'InCombatCreator',
    
    # Bomb creators
    'BombPlantedStateCreator',
    'BombSitePredictionCreator',
    'PostPlantOutcomeCreator',
    'RoundWinnerCreator',
]
