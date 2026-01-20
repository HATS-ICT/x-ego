"""
Task creator helper module.

Exports all task creator classes for use in create_all_labels.py.
"""

from .base_task_creator import TaskCreatorBase
from .location_tasks import (
    TeammateLocationNowcastCreator,
    EnemyLocationNowcastCreator,
    LocationForecastCreator,
)
from .coordination_tasks import (
    TeamSpreadCreator,
    TeamCentroidCreator,
    AliveCountCreator,
    NearestTeammateDistanceCreator,
    TeamMovementDirectionCreator,
)
from .coordination_tasks_addon import (
    TeammateMovementDirectionCreator,
    TeammateSpeedCreator,
    TeammateWeaponCreator,
)
from .combat_tasks import (
    ImminentKillCreator,
    ImminentDeathSelfCreator,
    ImminentDamageCreator,
    InCombatCreator,
    ImminentKillSelfCreator,
)
from .bomb_tasks import (
    BombPlantedStateCreator,
    BombSitePredictionCreator,
    PostPlantOutcomeCreator,
    RoundWinnerCreator,
    RoundOutcomeReasonCreator,
)
from .bomb_tasks_addon import (
    WillPlantPredictionCreator,
)
from .spatial_tasks import (
    POVMovementDirectionCreator,
    POVSpeedCreator,
)
from .action_tasks import (
    WeaponInUseCreator,
)

__all__ = [
    # Base
    "TaskCreatorBase",
    # Location
    "TeammateLocationNowcastCreator",
    "EnemyLocationNowcastCreator",
    "LocationForecastCreator",
    # Coordination
    "TeamSpreadCreator",
    "TeamCentroidCreator",
    "AliveCountCreator",
    "NearestTeammateDistanceCreator",
    "TeamMovementDirectionCreator",
    "TeammateMovementDirectionCreator",
    "TeammateSpeedCreator",
    "TeammateWeaponCreator",
    # Combat
    "ImminentKillCreator",
    "ImminentDeathSelfCreator",
    "ImminentDamageCreator",
    "InCombatCreator",
    "ImminentKillSelfCreator",
    # Bomb/Round
    "BombPlantedStateCreator",
    "BombSitePredictionCreator",
    "WillPlantPredictionCreator",
    "PostPlantOutcomeCreator",
    "RoundWinnerCreator",
    "RoundOutcomeReasonCreator",
    # Spatial
    "POVMovementDirectionCreator",
    "POVSpeedCreator",
    # Action
    "WeaponInUseCreator",
]
