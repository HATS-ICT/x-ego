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
from .location_tasks_addon import (
    SelfLocationNowcastCreator,
)
from .coordination_tasks import (
    AliveCountCreator,
)
from .coordination_tasks_addon import (
    TeammateMovementDirectionCreator,
    TeammateSpeedCreator,
)
from .combat_tasks import (
    ImminentKillCreator,
    ImminentDeathSelfCreator,
    InCombatCreator,
    SelfInCombatCreator,
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
    SelfMovementDirectionCreator,
    SelfSpeedCreator,
)

__all__ = [
    # Base
    "TaskCreatorBase",
    # Location
    "SelfLocationNowcastCreator",
    "TeammateLocationNowcastCreator",
    "EnemyLocationNowcastCreator",
    "LocationForecastCreator",
    # Coordination
    "AliveCountCreator",
    "TeammateMovementDirectionCreator",
    "TeammateSpeedCreator",
    # Combat
    "ImminentKillCreator",
    "ImminentDeathSelfCreator",
    "InCombatCreator",
    "SelfInCombatCreator",
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
    "SelfMovementDirectionCreator",
    "SelfSpeedCreator",
]
