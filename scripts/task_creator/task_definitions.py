"""
Task definition constants and enums for linear probing experiments.

This module defines the categories, ML forms, and relevance levels
for downstream prediction tasks used to evaluate team POV contrastive learning.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import pandas as pd


class TaskCategory(str, Enum):
    """Categories of prediction tasks."""
    LOCATION = "location"
    COORDINATION = "coordination"
    COMBAT = "combat"
    BOMB = "bomb"
    ROUND = "round"
    SPATIAL = "spatial"
    ACTION = "action"


class MLForm(str, Enum):
    """Machine learning task formulations."""
    BINARY_CLS = "binary_cls"
    MULTI_CLS = "multi_cls"
    MULTI_LABEL_CLS = "multi_label_cls"
    REGRESSION = "regression"


class TemporalType(str, Enum):
    """Temporal type of prediction."""
    NOWCAST = "nowcast"   # Predict current state
    FORECAST = "forecast"  # Predict future state


class DataSource(str, Enum):
    """Primary data sources for tasks."""
    TRAJECTORY = "trajectory"
    BOMB = "bomb"
    DAMAGES = "damages"
    KILLS = "kills"
    ROUNDS = "rounds"
    SHOTS = "shots"
    METADATA = "metadata"


@dataclass
class TaskDefinition:
    """Complete definition of a prediction task."""
    task_id: str
    task_name: str
    category: TaskCategory
    description: str
    ml_form: MLForm
    num_classes: Optional[int]  # None for regression
    output_dim: int
    primary_data_source: DataSource
    label_field: str
    temporal_type: TemporalType
    horizon_sec: float
    feasibility_notes: str
    implemented: bool
    
    @property
    def is_classification(self) -> bool:
        """Check if task is a classification task."""
        return self.ml_form in [MLForm.BINARY_CLS, MLForm.MULTI_CLS, MLForm.MULTI_LABEL_CLS]
    
    @property
    def is_regression(self) -> bool:
        """Check if task is a regression task."""
        return self.ml_form == MLForm.REGRESSION
    
    @property
    def is_forecast(self) -> bool:
        """Check if task is a forecast (future prediction) task."""
        return self.temporal_type == TemporalType.FORECAST


# Map places on de_mirage to indices (actual places found in dataset)
MIRAGE_PLACES = [
    "Apartments", "BackAlley", "BombsiteA", "BombsiteB", "CTSpawn",
    "Catwalk", "Connector", "House", "Jungle", "Ladder",
    "Middle", "PalaceAlley", "PalaceInterior", "Scaffolding", "Shop",
    "SideAlley", "SnipersNest", "Stairs", "TRamp", "TSpawn",
    "TopofMid", "Truck", "Underpass"
]

PLACE_TO_IDX = {place: idx for idx, place in enumerate(MIRAGE_PLACES)}
IDX_TO_PLACE = {idx: place for idx, place in enumerate(MIRAGE_PLACES)}
NUM_PLACES = len(MIRAGE_PLACES)

# Movement direction mapping (8 directions + stationary)
MOVEMENT_DIRECTIONS = [
    "N", "NE", "E", "SE", "S", "SW", "W", "NW", "STATIONARY"
]
DIRECTION_TO_IDX = {d: i for i, d in enumerate(MOVEMENT_DIRECTIONS)}
NUM_DIRECTIONS = len(MOVEMENT_DIRECTIONS)

# Round outcome reasons
ROUND_OUTCOMES = ["t_killed", "ct_killed", "bomb_exploded", "bomb_defused", "time_ran_out"]
OUTCOME_TO_IDX = {o: i for i, o in enumerate(ROUND_OUTCOMES)}
NUM_OUTCOMES = len(ROUND_OUTCOMES)

# Weapons (actual weapons found in dataset, sorted alphabetically)
WEAPONS = [
    "ak47", "aug", "awp", "bayonet", "bizon", "cz75a", "deagle", "elite",
    "famas", "fiveseven", "galilar", "glock", "hegrenade", "hkp2000",
    "incgrenade", "inferno", "knife", "knife_butterfly", "knife_canis",
    "knife_cord", "knife_falchion", "knife_flip", "knife_gut",
    "knife_gypsy_jackknife", "knife_karambit", "knife_kukri", "knife_m9_bayonet",
    "knife_outdoor", "knife_push", "knife_skeleton", "knife_stiletto",
    "knife_survival_bowie", "knife_t", "knife_tactical", "knife_ursus",
    "knife_widowmaker", "m4a1", "m4a1_silencer", "mac10", "mag7", "molotov",
    "mp5sd", "mp7", "mp9", "nova", "p250", "p90", "revolver", "scar20",
    "sg556", "ssg08", "taser", "tec9", "ump45", "usp_silencer", "xm1014"
]
WEAPON_TO_IDX = {w: i for i, w in enumerate(WEAPONS)}
NUM_WEAPONS = len(WEAPONS)


def load_task_definitions(csv_path: Optional[Path] = None) -> List[TaskDefinition]:
    """
    Load task definitions from CSV file.
    
    Args:
        csv_path: Path to task_definitions.csv. If None, uses default location.
        
    Returns:
        List of TaskDefinition objects
    """
    if csv_path is None:
        csv_path = Path(__file__).parent / "task_definitions.csv"
    
    df = pd.read_csv(csv_path)
    tasks = []
    
    for _, row in df.iterrows():
        # Parse data source (may contain multiple sources separated by ;)
        primary_source = row['primary_data_source'].split(';')[0]
        
        task = TaskDefinition(
            task_id=row['task_id'],
            task_name=row['task_name'],
            category=TaskCategory(row['category']),
            description=row['description'],
            ml_form=MLForm(row['ml_form']),
            num_classes=int(row['num_classes']) if pd.notna(row['num_classes']) else None,
            output_dim=int(row['output_dim']),
            primary_data_source=DataSource(primary_source),
            label_field=row['label_field'],
            temporal_type=TemporalType(row['temporal_type']),
            horizon_sec=float(row['horizon_sec']),
            feasibility_notes=row['feasibility_notes'],
            implemented=row['implemented'] == 'yes'
        )
        tasks.append(task)
    
    return tasks


def get_tasks_by_category(tasks: List[TaskDefinition], category: TaskCategory) -> List[TaskDefinition]:
    """Filter tasks by category."""
    return [t for t in tasks if t.category == category]


def get_implemented_tasks(tasks: List[TaskDefinition]) -> List[TaskDefinition]:
    """Get tasks that are implemented."""
    return [t for t in tasks if t.implemented]


def get_classification_tasks(tasks: List[TaskDefinition]) -> List[TaskDefinition]:
    """Get classification tasks."""
    return [t for t in tasks if t.is_classification]


def get_regression_tasks(tasks: List[TaskDefinition]) -> List[TaskDefinition]:
    """Get regression tasks."""
    return [t for t in tasks if t.is_regression]


if __name__ == "__main__":
    # Test loading and print summary
    tasks = load_task_definitions()
    print(f"Loaded {len(tasks)} task definitions")
    print()
    
    for category in TaskCategory:
        cat_tasks = get_tasks_by_category(tasks, category)
        print(f"{category.value}: {len(cat_tasks)} tasks")
    
    print()
    print(f"Implemented tasks: {len(get_implemented_tasks(tasks))}")
    print(f"Classification tasks: {len(get_classification_tasks(tasks))}")
    print(f"Regression tasks: {len(get_regression_tasks(tasks))}")
