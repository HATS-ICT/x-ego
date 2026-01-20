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


class TeamAlignmentRelevance(str, Enum):
    """Expected relevance to team POV alignment."""
    HIGH = "high"      # Likely to improve with team contrastive
    MEDIUM = "medium"  # May improve moderately
    LOW = "low"        # Unlikely to improve
    NEGATIVE = "negative"  # May actually degrade


class ExpectedEffect(str, Enum):
    """Expected effect of team contrastive learning."""
    IMPROVE = "improve"
    NEUTRAL = "neutral"
    DEGRADE = "degrade"
    UNKNOWN = "unknown"


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
    relevance_to_team_alignment: TeamAlignmentRelevance
    expected_effect: ExpectedEffect
    ml_form: MLForm
    num_classes: Optional[int]  # None for regression
    output_dim: int
    primary_data_source: DataSource
    label_field: str
    temporal_type: TemporalType
    horizon_sec: float
    feasibility_notes: str
    
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
    
    @property
    def is_high_relevance(self) -> bool:
        """Check if task has high relevance to team alignment."""
        return self.relevance_to_team_alignment == TeamAlignmentRelevance.HIGH


# Map places on de_mirage to indices
MIRAGE_PLACES = [
    "TSpawn", "CTSpawn", "TopofMid", "Catwalk", "Ladder",
    "BombsiteA", "BombsiteB", "Palace", "PalaceAlley", "PalaceInterior",
    "Ramp", "TRamp", "Apartments", "BackAlley", "House",
    "Jungle", "Connector", "Shop", "SnipersNest", "Scaffolding",
    "Kitchen", "SideAlley", "Market", "Underpass", "Window"
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
ROUND_OUTCOMES = ["t_killed", "ct_killed", "bomb_exploded", "bomb_defused"]
OUTCOME_TO_IDX = {o: i for i, o in enumerate(ROUND_OUTCOMES)}
NUM_OUTCOMES = len(ROUND_OUTCOMES)

# Common weapon categories
WEAPONS = [
    "glock", "usp_silencer", "hkp2000", "p250", "deagle", "fiveseven", "tec9",
    "ak47", "m4a1", "m4a1_silencer", "awp", "sg556", "aug", "famas", "galil",
    "mac10", "mp9", "ump45", "p90", "nova"
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
            relevance_to_team_alignment=TeamAlignmentRelevance(row['relevance_to_team_alignment']),
            expected_effect=ExpectedEffect(row['expected_effect']),
            ml_form=MLForm(row['ml_form']),
            num_classes=int(row['num_classes']) if pd.notna(row['num_classes']) else None,
            output_dim=int(row['output_dim']),
            primary_data_source=DataSource(primary_source),
            label_field=row['label_field'],
            temporal_type=TemporalType(row['temporal_type']),
            horizon_sec=float(row['horizon_sec']),
            feasibility_notes=row['feasibility_notes']
        )
        tasks.append(task)
    
    return tasks


def get_tasks_by_category(tasks: List[TaskDefinition], category: TaskCategory) -> List[TaskDefinition]:
    """Filter tasks by category."""
    return [t for t in tasks if t.category == category]


def get_tasks_by_relevance(tasks: List[TaskDefinition], 
                           relevance: TeamAlignmentRelevance) -> List[TaskDefinition]:
    """Filter tasks by team alignment relevance."""
    return [t for t in tasks if t.relevance_to_team_alignment == relevance]


def get_high_relevance_tasks(tasks: List[TaskDefinition]) -> List[TaskDefinition]:
    """Get tasks with high relevance to team alignment."""
    return get_tasks_by_relevance(tasks, TeamAlignmentRelevance.HIGH)


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
    print(f"High relevance tasks: {len(get_high_relevance_tasks(tasks))}")
    print(f"Classification tasks: {len(get_classification_tasks(tasks))}")
    print(f"Regression tasks: {len(get_regression_tasks(tasks))}")
