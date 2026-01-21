"""
Configuration utilities for X-EGO project.
Handles loading, validation, and manipulation of training configurations.
"""

from pathlib import Path
import pandas as pd
from omegaconf import OmegaConf


def load_cfg(cfg_path):
    """Load configuration from YAML file using OmegaConf"""
    cfg = OmegaConf.load(cfg_path)
    return cfg


def _task_id_to_labels_filename(task_id: str) -> str:
    """
    Convert task_id to the corresponding labels CSV filename.
    
    CSV files are named exactly as their task_id: {task_id}.csv
    """
    return f"all_tasks/{task_id}.csv"


def load_task_config(task_id: str, data_path: Path) -> dict:
    """
    Load task configuration from task_definitions.csv based on task_id.
    
    Args:
        task_id: Task identifier (e.g., 'self_location_0s', 'enemy_location_5s')
        data_path: Path to data directory containing labels/task_definitions.csv
        
    Returns:
        Dictionary with task configuration (ml_form, num_classes, output_dim, label_column, labels_filename)
    """
    task_def_path = data_path / "labels" / "task_definitions.csv"
    
    if not task_def_path.exists():
        raise FileNotFoundError(f"Task definitions file not found: {task_def_path}")
    
    df = pd.read_csv(task_def_path)
    task_row = df[df['task_id'] == task_id]
    
    if len(task_row) == 0:
        available_tasks = df['task_id'].tolist()
        raise ValueError(f"Task '{task_id}' not found in task_definitions.csv. Available: {available_tasks}")
    
    row = task_row.iloc[0]
    
    # Determine label column based on task type
    label_column = _get_label_column_for_task(task_id, row['ml_form'], int(row['output_dim']))
    
    # Determine labels filename from task_id
    labels_filename = _task_id_to_labels_filename(task_id)
    
    config = {
        'ml_form': row['ml_form'],
        'num_classes': int(row['num_classes']) if pd.notna(row['num_classes']) else None,
        'output_dim': int(row['output_dim']),
        'label_column': label_column,
        'labels_filename': labels_filename,
    }
    
    return config


def _get_label_column_for_task(task_id: str, ml_form: str, output_dim: int) -> str:
    """
    Determine the label column name for a task.
    
    The label column naming convention varies by task:
    - Most tasks: 'label'
    - Multi-label/multi-output tasks: 'label_0;label_1;...' (semicolon-separated)
    - Some tasks have specific column names based on the task
    
    Args:
        task_id: Task identifier
        ml_form: ML formulation (binary_cls, multi_cls, multi_label_cls, regression)
        output_dim: Output dimension of the task
        
    Returns:
        Label column name(s), semicolon-separated for multi-column tasks
    """
    # Task-specific label column mappings (for tasks that don't use 'label')
    TASK_LABEL_COLUMNS = {
        # Combat tasks with specific column names
        'self_kill_5s': 'label_pov_kills',
        'self_kill_10s': 'label_pov_kills',
        'self_kill_20s': 'label_pov_kills',
        # Bomb tasks
        'global_bombPlanted': 'label_bomb_planted',
        'global_bombSite': 'label_bomb_site',
        'global_willPlant': 'label_will_plant',
        'global_postPlantOutcome': 'label_outcome',
        # Round tasks
        'global_roundWinner': 'label_round_winner',
        'global_roundOutcome': 'label_outcome_reason',
    }
    
    # Check if task has a specific label column mapping
    if task_id in TASK_LABEL_COLUMNS:
        return TASK_LABEL_COLUMNS[task_id]
    
    # Multi-label classification and multi-output regression use label_0, label_1, etc.
    if ml_form == 'multi_label_cls':
        # Multi-label tasks have one column per class (e.g., 23 location classes)
        # For location tasks, we have 23 columns (label_0 to label_22)
        # For movement direction, we have 4 columns (label_0 to label_3) for 4 teammates
        if 'location' in task_id and ('teammate' in task_id or 'enemy' in task_id):
            # Location multi-label: 23 places
            return ';'.join([f'label_{i}' for i in range(23)])
        elif 'movementDir' in task_id and 'teammate' in task_id:
            # Teammate movement direction: 4 teammates
            return ';'.join([f'label_{i}' for i in range(4)])
        else:
            # Default: use output_dim
            return ';'.join([f'label_{i}' for i in range(output_dim)])
    
    # Multi-output regression (e.g., teammate_speed with 4 outputs)
    if ml_form == 'regression' and output_dim > 1:
        return ';'.join([f'label_{i}' for i in range(output_dim)])
    
    # Default: single 'label' column
    return 'label'


def apply_task_config(cfg, data_path: Path):
    """
    Apply task-specific configuration based on task_id.
    
    Loads task definition from task_definitions.csv and updates cfg.task
    with ml_form, num_classes, output_dim, label_column, and labels_filename.
    
    Args:
        cfg: OmegaConf configuration
        data_path: Path to data directory
        
    Returns:
        Updated configuration
    """
    task_id = cfg.task.task_id
    
    try:
        task_config = load_task_config(task_id, data_path)
        
        # Update task configuration
        task_updates = OmegaConf.create({
            'task': {
                'ml_form': task_config['ml_form'],
                'num_classes': task_config['num_classes'],
                'output_dim': task_config['output_dim'],
                'label_column': task_config['label_column'],
            },
            'data': {
                'labels_filename': task_config['labels_filename'],
            }
        })
        
        cfg = OmegaConf.merge(cfg, task_updates)
        print(f"[Task Config] Loaded config for task '{task_id}':")
        print(f"  ml_form: {task_config['ml_form']}")
        print(f"  num_classes: {task_config['num_classes']}")
        print(f"  output_dim: {task_config['output_dim']}")
        print(f"  label_column: {task_config['label_column']}")
        print(f"  labels_filename: {task_config['labels_filename']}")
        
    except Exception as e:
        print(f"[Task Config] Warning: Could not auto-load task config: {e}")
        print("[Task Config] Using config values from YAML file")
    
    return cfg



def apply_cfg_overrides(cfg, overrides):
    """Apply cfg overrides using OmegaConf
    
    Args:
        cfg (DictConfig): Original OmegaConf configuration
        overrides (list): List of override strings like ['meta.seed=42', 'data.batch_size=16']
        
    Returns:
        DictConfig: Updated OmegaConf configuration
    """
    if not overrides:
        return cfg
    
    print(f"Applying {len(overrides)} cfg overrides:")
    
    for override_str in overrides:
        try:
            # OmegaConf can parse the override string directly
            override_cfg = OmegaConf.from_dotlist([override_str])
            
            # Show what's being changed
            key_path = override_str.split('=')[0]
            old_value = OmegaConf.select(cfg, key_path, default='<not set>')
            new_value = override_str.split('=')[1]
            print(f"  {key_path}: {old_value} -> {new_value}")
            
            # Merge the override into the main cfg
            cfg = OmegaConf.merge(cfg, override_cfg)
            
        except Exception as e:
            print(f"Warning: Failed to apply override '{override_str}': {e}")
            continue
    
    return cfg
