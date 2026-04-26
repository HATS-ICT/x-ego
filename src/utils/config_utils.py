"""
Configuration utilities for X-EGO project.
Handles loading, validation, and manipulation of training configurations.
"""

from pathlib import Path
import pandas as pd
from omegaconf import OmegaConf

from src.scripts.task_creator.task_definitions import get_place_names_for_map


_MISSING = object()


CONTRASTIVE_REQUIRED_CONFIG_KEYS = (
    "meta.seed",
    "meta.run_name",
    "meta.resume_exp",
    "data.partition",
    "data.labels_folder",
    "data.video_folder",
    "data.map",
    "data.persistent_workers",
    "data.pin_mem",
    "data.batch_size",
    "data.num_workers",
    "data.prefetch_factor",
    "data.data_path_csv_filename",
    "data.fixed_duration_seconds",
    "data.target_fps",
    "data.video_column_name",
    "data.ui_mask",
    "data.random_mask.enable",
    "data.random_mask.num_tubes",
    "data.random_mask.min_size_ratio",
    "data.random_mask.max_size_ratio",
    "data.time_jitter_max_seconds",
    "data.labels_filename",
    "model.activation",
    "model.encoder.finetune_last_k_layers",
    "model.encoder.model_type",
    "model.encoder.temporal_heads",
    "model.encoder.temporal_depth",
    "model.projector.proj_dim",
    "model.projector.num_hidden_layers",
    "model.contrastive.enable",
    "model.contrastive.logit_scale_init",
    "model.contrastive.logit_bias_init",
    "model.contrastive.turn_off_bias",
    "model.contrastive.loss_weight",
    "training.max_epochs",
    "training.max_steps",
    "training.accelerator",
    "training.devices",
    "training.strategy",
    "training.precision",
    "training.gradient_clip_val",
    "training.accumulate_grad_batches",
    "training.contrastive_accumulate_batches",
    "training.val_check_interval",
    "training.check_val_every_n_epoch",
    "training.log_every_n_steps",
    "training.enable_checkpointing",
    "training.enable_progress_bar",
    "training.enable_model_summary",
    "training.torch_compile",
    "training.deterministic",
    "training.num_sanity_val_steps",
    "training.limit_train_batches",
    "training.limit_val_batches",
    "training.limit_test_batches",
    "optimization.optimizer",
    "optimization.lr",
    "optimization.betas",
    "optimization.weight_decay",
    "optimization.fused_optimizer",
    "optimization.muon.lr",
    "optimization.muon.weight_decay",
    "optimization.scheduler.type",
    "optimization.scheduler.warmup_steps",
    "optimization.scheduler.min_lr_ratio",
    "wandb.enabled",
    "wandb.project",
    "wandb.name",
    "wandb.group",
    "wandb.tags",
    "wandb.notes",
    "wandb.save_dir",
    "checkpoint.epoch.filename",
    "checkpoint.epoch.monitor",
    "checkpoint.epoch.mode",
    "checkpoint.epoch.save_top_k",
    "checkpoint.epoch.save_last",
    "checkpoint.epoch.auto_insert_metric_name",
    "checkpoint.epoch.save_on_train_epoch_end",
    "checkpoint.epoch.every_n_epochs",
    "checkpoint.step.filename",
    "checkpoint.step.monitor",
    "checkpoint.step.mode",
    "checkpoint.step.save_top_k",
    "checkpoint.step.save_last",
    "checkpoint.step.auto_insert_metric_name",
    "checkpoint.step.save_on_train_epoch_end",
    "checkpoint.step.every_n_train_steps",
)


def load_cfg(cfg_path):
    """Load configuration from YAML file using OmegaConf"""
    cfg = OmegaConf.load(cfg_path)
    return cfg


def validate_required_config_keys(cfg, required_keys, config_name: str) -> None:
    """Fail fast when a required config setting is absent."""
    missing = [
        key for key in required_keys
        if OmegaConf.select(cfg, key, default=_MISSING) is _MISSING
    ]
    if missing:
        formatted = "\n".join(f"  - {key}" for key in missing)
        raise ValueError(f"{config_name} is missing required config setting(s):\n{formatted}")


def validate_contrastive_cfg(cfg) -> None:
    """Validate that the contrastive pipeline is fully configured explicitly."""
    validate_required_config_keys(
        cfg,
        CONTRASTIVE_REQUIRED_CONFIG_KEYS,
        "contrastive config",
    )

    if not cfg.model.contrastive.enable:
        raise ValueError("model.contrastive.enable must be true for contrastive training")

    if cfg.data.ui_mask not in ("none", "minimap_only", "all"):
        raise ValueError(f"Unsupported data.ui_mask: {cfg.data.ui_mask}")

    if cfg.data.batch_size <= 0:
        raise ValueError("data.batch_size must be positive")
    if cfg.training.accumulate_grad_batches != 1:
        raise ValueError(
            "contrastive training uses embedding-cache accumulation; "
            "set training.accumulate_grad_batches to 1"
        )
    if cfg.training.contrastive_accumulate_batches <= 0:
        raise ValueError("training.contrastive_accumulate_batches must be positive")
    if cfg.data.num_workers < 0:
        raise ValueError("data.num_workers must be non-negative")
    if cfg.data.target_fps <= 0 or cfg.data.fixed_duration_seconds <= 0:
        raise ValueError("data.target_fps and data.fixed_duration_seconds must be positive")
    if cfg.data.random_mask.num_tubes < 0:
        raise ValueError("data.random_mask.num_tubes must be non-negative")
    if not (0 < cfg.data.random_mask.min_size_ratio <= cfg.data.random_mask.max_size_ratio <= 1):
        raise ValueError(
            "random mask size ratios must satisfy "
            "0 < min_size_ratio <= max_size_ratio <= 1"
        )

    if cfg.model.contrastive.logit_scale_init <= 0:
        raise ValueError("model.contrastive.logit_scale_init must be positive")
    if cfg.model.contrastive.loss_weight < 0:
        raise ValueError("model.contrastive.loss_weight must be non-negative")

def _task_id_to_labels_filename(task_id: str) -> str:
    """
    Convert task_id to the corresponding labels CSV filename.
    
    CSV files are named exactly as their task_id: {task_id}.csv
    """
    return f"{task_id}.csv"


def _find_task_definitions_path(data_path: Path, map_name: str | None) -> Path:
    """Find the most specific task_definitions.csv available."""
    candidates = []
    if map_name:
        candidates.append(data_path / str(map_name) / "labels" / "task_definitions.csv")
    candidates.append(data_path / "labels" / "task_definitions.csv")

    for path in candidates:
        if path.exists():
            return path

    for path in sorted(data_path.glob("*/labels/task_definitions.csv")):
        if path.exists():
            return path

    formatted = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Task definitions file not found. Checked:\n"
        f"{formatted}\n"
        f"  - {data_path}/*/labels/task_definitions.csv"
    )


def _uses_place_vocabulary(task_id: str, row: pd.Series | None = None) -> bool:
    """Return whether a task's output dimension is the map-specific place vocabulary."""
    if row is not None:
        category = str(row.get("category", "")).lower()
        label_field = str(row.get("label_field", "")).lower()
        if category == "location" or label_field == "place":
            return True
    return "location" in task_id


def load_task_config(task_id: str, data_path: Path, map_name: str | None = None) -> dict:
    """
    Load task configuration from task_definitions.csv based on task_id.
    
    Args:
        task_id: Task identifier (e.g., 'self_location_0s', 'enemy_location_5s')
        data_path: Path to data directory containing task_definitions.csv
        map_name: Optional map name for map-specific task dimensions
        
    Returns:
        Dictionary with task configuration (ml_form, num_classes, output_dim, label_column, labels_filename)
    """
    task_def_path = _find_task_definitions_path(data_path, map_name)
    
    df = pd.read_csv(task_def_path)
    task_row = df[df['task_id'] == task_id]
    
    if len(task_row) == 0:
        available_tasks = df['task_id'].tolist()
        raise ValueError(f"Task '{task_id}' not found in task_definitions.csv. Available: {available_tasks}")
    
    row = task_row.iloc[0]
    output_dim = int(row['output_dim'])
    num_classes = int(row['num_classes']) if pd.notna(row['num_classes']) else None

    if _uses_place_vocabulary(task_id, row):
        output_dim = len(get_place_names_for_map(map_name))
        if pd.notna(row['num_classes']):
            num_classes = output_dim
    
    # Determine label column based on task type
    label_column = _get_label_column_for_task(task_id, row['ml_form'], output_dim)
    
    # Determine labels filename from task_id
    labels_filename = _task_id_to_labels_filename(task_id)
    
    config = {
        'ml_form': row['ml_form'],
        'num_classes': num_classes,
        'output_dim': output_dim,
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
        # For location tasks, the number of place columns depends on the map.
        # For movement direction, we have 4 columns (label_0 to label_3) for 4 teammates
        if 'location' in task_id and ('teammate' in task_id or 'enemy' in task_id):
            # Location multi-label: one column per map-specific place.
            return ';'.join([f'label_{i}' for i in range(output_dim)])
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
    map_name = cfg.data.map
    
    try:
        task_config = load_task_config(task_id, data_path, map_name)
        
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
        if _uses_place_vocabulary(task_id):
            output_dim = len(get_place_names_for_map(map_name))
            task_updates = OmegaConf.create({
                'task': {
                    'num_classes': output_dim if cfg.task.num_classes is not None else None,
                    'output_dim': output_dim,
                    'label_column': _get_label_column_for_task(
                        task_id,
                        cfg.task.ml_form,
                        output_dim,
                    ),
                }
            })
            cfg = OmegaConf.merge(cfg, task_updates)
            print(
                f"[Task Config] Applied map-specific place vocabulary for "
                f"'{map_name}': {output_dim} classes"
            )
    
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
            old_value = OmegaConf.select(cfg, key_path, default=_MISSING)
            if old_value is _MISSING:
                raise KeyError(
                    f"Override targets missing config key '{key_path}'. "
                    "Add the setting to YAML before overriding it."
                )
            new_value = override_str.split('=')[1]
            print(f"  {key_path}: {old_value} -> {new_value}")
            
            # Merge the override into the main cfg
            cfg = OmegaConf.merge(cfg, override_cfg)
            
        except Exception as e:
            raise ValueError(f"Failed to apply override '{override_str}': {e}") from e
    
    return cfg
