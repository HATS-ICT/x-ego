# Local imports
from models.cross_ego_video_location_net import CrossEgoVideoLocationNet
from models.contrastive_model import ContrastiveModel
from models.linear_probe import LinearProbeModel
from data_module.enemy_location_nowcast import EnemyLocationNowcastDataModule
from data_module.teammate_location_nowcast import TeammateLocationNowcastDataModule
from data_module.contrastive import ContrastiveDataModule
from data_module.linear_probe import LinearProbeDataModule
from train.train_pipeline import run_training_pipeline
from train.test_pipeline import run_test_only_pipeline


# ============================================================================
# Stage 1: Contrastive Learning (Team Alignment)
# ============================================================================

def train_contrastive(cfg):
    """
    Stage 1: Train contrastive learning model for team alignment.
    
    This trains a video encoder to produce embeddings where agents from
    the same team are aligned in the embedding space.
    """
    run_training_pipeline(
        cfg=cfg,
        model_class=ContrastiveModel,
        datamodule_class=ContrastiveDataModule,
        task_name="contrastive",
        print_header="=== STAGE 1: CONTRASTIVE LEARNING (TEAM ALIGNMENT) ==="
    )


def test_contrastive(cfg):
    """Test-only mode for contrastive learning model."""
    run_test_only_pipeline(
        cfg=cfg,
        model_class=ContrastiveModel,
        datamodule_class=ContrastiveDataModule,
        task_name="contrastive"
    )


# ============================================================================
# Stage 2: Linear Probing on Downstream Tasks
# ============================================================================

def train_linear_probe(cfg):
    """
    Stage 2: Train linear probe on a downstream task.
    
    Uses frozen encoder from Stage 1 and trains a linear head
    for the specified task (binary_cls, multi_cls, multi_label_cls, regression).
    """
    task_id = cfg.task.task_id
    ml_form = cfg.task.ml_form
    
    run_training_pipeline(
        cfg=cfg,
        model_class=LinearProbeModel,
        datamodule_class=LinearProbeDataModule,
        task_name="linear_probe",
        print_header=f"=== STAGE 2: LINEAR PROBING - {task_id} ({ml_form}) ==="
    )


def test_linear_probe(cfg):
    """Test-only mode for linear probe model."""
    run_test_only_pipeline(
        cfg=cfg,
        model_class=LinearProbeModel,
        datamodule_class=LinearProbeDataModule,
        task_name="linear_probe"
    )


# ============================================================================
# Stage 2 and Legacy Tasks: Location Prediction
# ============================================================================

def train_teammate_location_nowcast(cfg):
    """Training mode implementation for multi-agent teammate location nowcast"""
    run_training_pipeline(
        cfg=cfg,
        model_class=CrossEgoVideoLocationNet,
        datamodule_class=TeammateLocationNowcastDataModule,
        task_name="teammate_location_nowcast",
        print_header="=== TRAINING MODE MULTI-AGENT TEAMMATE LOCATION NOWCAST ==="
    )


def train_enemy_location_nowcast(cfg):
    """Training mode implementation for multi-agent enemy location nowcast"""
    run_training_pipeline(
        cfg=cfg,
        model_class=CrossEgoVideoLocationNet,
        datamodule_class=EnemyLocationNowcastDataModule,
        task_name="enemy_location_nowcast",
        print_header="=== TRAINING MODE MULTI-AGENT ENEMY LOCATION PREDICTION ==="
    )


# ============================================================================
# Test-only mode functions
# ============================================================================

def test_enemy_location_nowcast(cfg):
    """Test-only mode for multi-agent enemy location nowcast"""
    run_test_only_pipeline(
        cfg=cfg,
        model_class=CrossEgoVideoLocationNet,
        datamodule_class=EnemyLocationNowcastDataModule,
        task_name="enemy_location_nowcast"
    )


def test_teammate_location_nowcast(cfg):
    """Test-only mode for multi-agent teammate location nowcast"""
    run_test_only_pipeline(
        cfg=cfg,
        model_class=CrossEgoVideoLocationNet,
        datamodule_class=TeammateLocationNowcastDataModule,
        task_name="teammate_location_nowcast"
    )


