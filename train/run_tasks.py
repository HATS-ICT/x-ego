# Local imports
from models.cross_ego_video_location_net import CrossEgoVideoLocationNet
from data_module.enemy_location_nowcast import EnemyLocationNowcastDataModule
from data_module.teammate_location_nowcast import TeammateLocationNowcastDataModule
from train.train_pipeline import run_training_pipeline
from train.test_pipeline import run_test_only_pipeline


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


