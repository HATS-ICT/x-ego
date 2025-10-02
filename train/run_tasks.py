# Local imports
from models.cross_ego_video_location_net import CrossEgoVideoLocationNet
from data_module.enemy_location_nowcast import EnemyLocationNowcastDataModule
from data_module.enemy_location_forecast import EnemyLocationForecastDataModule
from data_module.teammate_location_forecast import TeammateLocationForecastDataModule
from train.train_pipeline import run_training_pipeline
from train.test_pipeline import run_test_only_pipeline


def train_teammate_location_forecast(cfg):
    """Training mode implementation for multi-agent self-team future location prediction"""
    run_training_pipeline(
        cfg=cfg,
        model_class=CrossEgoVideoLocationNet,
        datamodule_class=TeammateLocationForecastDataModule,
        task_name="teammate_location_forecast",
        print_header="=== TRAINING MODE MULTI-AGENT SELF-TEAM FUTURE LOCATION PREDICTION ==="
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


def train_enemy_location_forecast(cfg):
    """Training mode implementation for multi-agent enemy location forecast"""
    run_training_pipeline(
        cfg=cfg,
        model_class=CrossEgoVideoLocationNet,
        datamodule_class=EnemyLocationForecastDataModule,
        task_name="enemy_location_forecast",
        print_header="=== TRAINING MODE MULTI-AGENT ENEMY FUTURE LOCATION PREDICTION ==="
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


def test_enemy_location_forecast(cfg):
    """Test-only mode for multi-agent enemy location forecast"""
    run_test_only_pipeline(
        cfg=cfg,
        model_class=CrossEgoVideoLocationNet,
        datamodule_class=EnemyLocationForecastDataModule,
        task_name="enemy_location_forecast"
    )


def test_teammate_location_forecast(cfg):
    """Test-only mode for multi-agent self-team future location prediction"""
    run_test_only_pipeline(
        cfg=cfg,
        model_class=CrossEgoVideoLocationNet,
        datamodule_class=TeammateLocationForecastDataModule,
        task_name="teammate_location_forecast"
    )
