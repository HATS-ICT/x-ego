# Local imports
from ctfm.models.multi_agent_enemy_location_prediction import MultiAgentEnemyLocationPredictionModel
from ctfm.models.ctfm_contrastive import CTFMContrastive
from ctfm.dataset.data_module.data_module_enemy_location_prediction import EnemyLocationPredictionDataModule
from ctfm.dataset.data_module.data_module_self_team_future_location_prediction import SelfTeamFutureLocationPredictionDataModule
from ctfm.dataset.data_module.data_module import CTFMContrastiveDataModule
from train.train_pipeline import run_training_pipeline


def train_teammate_location_forecast(config):
    """Training mode implementation for multi-agent self-team future location prediction"""
    run_training_pipeline(
        config=config,
        model_class=MultiAgentEnemyLocationPredictionModel,
        datamodule_class=SelfTeamFutureLocationPredictionDataModule,
        task_name="teammate_location_forecast",
        print_header="=== TRAINING MODE MULTI-AGENT SELF-TEAM FUTURE LOCATION PREDICTION ==="
    )


def train_enemy_location_nowcast(config):
    """Training mode implementation for multi-agent enemy location nowcast"""
    run_training_pipeline(
        config=config,
        model_class=MultiAgentEnemyLocationPredictionModel,
        datamodule_class=EnemyLocationPredictionDataModule,
        task_name="enemy_location_nowcast",
        print_header="=== TRAINING MODE MULTI-AGENT ENEMY LOCATION PREDICTION ==="
    )


def train_enemy_location_forecast(config):
    """Training mode implementation for multi-agent enemy location forecast"""
    run_training_pipeline(
        config=config,
        model_class=MultiAgentEnemyLocationPredictionModel,
        datamodule_class=EnemyLocationPredictionDataModule,
        task_name="enemy_location_forecast",
        print_header="=== TRAINING MODE MULTI-AGENT ENEMY LOCATION PREDICTION ==="
    )


def train_contrastive(config):
    """Training mode implementation for contrastive learning"""
    run_training_pipeline(
        config=config,
        model_class=CTFMContrastive,
        datamodule_class=CTFMContrastiveDataModule,
        task_name="contrastive",
        print_header="=== TRAINING MODE CONTRASTIVE ==="
    )
