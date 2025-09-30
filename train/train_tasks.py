# Local imports
from models.multi_agent_location_predictor import MultiAgentEnemyLocationPredictionModel
# from models.ctfm_contrastive import CTFMContrastive
from data_module.enemy_location_nowcast import EnemyLocationPredictionDataModule
from data_module.teammate_location_forecast import SelfTeamFutureLocationPredictionDataModule
# from data_module.contrastive import CTFMContrastiveDataModule
from train.train_pipeline import run_training_pipeline


def train_teammate_location_forecast(cfg):
    """Training mode implementation for multi-agent self-team future location prediction"""
    run_training_pipeline(
        cfg=cfg,
        model_class=MultiAgentEnemyLocationPredictionModel,
        datamodule_class=SelfTeamFutureLocationPredictionDataModule,
        task_name="teammate_location_forecast",
        print_header="=== TRAINING MODE MULTI-AGENT SELF-TEAM FUTURE LOCATION PREDICTION ==="
    )


def train_enemy_location_nowcast(cfg):
    """Training mode implementation for multi-agent enemy location nowcast"""
    run_training_pipeline(
        cfg=cfg,
        model_class=MultiAgentEnemyLocationPredictionModel,
        datamodule_class=EnemyLocationPredictionDataModule,
        task_name="enemy_location_nowcast",
        print_header="=== TRAINING MODE MULTI-AGENT ENEMY LOCATION PREDICTION ==="
    )


def train_enemy_location_forecast(cfg):
    """Training mode implementation for multi-agent enemy location forecast"""
    run_training_pipeline(
        cfg=cfg,
        model_class=MultiAgentEnemyLocationPredictionModel,
        datamodule_class=EnemyLocationPredictionDataModule,
        task_name="enemy_location_forecast",
        print_header="=== TRAINING MODE MULTI-AGENT ENEMY LOCATION PREDICTION ==="
    )


# def train_contrastive(cfg):
#     """Training mode implementation for contrastive learning"""
#     run_training_pipeline(
#         cfg=cfg,
#         model_class=CTFMContrastive,
#         datamodule_class=CTFMContrastiveDataModule,
#         task_name="contrastive",
#         print_header="=== TRAINING MODE CONTRASTIVE ==="
#     )
