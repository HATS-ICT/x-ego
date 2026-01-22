# ============================================================================
# Stage 1: Contrastive Learning (Team Alignment)
# ============================================================================

def train_contrastive(cfg):
    """
    Stage 1: Train contrastive learning model for team alignment.
    
    This trains a video encoder to produce embeddings where agents from
    the same team are aligned in the embedding space.
    """
    from src.models.contrastive_model import ContrastiveModel
    from src.data_module.contrastive import ContrastiveDataModule
    from src.train.train_pipeline import run_training_pipeline
    
    run_training_pipeline(
        cfg=cfg,
        model_class=ContrastiveModel,
        datamodule_class=ContrastiveDataModule,
        task_name="contrastive",
        print_header="=== STAGE 1: CONTRASTIVE LEARNING (TEAM ALIGNMENT) ==="
    )


def test_contrastive(cfg):
    """Test-only mode for contrastive learning model."""
    from src.models.contrastive_model import ContrastiveModel
    from src.data_module.contrastive import ContrastiveDataModule
    from src.train.test_pipeline import run_test_only_pipeline
    
    run_test_only_pipeline(
        cfg=cfg,
        model_class=ContrastiveModel,
        datamodule_class=ContrastiveDataModule,
        task_name="contrastive"
    )


# ============================================================================
# Stage 2: Linear Probing on Downstream Tasks
# ============================================================================

def train_downstream(cfg):
    """
    Stage 2: Train linear probe on a downstream task.
    
    Uses frozen encoder (off-the-shelf HuggingFace or pretrained from Stage 1)
    and trains a linear head for the specified task.
    
    Supports task types: binary_cls, multi_cls, multi_label_cls, regression.
    """
    from src.models.downstream import LinearProbeModel
    from src.data_module.downstream import DownstreamDataModule
    from src.train.train_pipeline import run_training_pipeline
    
    task_id = cfg.task.task_id
    ml_form = cfg.task.ml_form
    
    run_training_pipeline(
        cfg=cfg,
        model_class=LinearProbeModel,
        datamodule_class=DownstreamDataModule,
        task_name="downstream",
        print_header=f"=== STAGE 2: LINEAR PROBING - {task_id} ({ml_form}) ==="
    )


def test_downstream(cfg):
    """Test-only mode for linear probe model."""
    from src.models.downstream import LinearProbeModel
    from src.data_module.downstream import DownstreamDataModule
    from src.train.test_pipeline import run_test_only_pipeline
    
    run_test_only_pipeline(
        cfg=cfg,
        model_class=LinearProbeModel,
        datamodule_class=DownstreamDataModule,
        task_name="downstream"
    )




