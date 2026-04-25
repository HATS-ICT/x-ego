import lightning as L
import wandb
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
import sys

# Local imports
from .training_utils import setup_callbacks, setup_logger, print_task_info, setup_test_model_with_dataset_info


def run_training_pipeline(cfg, model_class, datamodule_class, task_name, print_header=None):
    """
    Generic training pipeline that handles common training logic.
    
    Args:
        cfg: Training configuration
        model_class: Model class to instantiate
        datamodule_class: DataModule class to instantiate
        task_name: Name of the task for logging
        print_header: Optional custom header to print, defaults to task_name based header
    """
    # Default header if none provided
    if print_header is None:
        print_header = f"=== TRAINING MODE {task_name.upper().replace('_', ' ')} ==="
    
    print(print_header)
    
    # Set tensor core precision for better performance
    torch.set_float32_matmul_precision('medium')
    L.seed_everything(cfg.meta.seed, workers=True)
    
    # Create datamodule
    print("Creating datamodule...")
    datamodule = datamodule_class(cfg)
    datamodule.prepare_data()
    datamodule.setup("fit")
    
    print(f"Train samples: {len(datamodule.train_dataset) if datamodule.train_dataset else 0}")
    print(f"Validation samples: {len(datamodule.val_dataset) if datamodule.val_dataset else 0}")
    
    # Print task-specific information
    print_task_info(cfg, datamodule, task_name)
    
    print(f"Creating {task_name} model...")
    model = model_class(cfg)
    
    # Update model with dataset info for location prediction tasks
    setup_model_with_dataset_info(cfg, datamodule, model)
    
    if cfg.training.torch_compile and sys.platform.startswith('linux'):
        model.video_encoder = torch.compile(model.video_encoder)
    
    callbacks = setup_callbacks(cfg)
    logger = setup_logger(cfg)
    
    # Create trainer
    trainer = create_trainer(cfg, callbacks, logger)
    
    # Log hyperparameters to wandb
    if logger is not None:
        logger.log_hyperparams(cfg)
    
    # Start training
    run_training(cfg, trainer, model, datamodule)
    
    # Run testing
    run_testing(cfg, datamodule, model_class, trainer, callbacks, task_name)
    
    # Finish wandb run
    if logger is not None:
        wandb.finish()
    
    print(f"{task_name.replace('_', ' ').title()} training completed!")


def setup_model_with_dataset_info(cfg, datamodule, model):
    """Setup model with dataset-specific information"""
    # Model setup logic can be added here if needed in the future
    pass


def create_trainer(cfg, callbacks, logger):
    """Create Lightning trainer with common configuration"""
    training_cfg = cfg.training
    
    # Configure gradient clipping - disable only for fused AdamW.
    optimizer_name = getattr(cfg.optimization, "optimizer", "adamw")
    fused_adamw = optimizer_name == "adamw" and cfg.optimization.fused_optimizer
    gradient_clip_val = None if fused_adamw else training_cfg.gradient_clip_val
    
    trainer = L.Trainer(
        max_epochs=training_cfg.max_epochs,
        max_steps=training_cfg.max_steps,
        accelerator=training_cfg.accelerator,
        devices=training_cfg.devices,
        strategy=training_cfg.strategy,
        precision=training_cfg.precision,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=training_cfg.accumulate_grad_batches,
        val_check_interval=training_cfg.val_check_interval,
        check_val_every_n_epoch=training_cfg.check_val_every_n_epoch,
        log_every_n_steps=training_cfg.log_every_n_steps,
        enable_checkpointing=training_cfg.enable_checkpointing,
        enable_progress_bar=training_cfg.enable_progress_bar,
        enable_model_summary=training_cfg.enable_model_summary,
        callbacks=callbacks,
        logger=logger,
        deterministic=training_cfg.deterministic,
        num_sanity_val_steps=training_cfg.num_sanity_val_steps,
        limit_train_batches=training_cfg.limit_train_batches,
        limit_val_batches=training_cfg.limit_val_batches,
        limit_test_batches=training_cfg.limit_test_batches,
    )
    
    return trainer


def run_training(cfg, trainer, model, datamodule):
    """Run the training process"""
    if 'resume_checkpoint_path' in cfg.checkpoint:
        print(f"Starting training from checkpoint: {cfg.checkpoint.resume_checkpoint_path}")
        trainer.fit(model, datamodule, ckpt_path=str(cfg.checkpoint.resume_checkpoint_path), weights_only=False)
    else:
        print("Starting training from scratch...")
        trainer.fit(model, datamodule)


def run_testing(cfg, datamodule, model_class, trainer, callbacks, task_name):
    """Run testing on test dataset with both last and best checkpoints"""
    print("Setting up datamodule for testing...")
    datamodule.setup("test")
    
    if datamodule.test_dataset is None:
        print("No test dataset available - skipping test evaluation")
        return
    
    # Find checkpoints to test
    checkpoint_paths = find_checkpoints_to_test(callbacks)
    
    print(f"Will test on {len(checkpoint_paths)} checkpoints: {[cp[0] for cp in checkpoint_paths]}")
    
    # Run tests on each checkpoint
    for checkpoint_name, checkpoint_path in checkpoint_paths:
        print(f"\n=== Running test evaluation on {checkpoint_name} checkpoint ===")
        print(f"Checkpoint path: {checkpoint_path}")
        
        # Create fresh model for testing
        test_model = model_class(cfg)
        
        # Update model with dataset info for location prediction tasks
        setup_test_model_with_dataset_info(cfg, datamodule, test_model)
        
        # Store the checkpoint name for the model to use
        test_model.checkpoint_name = checkpoint_name
        
        # Clear any previous test results
        if hasattr(test_model, 'test_predictions'):
            test_model.test_predictions = []
        if hasattr(test_model, 'test_targets'):
            test_model.test_targets = []
        
        trainer.test(test_model, datamodule, ckpt_path=checkpoint_path, weights_only=False)
        print(f"Test evaluation completed for {checkpoint_name} checkpoint")


def find_checkpoints_to_test(callbacks):
    """Find checkpoints to test (last and best)"""
    checkpoint_paths = []
    
    # Always test on last checkpoint
    checkpoint_paths.append(("last", "last"))
    
    # Find and test on best checkpoint if available
    best_checkpoint_found = False
    if callbacks:
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint):
                print(f"Found ModelCheckpoint callback with save_top_k: {callback.save_top_k}")
                print(f"Best model path: {callback.best_model_path}")
                
                # For save_top_k > 0, Lightning tracks the best model
                if callback.save_top_k > 0 and callback.best_model_path and callback.best_model_path != "":
                    checkpoint_paths.append(("best", callback.best_model_path))
                    print(f"Added best checkpoint: {callback.best_model_path}")
                    best_checkpoint_found = True
                    break
    if not best_checkpoint_found:
        print("No best checkpoint found - will only test on last checkpoint")
    
    return checkpoint_paths
