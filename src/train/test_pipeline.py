import lightning as L
import wandb
import torch
from pathlib import Path

# Local imports
from utils.training_utils import setup_logger, print_task_info, setup_test_model_with_dataset_info



def run_test_only_pipeline(cfg, model_class, datamodule_class, task_name, checkpoint_path=None):
    """
    Test-only pipeline that loads a checkpoint and runs evaluation.
    
    Args:
        cfg: Configuration with experiment directory and checkpoint info
        model_class: Model class to instantiate
        datamodule_class: DataModule class to instantiate
        task_name: Name of the task for logging
        checkpoint_path: Specific checkpoint path to test (if None, uses 'last' and 'best')
    """
    print(f"=== TEST ONLY MODE {task_name.upper().replace('_', ' ')} ===")
    
    # Set tensor core precision for better performance
    torch.set_float32_matmul_precision('medium')
    L.seed_everything(cfg.meta.seed, workers=True)
    
    # Create datamodule for testing
    print("Creating datamodule for testing...")
    if task_name == "contrastive":
        datamodule = datamodule_class(config=cfg)
    else:
        datamodule = datamodule_class(cfg)
    datamodule.prepare_data()
    datamodule.setup("test")
    
    if datamodule.test_dataset is None:
        print("No test dataset available - cannot run test evaluation")
        return
    
    print(f"Test samples: {len(datamodule.test_dataset)}")
    
    # Print task-specific information
    print_task_info(cfg, datamodule, task_name)
    
    # Setup logger (optional for test-only mode)
    logger = setup_logger(cfg) if cfg.wandb.enabled else None
    
    # Create trainer for testing
    trainer = create_test_trainer(cfg, logger)
    
    # Determine which checkpoints to test
    if checkpoint_path:
        # Test specific checkpoint
        checkpoints_to_test = [("specified", checkpoint_path)]
    else:
        # Test 'last' and 'best' checkpoints from the experiment directory
        checkpoints_to_test = find_saved_checkpoints(cfg.path.ckpt)
    
    print(f"Will test on {len(checkpoints_to_test)} checkpoint(s): {[cp[0] for cp in checkpoints_to_test]}")
    
    # Run tests on each checkpoint
    for checkpoint_name, ckpt_path in checkpoints_to_test:
        print(f"\n=== Running test evaluation on {checkpoint_name} checkpoint ===")
        print(f"Checkpoint path: {ckpt_path}")
        
        # Create fresh model for testing
        test_model = model_class(cfg)
        
        # Update model with dataset info
        setup_test_model_with_dataset_info(cfg, datamodule, test_model)
        
        # Store the checkpoint name for the model to use
        test_model.checkpoint_name = checkpoint_name
        
        try:
            trainer.test(test_model, datamodule, ckpt_path=ckpt_path, weights_only=False)
            print(f"Test evaluation completed for {checkpoint_name} checkpoint")
        except Exception as e:
            print(f"Error testing {checkpoint_name} checkpoint: {e}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            continue
    
    # Finish wandb run
    if logger is not None:
        wandb.finish()
    
    print(f"\n{task_name.replace('_', ' ').title()} testing completed!")


def create_test_trainer(cfg, logger):
    """Create Lightning trainer for testing only"""
    training_cfg = cfg.training
    
    trainer = L.Trainer(
        accelerator=training_cfg.accelerator,
        devices=training_cfg.devices,
        precision=training_cfg.precision,
        logger=logger,
        enable_progress_bar=training_cfg.enable_progress_bar,
        limit_test_batches=training_cfg.limit_test_batches,
    )
    
    return trainer


def find_saved_checkpoints(checkpoint_dir):
    """
    Find saved checkpoints in the experiment directory.
    Returns both 'last' and 'best' checkpoints if available.
    """
    checkpoints = []
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    # Check for last.ckpt
    last_ckpt = checkpoint_path / "last.ckpt"
    if last_ckpt.exists():
        checkpoints.append(("last", str(last_ckpt)))
        print(f"Found last checkpoint: {last_ckpt}")
    
    # Find best checkpoint by looking for lowest validation loss
    ckpt_files = list(checkpoint_path.glob("*.ckpt"))
    ckpt_files = [f for f in ckpt_files if f.name != "last.ckpt"]
    
    if ckpt_files:
        best_ckpt = None
        best_loss = float('inf')
        
        for ckpt_file in ckpt_files:
            try:
                # Extract loss from filename (format: *-l{loss}.ckpt)
                if '-l' in ckpt_file.stem:
                    loss_part = ckpt_file.stem.split('-l')[-1]
                    loss_value = float(loss_part)
                    if loss_value < best_loss:
                        best_loss = loss_value
                        best_ckpt = ckpt_file
            except (ValueError, IndexError):
                continue
        
        if best_ckpt:
            checkpoints.append(("best", str(best_ckpt)))
            print(f"Found best checkpoint: {best_ckpt} (val_loss: {best_loss})")
    
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    return checkpoints

