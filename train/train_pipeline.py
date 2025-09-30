import lightning as L
import wandb
import torch
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint

# Local imports
from ctfm.models.utils import setup_callbacks, setup_logger


def run_training_pipeline(config, model_class, datamodule_class, task_name, print_header=None):
    """
    Generic training pipeline that handles common training logic.
    
    Args:
        config: Training configuration
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
    L.seed_everything(config['meta']['seed'], workers=True)
    
    # Create datamodule
    print("Creating datamodule...")
    # Handle different datamodule initialization patterns
    if task_name == "contrastive":
        datamodule = datamodule_class(config=config)
    else:
        datamodule = datamodule_class(config)
    datamodule.prepare_data()
    datamodule.setup("fit")
    
    print(f"Train samples: {len(datamodule.train_dataset) if datamodule.train_dataset else 0}")
    print(f"Validation samples: {len(datamodule.val_dataset) if datamodule.val_dataset else 0}")
    
    # Print task-specific information
    print_task_info(config, datamodule, task_name)
    
    print(f"Creating {task_name} model...")
    model = model_class(config)
    
    # Update model with dataset info for location prediction tasks
    setup_model_with_dataset_info(config, datamodule, model)
    
    if config['training']['torch_compile']:
        model = torch.compile(model)
    
    callbacks = setup_callbacks(config)
    logger = setup_logger(config)
    
    # Create trainer
    trainer = create_trainer(config, callbacks, logger)
    
    # Log hyperparameters to wandb
    if logger is not None:
        logger.log_hyperparams(config)
    
    # Start training
    run_training(config, trainer, model, datamodule)
    
    # Run testing
    run_testing(config, datamodule, model_class, trainer, callbacks, task_name)
    
    # Finish wandb run
    if logger is not None:
        wandb.finish()
    
    print(f"{task_name.replace('_', ' ').title()} training completed!")


def print_task_info(config, datamodule, task_name):
    """Print task-specific information"""
    # Common info for location prediction tasks
    if 'location' in task_name:
        print(f"Number of agents: {config['data']['num_agents']}")
        print(f"Location form: {config['data']['task_form']}")
        print(f"Agent fusion method: {config['model']['agent_fusion_method']}")
        
        if config['data']['task_form'] in ['regression', 'generative']:
            print(f"Loss function: {config['data']['loss_fn']}")
            if config['data']['loss_fn'] == 'sinkhorn':
                print(f"  Sinkhorn blur: {config['data']['sinkhorn_blur']}")
                print(f"  Sinkhorn scaling: {config['data']['sinkhorn_scaling']}")
        
        if config['data']['task_form'] == 'classification':
            # Get num_places from datamodule after setup and update config
            config['num_places'] = datamodule.num_places
            print(f"Number of places: {config['num_places']}")


def setup_model_with_dataset_info(config, datamodule, model):
    """Setup model with dataset-specific information"""
    if config['data']['task_form'] in ['regression', 'generative']:
        # Set coordinate scaler from dataset for location prediction tasks
        if hasattr(datamodule, 'train_dataset') and hasattr(datamodule.train_dataset, 'get_coordinate_scaler'):
            scaler = datamodule.train_dataset.get_coordinate_scaler()
            if hasattr(model, 'set_coordinate_scaler'):
                model.set_coordinate_scaler(scaler)


def create_trainer(config, callbacks, logger):
    """Create Lightning trainer with common configuration"""
    training_config = config['training']
    
    # Configure gradient clipping - disable if fused optimizer is enabled
    fused_optimizer = config['optimization']['fused_optimizer']
    gradient_clip_val = None if fused_optimizer else training_config['gradient_clip_val']
    
    trainer = L.Trainer(
        max_epochs=training_config['max_epochs'],
        accelerator=training_config['accelerator'],
        devices=training_config['devices'],
        strategy=training_config['strategy'],
        precision=training_config['precision'],
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=training_config['accumulate_grad_batches'],
        val_check_interval=training_config['val_check_interval'],
        check_val_every_n_epoch=training_config['check_val_every_n_epoch'],
        log_every_n_steps=training_config['log_every_n_steps'],
        enable_checkpointing=training_config['enable_checkpointing'],
        enable_progress_bar=training_config['enable_progress_bar'],
        enable_model_summary=training_config['enable_model_summary'],
        callbacks=callbacks,
        logger=logger,
        deterministic=training_config['deterministic'],
        fast_dev_run=config['meta']['fast_dev_run'],
        num_sanity_val_steps=training_config['num_sanity_val_steps'],
        limit_train_batches=training_config['limit_train_batches'],
        limit_val_batches=training_config['limit_val_batches'],
        limit_test_batches=training_config['limit_test_batches'],
    )
    
    return trainer


def run_training(config, trainer, model, datamodule):
    """Run the training process"""
    if 'resume_checkpoint_path' in config['checkpoint']:
        print(f"Starting training from checkpoint: {config['checkpoint']['resume_checkpoint_path']}")
        trainer.fit(model, datamodule, ckpt_path=str(config['checkpoint']['resume_checkpoint_path']))
    else:
        print("Starting training from scratch...")
        trainer.fit(model, datamodule)


def run_testing(config, datamodule, model_class, trainer, callbacks, task_name):
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
        test_model = model_class(config)
        
        # Update model with dataset info for location prediction tasks
        setup_test_model_with_dataset_info(config, datamodule, test_model)
        
        # Store the checkpoint name for the model to use
        test_model.checkpoint_name = checkpoint_name
        
        # Clear any previous test results
        if hasattr(test_model, 'test_predictions'):
            test_model.test_predictions = []
        if hasattr(test_model, 'test_targets'):
            test_model.test_targets = []
        
        try:
            trainer.test(test_model, datamodule, ckpt_path=checkpoint_path)
            print(f"Test evaluation completed for {checkpoint_name} checkpoint")
        except Exception as e:
            print(f"Error testing {checkpoint_name} checkpoint: {e}")
            if hasattr(e, '__traceback__'):
                import traceback
                print("Full traceback:")
                traceback.print_exc()
            continue


def setup_test_model_with_dataset_info(config, datamodule, test_model):
    """Setup test model with dataset-specific information"""
    if config['data']['task_form'] in ['regression', 'generative']:
        # Set coordinate scaler from dataset
        if hasattr(datamodule, 'test_dataset') and hasattr(datamodule.test_dataset, 'get_coordinate_scaler'):
            scaler = datamodule.test_dataset.get_coordinate_scaler()
            if hasattr(test_model, 'set_coordinate_scaler'):
                test_model.set_coordinate_scaler(scaler)


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
                # For save_top_k == -1 (save all), we can manually find the best by looking at saved checkpoints
                elif callback.save_top_k == -1 and callback.dirpath:
                    print(f"save_top_k=-1 detected, will try to find best checkpoint manually in {callback.dirpath}")
                    try:
                        checkpoint_dir = Path(callback.dirpath)
                        if checkpoint_dir.exists():
                            # Find all checkpoint files and select the one with best validation loss
                            checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
                            checkpoint_files = [f for f in checkpoint_files if f.name != "last.ckpt"]  # Exclude last.ckpt
                            
                            if checkpoint_files:
                                # Sort by validation loss (assuming filename contains val loss)
                                best_ckpt = None
                                best_loss = float('inf')
                                
                                for ckpt_file in checkpoint_files:
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
                                    checkpoint_paths.append(("best", str(best_ckpt)))
                                    print(f"Found best checkpoint manually: {best_ckpt} (loss: {best_loss})")
                                    best_checkpoint_found = True
                                    break
                    except Exception as e:
                        print(f"Error finding best checkpoint manually: {e}")
                        
    if not best_checkpoint_found:
        print("No best checkpoint found - will only test on last checkpoint")
    
    return checkpoint_paths
