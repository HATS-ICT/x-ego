import torch
from pathlib import Path
import warnings
import argparse

from omegaconf import OmegaConf

# Local imports
from src.utils.config_utils import load_cfg, apply_cfg_overrides, apply_task_config
from src.utils.experiment_utils import create_experiment_dir, save_hyperparameters, setup_resume_cfg, load_experiment_cfg
from src.train.run_tasks import (
    train_contrastive,
    test_contrastive,
    train_downstream,
    test_downstream,
)
from src.utils.env_utils import get_src_base_path, get_data_base_path, get_output_base_path


# TODO: To be removed
# Debug so that tensor automatically show shape
normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"

# Suppress various noisy warnings
warnings.filterwarnings("ignore", message="FutureWarning: functools.partial will be a method descriptor")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*functools.partial will be a method descriptor.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*No device id is provided via.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Redirects are currently not supported in Windows or MacOs.*")


def setup_argument_parser():
    """Setup argument parser with config override support"""
    parser = argparse.ArgumentParser(
        description='X-EGO Training and Testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Every settings in config file can be overridden. Examples:
  
  Training mode (includes testing after training):
    python main.py --mode train --task contrastive meta.seed=123
    python main.py --mode dev --task downstream training.devices=[0,1] data.num_workers=8
  
  Test-only mode (requires resume_exp):
    python main.py --mode test --task downstream meta.resume_exp=probe-self_location-siglip-260120-152532-pe8y
        """
    )
    
    parser.add_argument('--mode', 
                       choices=['train', 'test', 'dev'], 
                       default='train',
                       help='Mode to run: train, test, or dev')
    
    parser.add_argument('--task',
                       choices=['contrastive',
                                'downstream'],
                       default='contrastive',
                       help='Task to run: contrastive (stage 1) or downstream (stage 2)')
    
    parser.add_argument('--config', 
                       type=str,
                       help='Path to config file (auto-determined from mode and task if not specified)')
    
    # Accept any remaining arguments as config overrides
    parser.add_argument('overrides', 
                       nargs='*',
                       help='Config overrides in format key.subkey=value (e.g., meta.seed=42 data.batch_size=16)')
    return parser


def setup_base_pathing(cfg):
    """Setup base paths for the project"""
    
    path_cfg = OmegaConf.create({
        'path': {
            'src': get_src_base_path(),
            'data': get_data_base_path(),
            'output': get_output_base_path()
        }
    })
    
    cfg = OmegaConf.merge(cfg, path_cfg)
    return cfg


def setup_directory(cfg):
    """Setup all directories and paths for the experiment"""
    resume_exp = cfg.meta.resume_exp
    
    if resume_exp:
        output_dir = cfg.path.output
        cfg = setup_resume_cfg(cfg, resume_exp, output_dir)
        experiment_dir = cfg.path.exp
        checkpoint_dir = cfg.path.ckpt
        plots_dir = cfg.path.plots
        experiment_name = resume_exp
    else:
        run_name = cfg.meta.run_name
        output_base_path = cfg.path.output
        experiment_dir, checkpoint_dir, plots_dir = create_experiment_dir(run_name, output_base_path)
        print(f"Created experiment directory: {experiment_dir}")
        
        save_hyperparameters(cfg, experiment_dir)
        experiment_name = Path(experiment_dir).name
        print(f"Setting WandB run name to: {experiment_name}")
        
        # Use OmegaConf to update paths
        path_updates = OmegaConf.create({
            'path': {
                'exp': experiment_dir,
                'ckpt': checkpoint_dir,
                'plots': plots_dir
            }
        })
        cfg = OmegaConf.merge(cfg, path_updates)
        
        # Update checkpoint and wandb configs if they exist
        if 'checkpoint' in cfg:
            checkpoint_update = OmegaConf.create({'checkpoint': {'dirpath': checkpoint_dir}})
            cfg = OmegaConf.merge(cfg, checkpoint_update)
        if 'wandb' in cfg:
            wandb_update = OmegaConf.create({
                'wandb': {
                    'save_dir': experiment_dir,
                    'name': experiment_name
                }
            })
            cfg = OmegaConf.merge(cfg, wandb_update)
    return cfg

def main():
    """Main function with argument parsing"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Check if we're in test mode and have resume_exp in overrides
    resume_exp_from_overrides = None
    for override in args.overrides:
        if override.startswith('meta.resume_exp='):
            resume_exp_from_overrides = override.split('=', 1)[1]
            break
    
    # Test mode: skip config files entirely, load only from saved experiment
    if args.mode == 'test':
        if not resume_exp_from_overrides:
            raise ValueError(
                "Test mode requires 'meta.resume_exp' to be set to the experiment name.\n"
                "Example: python main.py --mode test --task downstream "
                "meta.resume_exp=probe-self_location-siglip-260120-152532-pe8y"
            )
        
        print("=== TEST MODE: Loading configuration from saved experiment ===")
        
        # Create minimal config with just the resume_exp
        cfg = OmegaConf.create({'meta': {'resume_exp': resume_exp_from_overrides}})
        cfg = setup_base_pathing(cfg)
        
        # Load saved hyperparameters from the experiment
        saved_cfg, exp_path = load_experiment_cfg(resume_exp_from_overrides)
        
        print(f"Loaded saved hyperparameters from: {exp_path / 'hparam.yaml'}")
        cfg = saved_cfg
        
        # Apply command-line overrides (excluding meta.resume_exp which is already set)
        overrides_without_resume = [o for o in args.overrides if not o.startswith('meta.resume_exp=')]
        if overrides_without_resume:
            print(f"Applying command-line overrides: {overrides_without_resume}")
            cfg = apply_cfg_overrides(cfg, overrides_without_resume)
        
        # Ensure paths are set correctly
        cfg = setup_base_pathing(cfg)
        cfg.meta.resume_exp = resume_exp_from_overrides
        
    # Train/dev mode: load config files as usual
    else:
        if args.config is None:
            # Load task-specific config, then apply dev overrides if in dev mode
            train_cfg_path = f'configs/train/{args.task}.yaml'
            
            # 1. Load task-specific train config
            print(f"Loading config from: {train_cfg_path}")
            cfg = load_cfg(train_cfg_path)
            
            # 2. If dev mode, apply dev overrides
            if args.mode == 'dev':
                dev_cfg_path = f'configs/dev/{args.task}.yaml'
                print(f"Applying dev overrides from: {dev_cfg_path}")
                dev_cfg = load_cfg(dev_cfg_path)
                cfg = OmegaConf.merge(cfg, dev_cfg)
        else:
            print(f"Loading config from: {args.config}")
            cfg = load_cfg(args.config)
        
        cfg = apply_cfg_overrides(cfg, args.overrides)
        cfg = setup_base_pathing(cfg)
        
        # Auto-configure task settings for downstream tasks
        if args.task == 'downstream':
            cfg = apply_task_config(cfg, Path(cfg.path.data))
    
    # TODO: Validation need to be adjusted per training mode at the end
    # validate_cfg(cfg)
    
    cfg = setup_directory(cfg)
    
    # Dispatch based on task and mode
    if args.mode == 'test':
        # Test-only mode
        if args.task == 'contrastive':
            test_contrastive(cfg)
        elif args.task == 'downstream':
            test_downstream(cfg)
        else:
            raise ValueError(f"Unknown task: {args.task}")
    else:
        # Train mode (includes validation and testing after training)
        if args.task == 'contrastive':
            train_contrastive(cfg)
        elif args.task == 'downstream':
            train_downstream(cfg)
        else:
            raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
