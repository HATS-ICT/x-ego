import torch
from pathlib import Path
import warnings
import argparse

from omegaconf import OmegaConf

# Local imports
from utils.config_utils import load_cfg, apply_cfg_overrides
from utils.experiment_utils import create_experiment_dir, save_hyperparameters, setup_resume_cfg
from train.run_tasks import (
    train_enemy_location_nowcast,
    train_enemy_location_forecast,
    train_teammate_location_forecast,
    test_enemy_location_nowcast,
    test_enemy_location_forecast,
    test_teammate_location_forecast
)
from utils.env_utils import get_src_base_path, get_data_base_path, get_output_base_path


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
    python main.py --mode train --task enemy_location_nowcast meta.seed=123
    python main.py --mode train --task enemy_location_forecast data.batch_size=16 training.max_epochs=20
    python main.py --mode dev --task teammate_location_forecast training.devices=[0,1] data.num_workers=8
  
  Test-only mode (requires resume_exp):
    python main.py --mode test --task enemy_location_nowcast meta.resume_exp=enemy-nowcast-clip-250930-032609-1uqe
    python main.py --mode test --task enemy_location_forecast meta.resume_exp=your-experiment-name
        """
    )
    
    parser.add_argument('--mode', 
                       choices=['train', 'test', 'dev'], 
                       default='train',
                       help='Mode to run: train, test, or dev')
    
    parser.add_argument('--task',
                       choices=['enemy_location_nowcast', 
                                'enemy_location_forecast', 
                                'teammate_location_forecast'],
                       default='enemy_location_nowcast',
                       help='Task to run: enemy_location_nowcast, enemy_location_forecast, or teammate_location_forecast')
    
    
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
    
    if args.config is None:
        # Always start with train config as base
        base_cfg_path = f'configs/train/{args.task}.yaml'
        
        if args.mode == 'dev':
            # Load base train config first
            print(f"Loading base config from: {base_cfg_path}")
            cfg = load_cfg(base_cfg_path)
            
            # Then apply dev overrides
            dev_cfg_path = f'configs/dev/{args.task}.yaml'
            print(f"Applying dev overrides from: {dev_cfg_path}")
            dev_cfg = load_cfg(dev_cfg_path)
            cfg = OmegaConf.merge(cfg, dev_cfg)
        else:  # train or test mode
            print(f"Loading config from: {base_cfg_path}")
            cfg = load_cfg(base_cfg_path)
    else:
        print(f"Loading config from: {args.config}")
        cfg = load_cfg(args.config)
    cfg = apply_cfg_overrides(cfg, args.overrides)
    cfg = setup_base_pathing(cfg)
    
    # Validate test mode requirements
    if args.mode == 'test' and not cfg.meta.resume_exp:
        raise ValueError(
            "Test mode requires 'meta.resume_exp' to be set to the experiment name.\n"
            "Example: python main.py --mode test --task enemy_location_nowcast "
            "meta.resume_exp=enemy-nowcast-clip-250930-032609-1uqe"
        )
    
    # TODO: Validation need to be adjusted per training mode at the end
    # validate_cfg(cfg)
    
    cfg = setup_directory(cfg)
    
    # Dispatch based on task and mode
    if args.mode == 'test':
        # Test-only mode
        if args.task == 'enemy_location_nowcast':
            test_enemy_location_nowcast(cfg)
        elif args.task == 'enemy_location_forecast':
            test_enemy_location_forecast(cfg)
        elif args.task == 'teammate_location_forecast':
            test_teammate_location_forecast(cfg)
        else:
            raise ValueError(f"Unknown task: {args.task}")
    else:
        # Train mode (includes validation and testing after training)
        if args.task == 'enemy_location_nowcast':
            train_enemy_location_nowcast(cfg)
        elif args.task == 'enemy_location_forecast':
            train_enemy_location_forecast(cfg)
        elif args.task == 'teammate_location_forecast':
            train_teammate_location_forecast(cfg)
        else:
            raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
