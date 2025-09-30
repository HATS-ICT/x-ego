import torch
from pathlib import Path
import warnings
import argparse

from omegaconf import OmegaConf

# Local imports
from ctfm.models.utils import load_config, create_experiment_dir, save_hyperparameters, apply_config_overrides, setup_resume_config
from train.train_tasks import (
    train_enemy_location_nowcast,
    train_enemy_location_forecast,
    train_teammate_location_forecast
)
from ctfm.env_utils import get_src_base_path, get_data_base_path, get_output_base_path


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
        description='CTFM Training and Testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Every settings in config file can be overridden. Examples:
  python main.py --mode train --task enemy_location_nowcast meta.seed=123
  python main.py --mode train --task enemy_location_forecast data.batch_size=16 training.max_epochs=20
  python main.py --mode dev --task teammate_location_forecast training.devices=[0,1] data.num_workers=8
  python main.py --mode train --task enemy_location_nowcast data.num_agents=5
  python main.py --mode train --task teammate_location_forecast data.num_agents=5
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


def setup_base_pathing(config):
    """Setup base paths for the project"""
    
    path_config = OmegaConf.create({
        'path': {
            'src': get_src_base_path(),
            'data': get_data_base_path(),
            'output': get_output_base_path()
        }
    })
    
    config = OmegaConf.merge(config, path_config)
    return config


def setup_directory(config):
    """Setup all directories and paths for the experiment"""
    resume_exp = config.meta.resume_exp
    
    if resume_exp:
        output_dir = config.path.output
        config = setup_resume_config(config, resume_exp, output_dir)
        experiment_dir = config.path.exp
        checkpoint_dir = config.path.ckpt
        plots_dir = config.path.plots
        experiment_name = resume_exp
    else:
        run_name = config.meta.run_name
        output_base_path = config.path.output
        experiment_dir, checkpoint_dir, plots_dir = create_experiment_dir(run_name, output_base_path)
        print(f"Created experiment directory: {experiment_dir}")
        
        save_hyperparameters(config, experiment_dir)
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
        config = OmegaConf.merge(config, path_updates)
        
        # Update checkpoint and wandb configs if they exist
        if 'checkpoint' in config:
            checkpoint_update = OmegaConf.create({'checkpoint': {'dirpath': checkpoint_dir}})
            config = OmegaConf.merge(config, checkpoint_update)
        if 'wandb' in config:
            wandb_update = OmegaConf.create({
                'wandb': {
                    'save_dir': experiment_dir,
                    'name': experiment_name
                }
            })
            config = OmegaConf.merge(config, wandb_update)
    return config

def main():
    """Main function with argument parsing"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    if args.config is None:
        # Always start with train config as base
        base_config_path = f'configs/train/{args.task}.yaml'
        
        if args.mode == 'dev':
            # Load base train config first
            print(f"Loading base config from: {base_config_path}")
            config = load_config(base_config_path)
            
            # Then apply dev overrides
            dev_config_path = f'configs/dev/{args.task}.yaml'
            print(f"Applying dev overrides from: {dev_config_path}")
            dev_config = load_config(dev_config_path)
            config = OmegaConf.merge(config, dev_config)
        else:  # train or test mode
            print(f"Loading config from: {base_config_path}")
            config = load_config(base_config_path)
    else:
        print(f"Loading config from: {args.config}")
        config = load_config(args.config)
    config = apply_config_overrides(config, args.overrides)
    config = setup_base_pathing(config)
    
    # TODO: Validation need to be adjusted per training mode at the end
    # validate_config(config)
    
    config = setup_directory(config)
    
    # Dispatch based on task and mode
    if args.task == 'enemy_location_nowcast':
        train_enemy_location_nowcast(config)
    elif args.task == 'enemy_location_forecast':
        train_enemy_location_forecast(config)
    elif args.task == 'teammate_location_forecast':
        train_teammate_location_forecast(config)
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
