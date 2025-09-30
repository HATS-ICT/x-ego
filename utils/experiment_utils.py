"""
Experiment management utilities for X-EGO project.
Handles experiment directories, checkpoints, and resuming training.
"""

import random
import string
import json
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf

from .env_utils import get_output_base_path


def create_experiment_dir(exp_name_prefix=None, output_base_path=None):
    """Create experiment directory with timestamp and random hash"""
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    random_hash = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    
    if exp_name_prefix:
        experiment_name = f"{exp_name_prefix}-{timestamp}-{random_hash}"
    else:
        experiment_name = f"{timestamp}-{random_hash}"
    
    if output_base_path is None:
        output_base_path = get_output_base_path()
    
    output_dir = Path(output_base_path)
    experiment_dir = output_dir / experiment_name
    checkpoint_dir = experiment_dir / "checkpoint"
    plots_dir = experiment_dir / "plots"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    return str(experiment_dir), str(checkpoint_dir), str(plots_dir)


def save_hyperparameters(config, experiment_dir):
    """Save hyperparameters to hparam.yaml"""
    experiment_path = Path(experiment_dir)
    hparam_path = experiment_path / "hparam.yaml"
    OmegaConf.save(config, hparam_path)
    print(f"Saved hyperparameters to: {hparam_path}")
    return hparam_path


def find_experiment_directory(exp_dir: str) -> Path:
    """Find experiment directory, handling both full paths and experiment names."""
    exp_path = Path(exp_dir)
    
    # If it's already an absolute path and exists, use it
    if exp_path.is_absolute() and exp_path.exists():
        return exp_path
    
    # If it's a relative path and exists, use it
    if exp_path.exists():
        return exp_path.resolve()
    
    # Otherwise, look in the output directory
    output_dir = Path(get_output_base_path())
    candidate = output_dir / exp_dir
    
    if candidate.exists():
        return candidate
    
    raise FileNotFoundError(f"Experiment directory not found: {candidate}")


def load_experiment_config(exp_dir: str):
    """Load configuration from experiment directory"""
    exp_path = find_experiment_directory(exp_dir)
    print(f"Found experiment directory: {exp_path}")
    
    # Load hyperparameters
    hparams_path = exp_path / "hparam.yaml"
    if not hparams_path.exists():
        raise FileNotFoundError(f"Hyperparameters file not found: {hparams_path}")
    
    config = OmegaConf.load(hparams_path)
    
    return config, exp_path


def find_checkpoint(exp_path: Path, prefer_best: bool = True, ckpt_name: str = None):
    """Find the best or last checkpoint in experiment directory"""
    checkpoint_dir = exp_path / "checkpoint"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # If specific checkpoint is requested
    if ckpt_name:
        # Check if it's an absolute path
        if Path(ckpt_name).is_absolute():
            ckpt_path = Path(ckpt_name)
            if ckpt_path.exists():
                return str(ckpt_path)
            else:
                raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        else:
            # Look for checkpoint by name in checkpoint directory
            ckpt_path = checkpoint_dir / ckpt_name
            if ckpt_path.exists():
                return str(ckpt_path)
            # Also try adding .ckpt extension if not present
            if not ckpt_name.endswith('.ckpt'):
                ckpt_path = checkpoint_dir / f"{ckpt_name}.ckpt"
                if ckpt_path.exists():
                    return str(ckpt_path)
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_name} in {checkpoint_dir}")
    
    # Look for best checkpoint first if preferred
    if prefer_best:
        best_ckpt = checkpoint_dir / "best.ckpt"
        if best_ckpt.exists():
            return str(best_ckpt)
    
    # Look for last checkpoint
    last_ckpt = checkpoint_dir / "last.ckpt"
    if last_ckpt.exists():
        return str(last_ckpt)
    
    # Look for any .ckpt files
    ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
    if ckpt_files:
        # Return the most recent one
        latest_ckpt = max(ckpt_files, key=lambda x: x.stat().st_mtime)
        return str(latest_ckpt)
    
    raise FileNotFoundError(f"No checkpoint files found in: {checkpoint_dir}")


def find_resume_checkpoint(resume_exp, output_dir="output"):
    """Find the most recent checkpoint from a given experiment
    
    Args:
        resume_exp (str): Experiment name (e.g., 'mean_pooling_baseline-20250827-032048-qoin')
        output_dir (str): Output directory containing experiments
        
    Returns:
        tuple: (checkpoint_path, experiment_dir) or (None, None) if not found
    """
    if not resume_exp:
        return None, None
    
    # Construct experiment directory path using pathlib
    output_path = Path(output_dir)
    experiment_dir = output_path / resume_exp
    
    if not experiment_dir.exists():
        raise ValueError(f"Resume experiment directory not found: {experiment_dir}")
    
    checkpoint_dir = experiment_dir / "checkpoint"
    
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Look for checkpoints in order of preference:
    # 1. last.ckpt (most recent checkpoint)
    # 2. Best checkpoint based on filename pattern
    # 3. Any .ckpt file
    
    # Check for last.ckpt first
    last_ckpt = checkpoint_dir / "last.ckpt"
    if last_ckpt.exists():
        print(f"Found last checkpoint: {last_ckpt}")
        return last_ckpt, experiment_dir
    
    # Look for other checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in: {checkpoint_dir}")
    
    # Sort by modification time (most recent first)
    checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest_ckpt = checkpoint_files[0]
    
    print(f"Found latest checkpoint: {latest_ckpt}")
    return latest_ckpt, experiment_dir


def setup_resume_config(config, resume_exp, output_dir="output"):
    """Setup configuration for resuming from an existing experiment
    
    Args:
        config (dict): Current configuration
        resume_exp (str): Experiment name to resume from
        output_dir (str): Output directory containing experiments
        
    Returns:
        tuple: (updated_config, checkpoint_path)
    """
    if not resume_exp:
        return config, None
    
    print(f"=== RESUMING FROM EXPERIMENT: {resume_exp} ===")
    
    checkpoint_path, experiment_dir = find_resume_checkpoint(resume_exp, output_dir)
    
    if checkpoint_path is None:
        raise ValueError(f"Could not find checkpoint for experiment: {resume_exp}")
    
    experiment_path = Path(experiment_dir)
    hparam_path = experiment_path / "hparam.yaml"
    if hparam_path.exists():
        print(f"Loading original hyperparameters from: {hparam_path}")
        original_config = OmegaConf.load(hparam_path)
        
        # Use OmegaConf to merge configurations
        config = OmegaConf.merge(original_config, config)
    else:
        print(f"Warning: Could not find hparam.yaml at {hparam_path}, using current config")
    
    checkpoint_dir = experiment_path / "checkpoint"
    plots_dir = experiment_path / "plots"
    
    if 'path' not in config:
        config['path'] = {}
    
    config['path']['exp'] = experiment_dir
    config['path']['ckpt'] = checkpoint_dir
    config['path']['plots'] = plots_dir
    
    if 'checkpoint' in config:
        config['checkpoint']['dirpath'] = str(checkpoint_dir)
        config['checkpoint']['resume_checkpoint_path'] = str(checkpoint_path)
    if 'wandb' in config:
        config['wandb']['save_dir'] = str(experiment_dir)
        config['wandb']['name'] = resume_exp
    
    print(f"Will resume from checkpoint: {checkpoint_path}")
    print(f"Continuing in experiment directory: {experiment_dir}")
    
    return config


def save_evaluation_results(results: dict, exp_path: Path, eval_type: str) -> Path:
    """Save evaluation results to JSON file"""
    from .serialization_utils import convert_numpy_to_python
    
    # Create results directory
    results_dir = exp_path / "evaluation_results"
    results_dir.mkdir(exist_ok=True)
    
    # Convert numpy arrays to Python lists for JSON serialization
    results_json_compatible = convert_numpy_to_python(results)
    
    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"{eval_type}_eval_{timestamp}.json"
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(results_json_compatible, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    return results_file
