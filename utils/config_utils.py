"""
Configuration utilities for X-EGO project.
Handles loading, validation, and manipulation of training configurations.
"""

from pathlib import Path
from omegaconf import OmegaConf


def load_cfg(cfg_path):
    """Load configuration from YAML file using OmegaConf"""
    cfg = OmegaConf.load(cfg_path)
    return cfg


def validate_cfg(cfg):
    """Validate that cfg has all required keys"""
    required_keys = {
        'data': ['dir', 'data_path_csv_filename', 'batch_size', 'num_workers', 'persistent_workers', 'pin_mem',
                 'target_fps', 'video_column_name', 'label_column_name',
                 'audio_sample_rate', 'train_split', 'val_split', 'test_split', 'video_processor_model', 'audio_processor_model'],
        'model': ['encoder'],
        'training': ['max_epochs', 'accelerator', 'devices', 'precision', 'gradient_clip_val', 'accumulate_grad_batches', 
                    'val_check_interval', 'check_val_every_n_epoch', 'enable_checkpointing', 'enable_progress_bar', 'enable_model_summary', 'contrastive'],
        'optimization': ['lr', 'weight_decay', 'epochs', 'final_lr'],
        'meta': ['seed']  # resume_exp is optional
    }
    
    # Optional sections that have required keys if they exist
    optional_sections = {
        'wandb': ['enabled', 'project', 'tags', 'notes', 'save_dir'],
        'checkpoint': ['dirpath', 'filename', 'monitor', 'mode', 'save_top_k', 'save_last', 'auto_insert_metric_name'],
        'early_stopping': ['monitor', 'patience', 'mode']
    }
    
    # Required nested sections
    nested_required_keys = {
        'training.contrastive': ['style', 'project_to_shared_dim', 'do_projection', 'do_normalization', 'logit_scale_init', 'logit_bias_init'],
        'model.encoder': ['video', 'audio'],
        'model.encoder.video': ['backbone', 'freeze_backbone', 'from_pretrained'],
        'model.encoder.audio': ['backbone', 'freeze_backbone', 'from_pretrained']
    }
    
    # Check required sections
    for section, keys in required_keys.items():
        if section not in cfg:
            raise ValueError(f"Missing required cfg section: {section}")
        
        for key in keys:
            if key not in cfg[section]:
                raise ValueError(f"Missing required cfg key: {section}.{key}")
    
    # Check optional sections (if they exist, they must have all required keys)
    for section, keys in optional_sections.items():
        if section in cfg:
            for key in keys:
                if key not in cfg[section]:
                    raise ValueError(f"Missing required cfg key in optional section: {section}.{key}")
    
    # Check nested required sections
    for section_path, keys in nested_required_keys.items():
        parts = section_path.split('.')
        current = cfg
        
        # Navigate to the nested section
        try:
            for part in parts:
                current = current[part]
        except KeyError:
            raise ValueError(f"Missing required nested cfg section: {section_path}")
        
        # Check required keys in the nested section
        for key in keys:
            if key not in current:
                raise ValueError(f"Missing required cfg key: {section_path}.{key}")
    
    # Validate contrastive style is supported
    if 'training' in cfg and 'contrastive' in cfg.training:
        contrastive_style = cfg.training.contrastive.style
        if contrastive_style not in ['clip', 'siglip']:
            raise ValueError(f"Invalid contrastive style: {contrastive_style}. Only 'clip' and 'siglip' are supported.")
    
    print("✓ Config validation passed")


def apply_cfg_overrides(cfg, overrides):
    """Apply cfg overrides using OmegaConf
    
    Args:
        cfg (DictConfig): Original OmegaConf configuration
        overrides (list): List of override strings like ['meta.seed=42', 'data.batch_size=16']
        
    Returns:
        DictConfig: Updated OmegaConf configuration
    """
    if not overrides:
        return cfg
    
    print(f"Applying {len(overrides)} cfg overrides:")
    
    for override_str in overrides:
        try:
            # OmegaConf can parse the override string directly
            override_cfg = OmegaConf.from_dotlist([override_str])
            
            # Show what's being changed
            key_path = override_str.split('=')[0]
            old_value = OmegaConf.select(cfg, key_path, default='<not set>')
            new_value = override_str.split('=')[1]
            print(f"  {key_path}: {old_value} -> {new_value}")
            
            # Merge the override into the main cfg
            cfg = OmegaConf.merge(cfg, override_cfg)
            
        except Exception as e:
            print(f"Warning: Failed to apply override '{override_str}': {e}")
            continue
    
    return cfg
