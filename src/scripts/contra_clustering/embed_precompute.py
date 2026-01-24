import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
import h5py
from omegaconf import OmegaConf
from tqdm import tqdm
import pathlib
import platform

from src.models.contrastive_model import ContrastiveModel
from src.data_module.contrastive import ContrastiveDataModule
from src.utils.env_utils import get_output_base_path, get_data_base_path


if platform.system() == 'Windows':
    import pathlib._local
    pathlib._local.PosixPath = pathlib._local.WindowsPath
    pathlib.PosixPath = pathlib.WindowsPath


def load_checkpoint_model(experiment_name, epoch, cfg):
    output_base = Path(get_output_base_path())
    exp_dir = output_base / experiment_name
    checkpoint_dir = exp_dir / "checkpoint"
    
    ckpt_files = list(checkpoint_dir.glob(f"*-e{epoch:02d}-*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found for epoch {epoch} in {checkpoint_dir}")
    
    checkpoint_path = ckpt_files[0]
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    state_dict = checkpoint['state_dict']
    
    state_dict = ContrastiveModel._strip_orig_mod_prefix(state_dict)
    
    model = ContrastiveModel(cfg)
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    
    return model


def load_pretrained_model(cfg):
    print(f"Loading pretrained model (no finetuning)")
    
    model = ContrastiveModel(cfg)
    model.eval()
    model.cuda()
    
    return model


def precompute_embeddings(experiment_name, epoch, output_filename, use_pretrained=False):
    output_base = Path(get_output_base_path())
    data_base = Path(get_data_base_path())
    exp_dir = output_base / experiment_name
    hparam_path = exp_dir / "hparam.yaml"
    
    if not hparam_path.exists():
        raise FileNotFoundError(f"hparam.yaml not found at {hparam_path}")
    
    cfg = OmegaConf.load(hparam_path)
    cfg.data.partition = "test"
    
    path_cfg = OmegaConf.create({
        'path': {
            'exp': str(exp_dir),
            'ckpt': str(exp_dir / "checkpoint"),
            'plots': str(exp_dir / "plots"),
            'data': str(data_base),
            'src': str(Path(__file__).parent.parent.parent),
            'output': str(output_base)
        },
        'data': {
            'label_path': str(data_base / cfg.data.labels_folder / cfg.data.labels_filename),
            'video_base_path': str(data_base / cfg.data.video_folder)
        }
    })
    cfg = OmegaConf.merge(cfg, path_cfg)
    
    if use_pretrained:
        model = load_pretrained_model(cfg)
    else:
        model = load_checkpoint_model(experiment_name, epoch, cfg)
    
    data_module = ContrastiveDataModule(cfg)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    
    all_embeddings = []
    all_csv_indices = []
    all_agent_ids = []
    
    print(f"Precomputing embeddings for test partition...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            video = batch['video'].cuda()
            
            video_embeddings = model.video_encoder(video)
            projected_embeddings = model.video_projector(video_embeddings)
            
            all_embeddings.append(projected_embeddings.cpu())
            
            # Expand csv_indices to match number of agents per sample
            # batch['original_csv_idx'] is per-sample, but we need per-agent
            agent_counts = batch['agent_counts'].tolist()
            for csv_idx, count in zip(batch['original_csv_idx'], agent_counts):
                all_csv_indices.extend([csv_idx] * count)
            
            all_agent_ids.extend(batch['agent_ids'])
    
    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    
    artifacts_dir = Path("artifacts") / "contra_clustering"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifacts_dir / output_filename
    print(f"Saving embeddings to {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('embeddings', data=embeddings)
        f.create_dataset('csv_indices', data=all_csv_indices)
        
        agent_ids_str = [str(aid) for aid in all_agent_ids]
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('agent_ids', data=agent_ids_str, dtype=dt)
    
    print(f"Saved {len(embeddings)} embeddings with shape {embeddings.shape}")


if __name__ == "__main__":
    experiments = [
        ("main_ui_cover-dinov2-ui-all-260122-035704-my8c", "dinov2-ui_all"),
        ("main_ui_cover-dinov2-ui-minimap-260122-045334-demx", "dinov2-ui_minimap"),
        ("main_ui_cover-dinov2-ui-none-260122-051419-yq1p", "dinov2-ui_none"),
        ("main_ui_cover-siglip2-ui-all-260122-064933-md8t", "siglip2-ui_all"),
        ("main_ui_cover-siglip2-ui-minimap-260122-064933-1z0g", "siglip2-ui_minimap"),
        ("main_ui_cover-siglip2-ui-none-260122-071834-ct2l", "siglip2-ui_none"),
    ]
    
    epoch = 39
    
    artifacts_dir = Path("artifacts") / "contra_clustering"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    for experiment_name, short_name in experiments:
        print(f"\n{'='*80}")
        print(f"Processing: {short_name}")
        print(f"{'='*80}\n")
        
        output_filename_finetuned = f"{short_name}_e{epoch}.h5"
        output_path_finetuned = artifacts_dir / output_filename_finetuned
        
        if output_path_finetuned.exists():
            print(f"[1/2] Finetuned model (epoch {epoch}) - SKIPPING (already exists)")
        else:
            print(f"[1/2] Finetuned model (epoch {epoch})")
            precompute_embeddings(experiment_name, epoch, output_filename_finetuned, use_pretrained=False)
        
        output_filename_pretrained = f"{short_name}_pretrained.h5"
        output_path_pretrained = artifacts_dir / output_filename_pretrained
        
        if output_path_pretrained.exists():
            print(f"\n[2/2] Pretrained model (no finetuning) - SKIPPING (already exists)")
        else:
            print(f"\n[2/2] Pretrained model (no finetuning)")
            precompute_embeddings(experiment_name, None, output_filename_pretrained, use_pretrained=True)
