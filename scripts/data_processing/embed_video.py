"""
Pre-compute video embeddings for all video encoders.

This script processes all videos in the dataset and pre-computes embeddings using
different video encoders (clip, dinov2, siglip, vivit, videomae, vjepa2).

Embeddings are saved to HDF5 files in data/video_544x306_30fps_embed/ for fast
loading during training, avoiding redundant video encoding computations.

Output structure:
    {csv_filename}_{encoder_name}.h5:
        /{csv_idx}/{agent_position} -> [embed_dim] embedding tensor
        
        where:
        - csv_filename: Name of the CSV file (without .csv extension)
        - encoder_name: Name of the encoder (clip, dinov2, etc.)
        - csv_idx: Row index from the original CSV
        - agent_position: Agent position (0-4) for agents in that CSV row
        - embed_dim: Embedding dimension (depends on encoder)

Example output files:
    - teammate_location_nowcast_s1s_l5s_half_clip.h5
    - enemy_location_nowcast_s1s_l5s_dinov2.h5
"""

import sys
from pathlib import Path

# Add project root to path (must be before local imports)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Standard library and third-party imports
import argparse
import h5py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

# Local imports (dataset classes imported dynamically based on task)
from models.video_encoder import VideoEncoder
from utils.config_utils import load_cfg
from utils.env_utils import get_data_base_path


# Encoder configurations
ENCODER_CONFIGS = {
    'clip': {
        'model_type': 'clip',
        'freeze_backbone': True,
    },
    'dinov2': {
        'model_type': 'dinov2',
        'freeze_backbone': True,
    },
    'siglip': {
        'model_type': 'siglip',
        'freeze_backbone': True,
    },
    'vivit': {
        'model_type': 'vivit',
        'freeze_backbone': True,
    },
    'videomae': {
        'model_type': 'videomae',
        'freeze_backbone': True,
    },
    'vjepa2': {
        'model_type': 'vjepa2',
        'freeze_backbone': True,
    },
}


class VideoEmbeddingExtractor:
    """Extract and save video embeddings for a specific encoder."""
    
    def __init__(self, encoder_name, cfg, device='cuda'):
        """
        Initialize the embedding extractor.
        
        Args:
            encoder_name: Name of encoder (clip, dinov2, siglip, vivit, videomae, vjepa2)
            cfg: Configuration dictionary
            device: Device to run on (cuda or cpu)
        """
        self.encoder_name = encoder_name
        self.cfg = cfg
        self.device = device
        
        # Create encoder config
        encoder_cfg = OmegaConf.create(ENCODER_CONFIGS[encoder_name])
        
        # Initialize video encoder
        print(f"Initializing {encoder_name} encoder...")
        self.video_encoder = VideoEncoder(encoder_cfg).to(device)
        self.video_encoder.eval()
        
        self.embed_dim = self.video_encoder.embed_dim
        print(f"Encoder embedding dimension: {self.embed_dim}")
        
        # Output directory and file
        data_base_path = Path(get_data_base_path())
        self.output_dir = data_base_path / "video_544x306_30fps_embed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename from CSV name + encoder name
        # e.g., teammate_location_nowcast_s1s_l5s_half.csv -> teammate_location_nowcast_s1s_l5s_half_clip.h5
        csv_filename = cfg.data.labels_filename
        csv_name_without_ext = Path(csv_filename).stem  # Remove .csv extension
        self.output_file = self.output_dir / f"{csv_name_without_ext}_{encoder_name}.h5"
        print(f"Output file: {self.output_file}")
    
    def extract_embeddings(self, dataset, batch_size=8, max_steps=None, num_workers=8):
        """
        Extract embeddings for all samples in the dataset.
        
        Args:
            dataset: Dataset to extract embeddings from
            batch_size: Batch size for processing
            max_steps: Maximum number of batches to process (None for all)
            num_workers: Number of workers for data loading
        """
        print(f"\nExtracting embeddings for {len(dataset)} samples...")
        print(f"Using batch size: {batch_size}")
        if max_steps is not None:
            print(f"Maximum steps: {max_steps} (processing ~{max_steps * batch_size} samples)")
        
        # Create dataloader with actual batching
        # Since we're using 5 agents and no time jitter, all samples have same shape
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
        
        # Open h5 file for writing
        with h5py.File(self.output_file, 'w') as h5f:
            # Add metadata
            h5f.attrs['encoder_name'] = self.encoder_name
            h5f.attrs['embed_dim'] = self.embed_dim
            h5f.attrs['num_samples'] = len(dataset)
            
            # Track sample index across batches
            sample_idx = 0
            step_count = 0
            
            # Process all samples
            with torch.no_grad():
                pbar = tqdm(dataloader, desc=f"Processing {self.encoder_name}")
                if max_steps is not None:
                    pbar.total = min(len(dataloader), max_steps)
                
                for batch in pbar:
                    # Get video data: [B, A, T, C, H, W] where B is batch size, A is number of agents
                    video = batch['video']  # [B, A, T, C, H, W]
                    B, A, T, C, H, W = video.shape
                    
                    # Get original CSV indices from batch
                    original_csv_indices = batch['original_csv_idx']  # [B]
                    
                    # Move to device
                    video = video.to(self.device)
                    
                    # Encode all agents' videos
                    # Reshape to [B*A, T, C, H, W] for encoder
                    video_reshaped = video.view(B * A, T, C, H, W)
                    agent_embeddings = self.video_encoder(video_reshaped)  # [B*A, embed_dim]
                    
                    # Reshape back to [B, A, embed_dim]
                    agent_embeddings = agent_embeddings.view(B, A, -1)
                    
                    # Move to CPU and convert to numpy
                    agent_embeddings = agent_embeddings.cpu().numpy()  # [B, A, embed_dim]
                    
                    # Store embeddings for each sample in the batch
                    for batch_idx in range(B):
                        # Use original CSV index for H5 group name
                        original_idx = int(original_csv_indices[batch_idx])
                        sample_group = h5f.create_group(str(original_idx))
                        
                        # Store embedding for each agent
                        for agent_pos in range(A):
                            agent_embedding = agent_embeddings[batch_idx, agent_pos]  # [embed_dim]
                            sample_group.create_dataset(
                                str(agent_pos),
                                data=agent_embedding,
                                dtype='float32',
                                compression='gzip',
                                compression_opts=4
                            )
                        
                        # Store metadata for this sample
                        sample_group.attrs['csv_idx'] = original_idx
                        sample_group.attrs['num_agents'] = A
                        
                        sample_idx += 1
                    
                    step_count += 1
                    
                    # Check if we've reached max steps
                    if max_steps is not None and step_count >= max_steps:
                        print(f"\nReached max steps ({max_steps}). Processed {sample_idx} samples.")
                        break
        
        print(f"\nSuccessfully saved embeddings to: {self.output_file}")
        print(f"Total samples: {len(dataset)}")
        print(f"Embedding shape: [{self.embed_dim}]")


def create_embedding_dataset_config(base_cfg):
    """
    Create a configuration for embedding extraction.
    
    Uses the base config but modifies settings to ensure we process all data
    without augmentation.
    """
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    
    # Use all data (train + val + test)
    cfg.data.partition = 'all'
    
    # Disable any time jitter for consistent embeddings
    cfg.data.time_jitter_max_seconds = 0.0
    
    # Use full 5 agents per team
    cfg.data.num_pov_agents = 5
    
    # CRITICAL: Disable pre-computed embeddings - we need to extract from raw videos!
    cfg.data.use_precomputed_embeddings = False
    
    # Build label_path from config components (same as BaseDataModule._build_label_path)
    data_path = Path(cfg.path.data)
    label_path = data_path / cfg.data.labels_folder / cfg.data.labels_filename
    cfg.data.label_path = str(label_path)
    
    # Validate that label path exists
    if not label_path.exists():
        raise FileNotFoundError(f"Label CSV file not found: {label_path}")
    
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description='Pre-compute video embeddings for all encoders',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--encoders',
        nargs='+',
        default=['clip', 'dinov2', 'siglip', 'vivit', 'videomae', 'vjepa2'],
        choices=['clip', 'dinov2', 'siglip', 'vivit', 'videomae', 'vjepa2'],
        help='Which encoders to use (default: all)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for processing agent videos (default: 8)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on (default: cuda)'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        choices=['teammate_location_nowcast', 'teammate_location_forecast', 
                 'enemy_location_nowcast', 'enemy_location_forecast',
                 'teammate_opponent_traj_prediction'],
        default='teammate_location_nowcast',
        help='Task to extract embeddings for (determines which config/CSV to use)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (auto-determined from --task if not specified)'
    )
    
    parser.add_argument(
        '--max-steps',
        type=int,
        default=None,
        help='Maximum number of batches to process (default: None, process all)'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Number of workers for data loading (default: 8)'
    )
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print("=" * 80)
    print("Video Embedding Pre-computation")
    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"Encoders: {args.encoders}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    if args.max_steps is not None:
        print(f"Max steps: {args.max_steps} (~{args.max_steps * args.batch_size} samples)")
    print("=" * 80)
    
    # Determine config file
    if args.config is None:
        config_path = f'configs/train/{args.task}.yaml'
        print(f"\nAuto-determined config from task: {config_path}")
    else:
        config_path = args.config
        print(f"\nUsing specified config: {config_path}")
    
    # Load base configuration
    global_cfg = load_cfg('configs/global.yaml')
    task_cfg = load_cfg(config_path)
    base_cfg = OmegaConf.merge(global_cfg, task_cfg)
    
    # Setup base paths FIRST (needed by create_embedding_dataset_config)
    data_base_path = Path(get_data_base_path())
    base_cfg.path = OmegaConf.create({
        'data': str(data_base_path),
        'exp': str(data_base_path / 'video_544x306_30fps_embed')  # Use embed dir as temp output dir
    })
    
    # Create embedding-specific config
    cfg = create_embedding_dataset_config(base_cfg)
    
    print(f"Data partition: {cfg.data.partition}")
    print(f"Number of agents: {cfg.data.num_pov_agents}")
    print(f"Time jitter disabled: {cfg.data.time_jitter_max_seconds}")
    print(f"Labels file: {cfg.data.labels_filename}")
    
    # Create dataset once (reuse for all encoders)
    print("\nInitializing dataset...")
    
    # Import appropriate dataset class based on task
    if args.task == 'teammate_location_nowcast':
        from dataset.teammate_location_nowcast import TeammateLocationNowcastDataset
        dataset = TeammateLocationNowcastDataset(cfg)
    elif args.task == 'teammate_location_forecast':
        from dataset.teammate_location_forecast import TeammateLocationForecastDataset
        dataset = TeammateLocationForecastDataset(cfg)
    elif args.task == 'enemy_location_nowcast':
        from dataset.enemy_location_nowcast import EnemyLocationNowcastDataset
        dataset = EnemyLocationNowcastDataset(cfg)
    elif args.task == 'enemy_location_forecast':
        from dataset.enemy_location_forecast import EnemyLocationForecastDataset
        dataset = EnemyLocationForecastDataset(cfg)
    elif args.task == 'teammate_opponent_traj_prediction':
        from dataset.teammate_opponent_traj_prediction import TeammateOpponentTrajPredictionDataset
        dataset = TeammateOpponentTrajPredictionDataset(cfg)
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    print(f"Dataset size: {len(dataset)}")
    
    # Process each encoder
    for encoder_name in args.encoders:
        print("\n" + "=" * 80)
        print(f"Processing encoder: {encoder_name.upper()}")
        print("=" * 80)
        
        try:
            # Create extractor
            extractor = VideoEmbeddingExtractor(
                encoder_name=encoder_name,
                cfg=cfg,
                device=args.device
            )
            
            # Extract and save embeddings
            extractor.extract_embeddings(
                dataset, 
                batch_size=args.batch_size, 
                max_steps=args.max_steps,
                num_workers=args.num_workers
            )
            
            # Clear CUDA cache
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\nError processing {encoder_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("Embedding pre-computation completed!")
    print("=" * 80)
    
    # Print summary
    data_base_path = Path(get_data_base_path())
    output_dir = data_base_path / "video_544x306_30fps_embed"
    
    print(f"\nSaved embeddings to: {output_dir}")
    print("\nFiles created:")
    csv_name = Path(cfg.data.labels_filename).stem
    for encoder_name in args.encoders:
        output_file = output_dir / f"{csv_name}_{encoder_name}.h5"
        if output_file.exists():
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"  - {csv_name}_{encoder_name}.h5 ({file_size_mb:.2f} MB)")
    
    print("\nTo use pre-computed embeddings during training:")
    print(f"  python main.py --mode train --task {args.task} \\")
    print("    data.use_precomputed_embeddings=true \\")
    print("    data.precomputed_embeddings_encoder=<encoder_name>")


if __name__ == "__main__":
    main()

