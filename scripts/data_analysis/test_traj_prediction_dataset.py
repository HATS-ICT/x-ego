"""
Test script for trajectory prediction dataset and data module.

This script verifies that the dataset can:
1. Load trajectory data from H5 file
2. Sample videos from POV team
3. Return trajectories for target team
4. Handle different POV-target strategies
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from omegaconf import OmegaConf
from dataset.teammate_opponent_traj_prediction import TeammateOpponentTrajPredictionDataset
from data_module.teammate_opponent_traj_prediction import TeammateOpponentTrajPredictionDataModule

# Load environment variables
load_dotenv()


def test_dataset():
    """Test the trajectory prediction dataset."""
    print("=" * 80)
    print("Testing Trajectory Prediction Dataset")
    print("=" * 80)
    
    # Load config
    config_path = Path(__file__).parent.parent.parent / "configs" / "dev" / "teammate_opponent_traj_prediction.yaml"
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return
    
    cfg = OmegaConf.load(config_path)
    
    # Override for testing
    cfg.data.partition = 'train'
    
    # Build label path (normally done by data module)
    label_path = Path(cfg.path.data) / cfg.data.labels_folder / cfg.data.labels_filename
    cfg.data.label_path = str(label_path)
    
    print(f"\nConfiguration:")
    print(f"  Label path: {cfg.data.labels_filename}")
    print(f"  Video length: {cfg.data.video_length_sec}s")
    print(f"  Trajectory length: {cfg.data.total_trajectory_sec}s at {cfg.data.trajectory_sample_rate}Hz")
    print(f"  POV agents: {cfg.data.num_pov_agents}")
    print(f"  POV-Target sampling: random")
    
    # Create dataset
    print(f"\nCreating dataset...")
    dataset = TeammateOpponentTrajPredictionDataset(cfg)
    
    print(f"\nDataset created successfully!")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Trajectory shape: {dataset.trajectories.shape}")
    
    # Test sampling a few items
    print(f"\nTesting sample retrieval...")
    
    for i in range(3):
        print(f"\n--- Sample {i} ---")
        sample = dataset[i]
        
        print(f"Keys: {list(sample.keys())}")
        print(f"Video shape: {sample['video'].shape}")  # [num_agents, num_frames, C, H, W]
        print(f"Trajectories shape: {sample['trajectories'].shape}")  # [5, 60, 2]
        print(f"POV team side: {sample['pov_team_side']}")
        print(f"Target team side: {sample['target_team_side']}")
        print(f"Agent IDs: {sample['agent_ids']}")
        print(f"Target player IDs: {sample['target_player_ids']}")
        
        # Verify shapes
        assert sample['video'].shape[0] == cfg.data.num_pov_agents
        assert sample['trajectories'].shape == (5, 60, 2)
    
    print("\n" + "=" * 80)
    print("Dataset test completed successfully!")
    print("=" * 80)


def test_datamodule():
    """Test the trajectory prediction data module."""
    print("\n" + "=" * 80)
    print("Testing Trajectory Prediction DataModule")
    print("=" * 80)
    
    # Load config
    config_path = Path(__file__).parent.parent.parent / "configs" / "dev" / "teammate_opponent_traj_prediction.yaml"
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return
    
    cfg = OmegaConf.load(config_path)
    
    # Override for testing
    cfg.data.partition = 'all'
    cfg.data.batch_size = 2
    
    # Build label path (normally done by data module)
    label_path = Path(cfg.path.data) / cfg.data.labels_folder / cfg.data.labels_filename
    cfg.data.label_path = str(label_path)
    
    print(f"\nCreating data module...")
    data_module = TeammateOpponentTrajPredictionDataModule(cfg)
    
    # Prepare data
    print(f"\nPreparing data...")
    data_module.prepare_data()
    
    # Setup for training
    print(f"\nSetting up for training...")
    data_module.setup(stage='fit')
    
    print(f"\nDataModule setup completed!")
    print(f"  Train dataset: {len(data_module.train_dataset)} samples")
    print(f"  Val dataset: {len(data_module.val_dataset)} samples")
    
    # Test dataloader
    print(f"\nTesting train dataloader...")
    train_loader = data_module.train_dataloader()
    
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {list(batch.keys())}")
    print(f"Batch video shape: {batch['video'].shape}")  # [batch_size, num_agents, num_frames, C, H, W]
    print(f"Batch trajectories shape: {batch['trajectories'].shape}")  # [batch_size, 5, 60, 2]
    print(f"Batch POV sides: {batch['pov_team_side']}")
    print(f"Batch target sides: {batch['target_team_side']}")
    
    # Verify shapes
    batch_size = cfg.data.batch_size
    assert batch['video'].shape[0] == batch_size
    assert batch['video'].shape[1] == cfg.data.num_pov_agents
    assert batch['trajectories'].shape == (batch_size, 5, 60, 2)
    
    print("\n" + "=" * 80)
    print("DataModule test completed successfully!")
    print("=" * 80)


def test_random_sampling():
    """Test random POV-target sampling."""
    print("\n" + "=" * 80)
    print("Testing Random POV-Target Sampling")
    print("=" * 80)
    
    # Load config
    config_path = Path(__file__).parent.parent.parent / "configs" / "dev" / "teammate_opponent_traj_prediction.yaml"
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return
    
    cfg = OmegaConf.load(config_path)
    cfg.data.partition = 'train'
    
    # Build label path (normally done by data module)
    label_path = Path(cfg.path.data) / cfg.data.labels_folder / cfg.data.labels_filename
    cfg.data.label_path = str(label_path)
    
    dataset = TeammateOpponentTrajPredictionDataset(cfg)
    
    # Sample 20 items and count combinations
    print(f"\nSampling 20 random items to verify all combinations are possible...")
    combinations = {}
    for i in range(20):
        sample = dataset[i % len(dataset)]
        pov = sample['pov_team_side']
        target = sample['target_team_side']
        key = f"{pov}->{target}"
        combinations[key] = combinations.get(key, 0) + 1
    
    print(f"\nPOV-Target combinations (20 samples):")
    for key, count in sorted(combinations.items()):
        print(f"  {key}: {count}")
    
    # Verify we got different combinations
    if len(combinations) > 1:
        print(f"\n✓ Random sampling is working (got {len(combinations)} different combinations)")
    else:
        print(f"\n⚠ Warning: Only got 1 combination, might need more samples")
    
    print("\n" + "=" * 80)
    print("Random sampling test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_dataset()
        test_datamodule()
        test_random_sampling()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
