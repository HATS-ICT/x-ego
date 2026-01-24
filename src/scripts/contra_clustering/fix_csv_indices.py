import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import h5py
import pandas as pd
from omegaconf import OmegaConf
import platform

from src.utils.env_utils import get_output_base_path, get_data_base_path


if platform.system() == 'Windows':
    import pathlib._local
    pathlib._local.PosixPath = pathlib._local.WindowsPath
    pathlib.PosixPath = pathlib.WindowsPath


def fix_csv_indices_in_h5(h5_path, experiment_name):
    """Fix csv_indices in existing h5 file by expanding per agent count."""
    print(f"  Loading existing h5 file: {h5_path}")
    
    # Load existing data
    with h5py.File(h5_path, 'r') as f:
        embeddings = f['embeddings'][:]
        agent_ids = [s.decode('utf-8') if isinstance(s, bytes) else s for s in f['agent_ids'][:]]
    
    print(f"  Found {len(embeddings)} embeddings and {len(agent_ids)} agent_ids")
    
    # Load config to get label path
    output_base = Path(get_output_base_path())
    data_base = Path(get_data_base_path())
    exp_dir = output_base / experiment_name
    hparam_path = exp_dir / "hparam.yaml"
    
    if not hparam_path.exists():
        raise FileNotFoundError(f"hparam.yaml not found at {hparam_path}")
    
    cfg = OmegaConf.load(hparam_path)
    
    # Load CSV directly
    label_path = data_base / cfg.data.labels_folder / cfg.data.labels_filename
    print(f"  Loading CSV: {label_path}")
    # Read teammate IDs as strings to avoid float conversion and precision loss
    dtype_dict = {f'teammate_{i}_id': str for i in range(5)}
    df = pd.read_csv(label_path, dtype=dtype_dict)
    
    # Filter to test partition
    df_test = df[df['partition'] == 'test'].copy()
    print(f"  Filtered to {len(df_test)} test samples")
    
    # Build a mapping from agent_id to csv_idx
    agent_to_csv_idx = {}
    
    print(f"  Building agent_id to csv_idx mapping...")
    for _, row in df_test.iterrows():
        csv_idx = row['idx']
        num_agents = row['num_alive_teammates']
        
        for i in range(num_agents):
            agent_id = str(row[f'teammate_{i}_id']).strip()
            # Skip nan/empty values
            if agent_id and agent_id.lower() != 'nan':
                agent_to_csv_idx[agent_id] = csv_idx
    
    print(f"  Created mapping for {len(agent_to_csv_idx)} agents")
    
    # Now build csv_indices based on the actual agent_ids in the h5 file
    all_csv_indices = []
    missing_count = 0
    
    print(f"  Mapping agent_ids to csv_indices...")
    for agent_id in agent_ids:
        if agent_id in agent_to_csv_idx:
            all_csv_indices.append(agent_to_csv_idx[agent_id])
        else:
            print(f"    WARNING: Agent ID {agent_id} not found in CSV!")
            missing_count += 1
            # Use -1 as placeholder for missing
            all_csv_indices.append(-1)
    
    if missing_count > 0:
        print(f"  WARNING: {missing_count} agent IDs not found in CSV")
    
    print(f"  Reconstructed {len(all_csv_indices)} csv_indices")
    
    # Verify csv_indices length matches embeddings
    if len(all_csv_indices) != len(embeddings):
        print(f"  ERROR: csv_indices length {len(all_csv_indices)} doesn't match embeddings {len(embeddings)}")
        return False
    
    # Save back to h5 file
    print(f"  Updating h5 file with corrected csv_indices...")
    with h5py.File(h5_path, 'r+') as f:
        # Delete old csv_indices dataset
        del f['csv_indices']
        # Create new one with correct data
        f.create_dataset('csv_indices', data=all_csv_indices)
    
    print(f"  ✓ Successfully updated {h5_path}")
    return True


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
    
    for experiment_name, short_name in experiments:
        print(f"\n{'='*80}")
        print(f"Processing: {short_name}")
        print(f"{'='*80}")
        
        # Fix finetuned h5
        finetuned_file = artifacts_dir / f"{short_name}_e{epoch}.h5"
        if finetuned_file.exists():
            print(f"\n[1/2] Fixing finetuned model (epoch {epoch})")
            fix_csv_indices_in_h5(finetuned_file, experiment_name)
        else:
            print(f"\n[1/2] Finetuned file not found: {finetuned_file}")
        
        # Fix pretrained h5
        pretrained_file = artifacts_dir / f"{short_name}_pretrained.h5"
        if pretrained_file.exists():
            print(f"\n[2/2] Fixing pretrained model")
            fix_csv_indices_in_h5(pretrained_file, experiment_name)
        else:
            print(f"\n[2/2] Pretrained file not found: {pretrained_file}")
    
    print(f"\n{'='*80}")
    print("All h5 files updated!")
    print(f"{'='*80}")
