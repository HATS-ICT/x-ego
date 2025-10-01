"""
Compute class weights for enemy location nowcast task.

This script calculates class weights to handle class imbalance in multi-label classification.
The weights can be saved and loaded during training to balance the loss function.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))


def extract_locations_per_partition(df: pd.DataFrame, partition: str) -> list:
    """
    Extract all location labels from a specific partition.
    
    Args:
        df: DataFrame containing the label data
        partition: Partition name (train/val/test)
        
    Returns:
        List of all location labels
    """
    partition_df = df[df['partition'] == partition] if partition != 'all' else df
    
    all_locations = []
    for i in range(10):
        place_col = f'player_{i}_place'
        if place_col in partition_df.columns:
            locations = partition_df[place_col].tolist()
            locations = [loc for loc in locations if loc and str(loc).strip() and str(loc) != 'nan']
            all_locations.extend(locations)
    
    return all_locations


def compute_class_weights(location_counts: Counter, method: str = 'inverse') -> dict:
    """
    Compute class weights for balancing.
    
    Args:
        location_counts: Counter object with location frequencies
        method: Weight computation method
            - 'inverse': weight = 1 / count
            - 'inverse_sqrt': weight = 1 / sqrt(count)
            - 'effective_num': Effective number of samples method
            
    Returns:
        Dictionary mapping location to weight
    """
    locations = sorted(location_counts.keys())
    counts = np.array([location_counts[loc] for loc in locations])
    
    if method == 'inverse':
        # Inverse frequency weighting
        weights = 1.0 / counts
    elif method == 'inverse_sqrt':
        # Square root of inverse frequency (less aggressive)
        weights = 1.0 / np.sqrt(counts)
    elif method == 'effective_num':
        # Effective number of samples method
        # From "Class-Balanced Loss Based on Effective Number of Samples"
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize weights to sum to number of classes
    weights = weights / weights.sum() * len(weights)
    
    # Create weight dictionary
    weight_dict = {loc: float(weight) for loc, weight in zip(locations, weights)}
    
    return weight_dict


def compute_pos_weight(location_counts: Counter, total_samples: int) -> dict:
    """
    Compute pos_weight for BCEWithLogitsLoss.
    
    pos_weight[c] = (num_negative_samples[c]) / (num_positive_samples[c])
    
    This is useful for BCEWithLogitsLoss which applies different weights
    to positive and negative samples.
    
    Args:
        location_counts: Counter with location frequencies (positive samples)
        total_samples: Total number of samples in dataset
        
    Returns:
        Dictionary mapping location to pos_weight
    """
    locations = sorted(location_counts.keys())
    
    pos_weights = {}
    for loc in locations:
        pos_count = location_counts[loc]
        neg_count = total_samples * 10 - pos_count  # 10 players per sample
        pos_weights[loc] = float(neg_count / pos_count) if pos_count > 0 else 1.0
    
    return pos_weights


def print_weight_statistics(weights: dict, location_counts: Counter):
    """Print statistics about computed weights."""
    print(f"\n{'='*80}")
    print("CLASS WEIGHT STATISTICS")
    print(f"{'='*80}")
    
    # Sort by weight (descending)
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 10 Highest Weights (Rare Classes):")
    print(f"{'Location':<30} {'Count':>10} {'Weight':>12}")
    print('-' * 80)
    for loc, weight in sorted_weights[:10]:
        count = location_counts[loc]
        print(f"{loc:<30} {count:>10} {weight:>12.4f}")
    
    print(f"\nTop 10 Lowest Weights (Common Classes):")
    print(f"{'Location':<30} {'Count':>10} {'Weight':>12}")
    print('-' * 80)
    for loc, weight in sorted_weights[-10:]:
        count = location_counts[loc]
        print(f"{loc:<30} {count:>10} {weight:>12.4f}")
    
    # Weight statistics
    weight_values = np.array(list(weights.values()))
    print(f"\n{'Statistic':<30} {'Value':>12}")
    print('-' * 80)
    print(f"{'Min Weight':<30} {weight_values.min():>12.4f}")
    print(f"{'Max Weight':<30} {weight_values.max():>12.4f}")
    print(f"{'Mean Weight':<30} {weight_values.mean():>12.4f}")
    print(f"{'Median Weight':<30} {np.median(weight_values):>12.4f}")
    print(f"{'Std Weight':<30} {weight_values.std():>12.4f}")
    print(f"{'Weight Ratio (max/min)':<30} {weight_values.max() / weight_values.min():>12.2f}")


def save_weights(weights: dict, output_path: Path, metadata: dict = None):
    """
    Save weights to JSON file.
    
    Args:
        weights: Dictionary of weights
        output_path: Path to save JSON file
        metadata: Optional metadata to include
    """
    output_data = {
        'weights': weights,
        'metadata': metadata or {}
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved weights to: {output_path}")


def main():
    """Main function to compute and save class weights."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    label_path = project_root / "data" / "labels" / "enemy_location_nowcast_s1s_l5s.csv"
    output_dir = project_root / "artifacts" / "class_weights"
    
    print("="*80)
    print("COMPUTING CLASS WEIGHTS FOR ENEMY LOCATION NOWCAST")
    print("="*80)
    print(f"\nReading label file: {label_path}")
    
    if not label_path.exists():
        print(f"ERROR: Label file not found at {label_path}")
        return
    
    # Load data
    df = pd.read_csv(label_path, keep_default_na=False)
    print(f"Loaded {len(df)} samples")
    
    # Compute weights for each partition and overall
    for partition_name in ['train', 'val', 'test', 'all']:
        if partition_name == 'all':
            partition_df = df
            num_samples = len(df)
        else:
            partition_df = df[df['partition'] == partition_name]
            num_samples = len(partition_df)
            if num_samples == 0:
                continue
        
        print(f"\n{'='*80}")
        print(f"Processing partition: {partition_name.upper()} ({num_samples} samples)")
        print(f"{'='*80}")
        
        # Extract locations and count
        all_locations = extract_locations_per_partition(df, partition_name)
        location_counts = Counter(all_locations)
        
        print(f"Total location labels: {len(all_locations)}")
        print(f"Unique locations: {len(location_counts)}")
        
        # Compute weights using different methods
        methods = ['inverse', 'inverse_sqrt', 'effective_num']
        
        for method in methods:
            print(f"\n{'-'*80}")
            print(f"Method: {method.upper()}")
            print(f"{'-'*80}")
            
            weights = compute_class_weights(location_counts, method=method)
            print_weight_statistics(weights, location_counts)
            
            # Save weights
            metadata = {
                'partition': partition_name,
                'num_samples': num_samples,
                'num_labels': len(all_locations),
                'num_classes': len(location_counts),
                'method': method,
                'location_counts': dict(location_counts)
            }
            
            output_path = output_dir / f"class_weights_{partition_name}_{method}.json"
            save_weights(weights, output_path, metadata)
        
        # Also compute pos_weight for BCEWithLogitsLoss
        print(f"\n{'-'*80}")
        print("BCEWithLogitsLoss pos_weight")
        print(f"{'-'*80}")
        
        pos_weights = compute_pos_weight(location_counts, num_samples)
        
        # Print top/bottom pos_weights
        sorted_pos = sorted(pos_weights.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop 10 Highest pos_weights (Rare Classes):")
        print(f"{'Location':<30} {'Count':>10} {'pos_weight':>12}")
        print('-' * 80)
        for loc, pw in sorted_pos[:10]:
            count = location_counts[loc]
            print(f"{loc:<30} {count:>10} {pw:>12.4f}")
        
        metadata['method'] = 'pos_weight_bce'
        output_path = output_dir / f"class_weights_{partition_name}_pos_weight.json"
        save_weights(pos_weights, output_path, metadata)
    
    print(f"\n{'='*80}")
    print("USAGE INSTRUCTIONS")
    print(f"{'='*80}")
    print("""
To use these weights in your training pipeline:

1. Load the weights file:
   
   import json
   with open('artifacts/class_weights/class_weights_train_inverse.json', 'r') as f:
       data = json.load(f)
       weights_dict = data['weights']

2. Convert to tensor (matching your label order):
   
   import torch
   # Assuming self.place_names is your ordered list of locations
   weights = torch.tensor([weights_dict[place] for place in self.place_names])

3. Use in loss function:
   
   # For BCEWithLogitsLoss (use pos_weight version):
   criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
   
   # For weighted BCE manually:
   loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
   weighted_loss = (loss * weights.unsqueeze(0)).mean()

4. Compare different weighting methods:
   - 'inverse': Most aggressive, highest boost to rare classes
   - 'inverse_sqrt': Moderate, balanced approach (RECOMMENDED)
   - 'effective_num': Based on research paper, good for extreme imbalance
   - 'pos_weight': Specifically for BCEWithLogitsLoss

Recommendation: Start with 'inverse_sqrt' or 'effective_num' as they provide
a good balance between addressing imbalance and not over-emphasizing rare classes.
""")
    
    print(f"\n{'='*80}")
    print(f"Class weights saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

