#!/usr/bin/env python3
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get data path
DATA_BASE_PATH = Path(os.getenv('DATA_BASE_PATH', 'data'))
if not DATA_BASE_PATH.is_absolute():
    DATA_BASE_PATH = Path(__file__).resolve().parent.parent.parent / DATA_BASE_PATH

def load_all_control_data():
    """Load all control CSV files from the data/control directory"""
    control_dir = DATA_BASE_PATH / "control"
    
    if not control_dir.exists():
        print(f"Control directory not found: {control_dir}")
        return None
    
    # Find all CSV files recursively
    csv_files = list(control_dir.rglob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {control_dir}")
        return None
    
    print(f"Found {len(csv_files)} CSV files")
    print("Loading data...")
    
    # Load and concatenate all CSV files
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not dfs:
        print("No valid CSV files loaded")
        return None
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined_df)} total rows from {len(dfs)} files")
    
    return combined_df

def analyze_buttons_distribution(df):
    """Analyze the distribution of the buttons column (categorical integers)"""
    print("\n" + "="*80)
    print("BUTTONS COLUMN DISTRIBUTION")
    print("="*80)
    
    buttons = df['buttons']
    
    # Value counts and percentages
    value_counts = buttons.value_counts().sort_index()
    percentages = (value_counts / len(buttons) * 100).round(2)
    
    print("\nValue Counts:")
    print("-" * 40)
    for value, count in value_counts.items():
        pct = percentages[value]
        print(f"  {value:10d}: {count:10d} ({pct:6.2f}%)")
    
    print(f"\nTotal unique values: {buttons.nunique()}")
    print(f"Most common value: {buttons.mode()[0]} ({percentages[buttons.mode()[0]]:.2f}%)")
    
    # Statistics
    print("\nBasic Statistics:")
    print(f"  Min: {buttons.min()}")
    print(f"  Max: {buttons.max()}")
    print(f"  Mean: {buttons.mean():.2f}")
    print(f"  Median: {buttons.median():.2f}")
    print(f"  Std: {buttons.std():.2f}")
    
    return value_counts

def analyze_mouse_distribution(df):
    """Analyze the distribution of mouse movement columns"""
    print("\n" + "="*80)
    print("MOUSE MOVEMENT DISTRIBUTION")
    print("="*80)
    
    mouse_dx = df['usercmd_mouse_dx']
    mouse_dy = df['usercmd_mouse_dy']
    
    # Mouse DX analysis
    print("\nusercmd_mouse_dx:")
    print("-" * 40)
    print(f"  Count: {len(mouse_dx)}")
    print(f"  Min: {mouse_dx.min()}")
    print(f"  Max: {mouse_dx.max()}")
    print(f"  Mean: {mouse_dx.mean():.2f}")
    print(f"  Median: {mouse_dx.median():.2f}")
    print(f"  Std: {mouse_dx.std():.2f}")
    print(f"  25th percentile: {mouse_dx.quantile(0.25):.2f}")
    print(f"  75th percentile: {mouse_dx.quantile(0.75):.2f}")
    print(f"  95th percentile: {mouse_dx.quantile(0.95):.2f}")
    print(f"  99th percentile: {mouse_dx.quantile(0.99):.2f}")
    print(f"  Zeros: {(mouse_dx == 0).sum()} ({(mouse_dx == 0).sum() / len(mouse_dx) * 100:.2f}%)")
    print(f"  Positive: {(mouse_dx > 0).sum()} ({(mouse_dx > 0).sum() / len(mouse_dx) * 100:.2f}%)")
    print(f"  Negative: {(mouse_dx < 0).sum()} ({(mouse_dx < 0).sum() / len(mouse_dx) * 100:.2f}%)")
    
    # Mouse DY analysis
    print("\nusercmd_mouse_dy:")
    print("-" * 40)
    print(f"  Count: {len(mouse_dy)}")
    print(f"  Min: {mouse_dy.min()}")
    print(f"  Max: {mouse_dy.max()}")
    print(f"  Mean: {mouse_dy.mean():.2f}")
    print(f"  Median: {mouse_dy.median():.2f}")
    print(f"  Std: {mouse_dy.std():.2f}")
    print(f"  25th percentile: {mouse_dy.quantile(0.25):.2f}")
    print(f"  75th percentile: {mouse_dy.quantile(0.75):.2f}")
    print(f"  95th percentile: {mouse_dy.quantile(0.95):.2f}")
    print(f"  99th percentile: {mouse_dy.quantile(0.99):.2f}")
    print(f"  Zeros: {(mouse_dy == 0).sum()} ({(mouse_dy == 0).sum() / len(mouse_dy) * 100:.2f}%)")
    print(f"  Positive: {(mouse_dy > 0).sum()} ({(mouse_dy > 0).sum() / len(mouse_dy) * 100:.2f}%)")
    print(f"  Negative: {(mouse_dy < 0).sum()} ({(mouse_dy < 0).sum() / len(mouse_dy) * 100:.2f}%)")
    
    return mouse_dx, mouse_dy

def create_visualizations(df, buttons_value_counts, mouse_dx, mouse_dy):
    """Create visualization plots for the distributions"""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Control Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Buttons distribution - bar plot (top values)
    ax1 = axes[0, 0]
    top_buttons = buttons_value_counts.head(20)
    ax1.bar(range(len(top_buttons)), top_buttons.values)
    ax1.set_xlabel('Button Value')
    ax1.set_ylabel('Count')
    ax1.set_title('Buttons Distribution (Top 20)')
    ax1.set_xticks(range(len(top_buttons)))
    ax1.set_xticklabels(top_buttons.index, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Buttons distribution - log scale
    ax2 = axes[0, 1]
    top_buttons_log = buttons_value_counts.head(20)
    ax2.bar(range(len(top_buttons_log)), top_buttons_log.values)
    ax2.set_xlabel('Button Value')
    ax2.set_ylabel('Count (log scale)')
    ax2.set_title('Buttons Distribution (Log Scale, Top 20)')
    ax2.set_yscale('log')
    ax2.set_xticks(range(len(top_buttons_log)))
    ax2.set_xticklabels(top_buttons_log.index, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Mouse DX histogram
    ax3 = axes[0, 2]
    ax3.hist(mouse_dx, bins=100, edgecolor='black', alpha=0.7)
    ax3.set_xlabel('usercmd_mouse_dx')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Mouse DX Distribution')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Mouse DY histogram
    ax4 = axes[1, 0]
    ax4.hist(mouse_dy, bins=100, edgecolor='black', alpha=0.7)
    ax4.set_xlabel('usercmd_mouse_dy')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Mouse DY Distribution')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Mouse DX vs DY scatter (sampled for performance)
    ax5 = axes[1, 1]
    sample_size = min(10000, len(mouse_dx))
    sample_indices = np.random.choice(len(mouse_dx), sample_size, replace=False)
    ax5.scatter(mouse_dx.iloc[sample_indices], mouse_dy.iloc[sample_indices], 
                alpha=0.1, s=1)
    ax5.set_xlabel('usercmd_mouse_dx')
    ax5.set_ylabel('usercmd_mouse_dy')
    ax5.set_title(f'Mouse DX vs DY (Sample of {sample_size})')
    ax5.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax5.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax5.grid(True, alpha=0.3)
    
    # 6. Mouse movement magnitude distribution
    ax6 = axes[1, 2]
    magnitude = np.sqrt(mouse_dx**2 + mouse_dy**2)
    ax6.hist(magnitude, bins=100, edgecolor='black', alpha=0.7)
    ax6.set_xlabel('Mouse Movement Magnitude')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Mouse Movement Magnitude Distribution')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).resolve().parent.parent.parent / "artifacts"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "control_distribution_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")
    
    plt.show()

def main():
    print("Control Distribution Inspector")
    print("="*80)
    print(f"Data path: {DATA_BASE_PATH}")
    
    # Load data
    df = load_all_control_data()
    
    if df is None:
        return
    
    # Check required columns exist
    required_cols = ['buttons', 'usercmd_mouse_dx', 'usercmd_mouse_dy']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return
    
    # Analyze distributions
    buttons_value_counts = analyze_buttons_distribution(df)
    mouse_dx, mouse_dy = analyze_mouse_distribution(df)
    
    # Create visualizations
    create_visualizations(df, buttons_value_counts, mouse_dx, mouse_dy)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()

