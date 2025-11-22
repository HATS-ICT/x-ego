import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

def extract_places_from_row(row, prefix='player', num_agents=10):
    places = []
    for i in range(num_agents):
        place_col = f'{prefix}_{i}_place'
        if place_col in row.index:
            place = row[place_col]
            if place and str(place).strip() and str(place) != 'nan':
                places.append(place)
    return places

def create_multihot_vector(places, place_to_idx):
    vector = np.zeros(len(place_to_idx))
    for place in places:
        if place in place_to_idx:
            vector[place_to_idx[place]] = 1
    return vector

def analyze_csv(csv_path, prefix='player', num_agents=10):
    df = pd.read_csv(csv_path, keep_default_na=False)
    
    all_places = set()
    for _, row in df.iterrows():
        places = extract_places_from_row(row, prefix, num_agents)
        all_places.update(places)
    
    all_places = sorted(list(all_places))
    place_to_idx = {place: idx for idx, place in enumerate(all_places)}
    
    multihot_vectors = []
    for _, row in df.iterrows():
        places = extract_places_from_row(row, prefix, num_agents)
        vector = create_multihot_vector(places, place_to_idx)
        multihot_vectors.append(vector)
    
    multihot_vectors = np.array(multihot_vectors)
    
    return multihot_vectors, all_places, place_to_idx, df

def plot_distribution(multihot_vectors, place_names, title, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16)
    
    num_active_per_sample = multihot_vectors.sum(axis=1)
    axes[0, 0].hist(num_active_per_sample, bins=np.arange(num_active_per_sample.max() + 2) - 0.5, edgecolor='black')
    axes[0, 0].set_xlabel('Number of Occupied Places per Sample')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Occupied Places per Sample')
    axes[0, 0].grid(True, alpha=0.3)
    
    place_frequencies = multihot_vectors.sum(axis=0)
    sorted_indices = np.argsort(place_frequencies)[::-1]
    top_k = min(20, len(place_names))
    top_indices = sorted_indices[:top_k]
    
    axes[0, 1].bar(range(top_k), place_frequencies[top_indices])
    axes[0, 1].set_xlabel('Place Index')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Top {top_k} Most Frequent Places')
    axes[0, 1].set_xticks(range(top_k))
    axes[0, 1].set_xticklabels([place_names[i] for i in top_indices], rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    occupation_rates = place_frequencies / len(multihot_vectors)
    axes[1, 0].hist(occupation_rates, bins=50, edgecolor='black')
    axes[1, 0].set_xlabel('Occupation Rate')
    axes[1, 0].set_ylabel('Number of Places')
    axes[1, 0].set_title('Distribution of Place Occupation Rates')
    axes[1, 0].grid(True, alpha=0.3)
    
    cooccurrence_matrix = multihot_vectors.T @ multihot_vectors
    np.fill_diagonal(cooccurrence_matrix, 0)
    
    top_n = min(15, len(place_names))
    top_place_indices = sorted_indices[:top_n]
    cooc_subset = cooccurrence_matrix[np.ix_(top_place_indices, top_place_indices)]
    
    im = axes[1, 1].imshow(cooc_subset, cmap='YlOrRd', aspect='auto')
    axes[1, 1].set_xticks(range(top_n))
    axes[1, 1].set_yticks(range(top_n))
    axes[1, 1].set_xticklabels([place_names[i] for i in top_place_indices], rotation=45, ha='right')
    axes[1, 1].set_yticklabels([place_names[i] for i in top_place_indices])
    axes[1, 1].set_title(f'Co-occurrence Matrix (Top {top_n} Places)')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    
    print(f"\nStatistics:")
    print(f"Total samples: {len(multihot_vectors)}")
    print(f"Total unique places: {len(place_names)}")
    print(f"Mean occupied places per sample: {num_active_per_sample.mean():.2f}")
    print(f"Std occupied places per sample: {num_active_per_sample.std():.2f}")
    print(f"Min/Max occupied places per sample: {num_active_per_sample.min():.0f}/{num_active_per_sample.max():.0f}")

def main():
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'labels'
    output_dir = Path(__file__).parent
    
    teammate_csv = data_dir / 'teammate_location_nowcast_s1s_l5s.csv'
    enemy_csv = data_dir / 'enemy_location_nowcast_s1s_l5s.csv'
    
    if teammate_csv.exists():
        print("=" * 60)
        print("Analyzing Teammate Location Labels")
        print("=" * 60)
        vectors, places, place_to_idx, df = analyze_csv(teammate_csv, prefix='teammate', num_agents=5)
        plot_distribution(vectors, places, 'Teammate Location Label Distribution', 
                        output_dir / 'teammate_label_dist.png')
    
    if enemy_csv.exists():
        print("\n" + "=" * 60)
        print("Analyzing Enemy Location Labels")
        print("=" * 60)
        vectors, places, place_to_idx, df = analyze_csv(enemy_csv, prefix='player', num_agents=10)
        plot_distribution(vectors, places, 'Enemy Location Label Distribution', 
                        output_dir / 'enemy_label_dist.png')

if __name__ == '__main__':
    main()

