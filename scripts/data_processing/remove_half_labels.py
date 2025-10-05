import pandas as pd
from pathlib import Path

# Set random seed for reproducibility
RANDOM_SEED = 42

# Define paths
data_dir = Path(__file__).parent.parent.parent / "data" / "labels"
input_file = data_dir / "teammate_location_nowcast_s1s_l5s.csv"
output_file = data_dir / "teammate_location_nowcast_s1s_l5s_half.csv"

# Read the CSV file
print(f"Reading {input_file}...")
df = pd.read_csv(input_file)

print(f"Original dataset size: {len(df)}")
print(f"Partition counts: {df['partition'].value_counts().to_dict()}")

# Sample half of the data from each partition independently
sampled_dfs = []
for partition in df["partition"].unique():
    partition_df = df[df["partition"] == partition]
    n_samples = len(partition_df) // 2
    sampled_df = partition_df.sample(n=n_samples, random_state=RANDOM_SEED)
    print(f"  {partition}: {len(partition_df)} -> {len(sampled_df)}")
    sampled_dfs.append(sampled_df)

# Concatenate all sampled partitions
result_df = pd.concat(sampled_dfs, ignore_index=False)

# Sort by index to maintain relative order
result_df = result_df.sort_index()

# Reset index and update the idx column
result_df = result_df.reset_index(drop=True)
result_df["idx"] = result_df.index

print(f"\nFinal dataset size: {len(result_df)}")
print(f"Partition counts: {result_df['partition'].value_counts().to_dict()}")

# Save to new file
print(f"\nSaving to {output_file}...")
result_df.to_csv(output_file, index=False)
print("Done!")
