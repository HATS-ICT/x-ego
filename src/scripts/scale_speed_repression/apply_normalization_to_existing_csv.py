"""
Apply min-max normalization to speed regression task CSVs.

This script:
1. Reads self_speed.csv and teammate_speed.csv
2. Copies original label values to new columns (label_original, label_0_original, etc.)
3. Applies min-max normalization to label columns (overwrites with normalized values)
4. Saves the modified CSVs back to disk

The global min-max values are computed from compute_scalar.py:
- Global MIN: 0.0
- Global MAX: 428.388608
"""

from pathlib import Path

import pandas as pd


# Global min-max values computed from compute_scalar.py
GLOBAL_MIN = 0.0
GLOBAL_MAX = 428.388608


def normalize_min_max(value: float, min_val: float, max_val: float) -> float:
    """Apply min-max normalization to scale value to [0, 1]."""
    return (value - min_val) / (max_val - min_val)


def apply_normalization_self_speed(labels_dir: Path) -> None:
    """Apply normalization to self_speed.csv."""
    csv_path = labels_dir / "all_tasks" / "self_speed.csv"
    print(f"Processing: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df)}")

    # Copy original values to new column
    df["label_original"] = df["label"].copy()

    # Apply normalization to label column
    df["label"] = df["label"].apply(lambda x: normalize_min_max(x, GLOBAL_MIN, GLOBAL_MAX))

    # Reorder columns to put label_original after label
    cols = df.columns.tolist()
    label_idx = cols.index("label")
    cols.remove("label_original")
    cols.insert(label_idx + 1, "label_original")
    df = df[cols]

    # Save back to CSV
    df.to_csv(csv_path, index=False)
    print(f"  Saved with normalized 'label' and original values in 'label_original'")

    # Print sample
    print(f"  Sample (first 3 rows):")
    print(f"    label (normalized): {df['label'].head(3).tolist()}")
    print(f"    label_original:     {df['label_original'].head(3).tolist()}")


def apply_normalization_teammate_speed(labels_dir: Path) -> None:
    """Apply normalization to teammate_speed.csv."""
    csv_path = labels_dir / "all_tasks" / "teammate_speed.csv"
    print(f"Processing: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df)}")

    label_cols = ["label_0", "label_1", "label_2", "label_3"]

    # Copy original values to new columns
    for col in label_cols:
        df[f"{col}_original"] = df[col].copy()

    # Apply normalization to label columns
    for col in label_cols:
        df[col] = df[col].apply(lambda x: normalize_min_max(x, GLOBAL_MIN, GLOBAL_MAX))

    # Reorder columns to put original columns after their normalized counterparts
    # Order: ..., label_0, label_0_original, label_1, label_1_original, ...
    cols = df.columns.tolist()
    new_cols = []
    for col in cols:
        if col in label_cols or col.endswith("_original"):
            continue
        new_cols.append(col)

    for col in label_cols:
        new_cols.append(col)
        new_cols.append(f"{col}_original")

    df = df[new_cols]

    # Save back to CSV
    df.to_csv(csv_path, index=False)
    print(f"  Saved with normalized label columns and original values in *_original columns")

    # Print sample
    print(f"  Sample (first 3 rows):")
    for col in label_cols:
        print(f"    {col} (normalized): {df[col].head(3).tolist()}")
        print(f"    {col}_original:     {df[f'{col}_original'].head(3).tolist()}")


def main() -> None:
    # Setup paths
    project_root = Path(__file__).resolve().parents[3]
    labels_dir = project_root / "data" / "labels"

    print(f"Project root: {project_root}")
    print(f"Labels dir: {labels_dir}")
    print(f"\nUsing global min-max values:")
    print(f"  GLOBAL_MIN: {GLOBAL_MIN}")
    print(f"  GLOBAL_MAX: {GLOBAL_MAX}")
    print()

    # Apply normalization
    apply_normalization_self_speed(labels_dir)
    print()
    apply_normalization_teammate_speed(labels_dir)

    print("\nDone! Normalization applied to both CSV files.")


if __name__ == "__main__":
    main()
