"""
Analyze label statistics for all task CSVs.

This script reads generated label CSV files and outputs statistics about
data balance for different ML task forms (binary_cls, multi_cls, multi_label_cls, regression).

Usage:
    python -m scripts.task_creator.analyze_label_stats [--labels_dir LABELS_DIR] [--output OUTPUT]
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.task_creator.task_definitions import load_task_definitions


def analyze_binary_cls(df: pd.DataFrame, label_col: str) -> Dict[str, Any]:
    """Analyze binary classification task balance."""
    if label_col not in df.columns:
        return {"error": f"Column '{label_col}' not found"}
    
    labels = df[label_col].dropna()
    total = len(labels)
    
    if total == 0:
        return {"error": "No valid labels"}
    
    value_counts = labels.value_counts().to_dict()
    
    # Assuming binary labels are 0 and 1
    count_0 = value_counts.get(0, 0)
    count_1 = value_counts.get(1, 0)
    
    ratio = count_1 / count_0 if count_0 > 0 else float('inf')
    
    return {
        "total_samples": total,
        "class_0_count": count_0,
        "class_1_count": count_1,
        "class_0_pct": round(100 * count_0 / total, 2),
        "class_1_pct": round(100 * count_1 / total, 2),
        "positive_ratio": round(ratio, 4),
        "imbalance_ratio": round(max(count_0, count_1) / min(count_0, count_1), 2) if min(count_0, count_1) > 0 else float('inf'),
    }


def analyze_multi_cls(df: pd.DataFrame, label_col: str, num_classes: int = None) -> Dict[str, Any]:
    """Analyze multi-class classification task balance."""
    if label_col not in df.columns:
        return {"error": f"Column '{label_col}' not found"}
    
    labels = df[label_col].dropna()
    total = len(labels)
    
    if total == 0:
        return {"error": "No valid labels"}
    
    value_counts = labels.value_counts().sort_index()
    
    # Class distribution
    class_dist = {}
    for cls, count in value_counts.items():
        class_dist[int(cls)] = {
            "count": int(count),
            "pct": round(100 * count / total, 2)
        }
    
    counts = list(value_counts.values)
    
    return {
        "total_samples": total,
        "num_classes_observed": len(value_counts),
        "num_classes_expected": num_classes,
        "class_distribution": class_dist,
        "min_class_count": int(min(counts)) if counts else 0,
        "max_class_count": int(max(counts)) if counts else 0,
        "mean_class_count": round(np.mean(counts), 2) if counts else 0,
        "std_class_count": round(np.std(counts), 2) if counts else 0,
        "imbalance_ratio": round(max(counts) / min(counts), 2) if min(counts) > 0 else float('inf'),
    }


def analyze_multi_label_cls(df: pd.DataFrame, label_prefix: str, num_classes: int = None) -> Dict[str, Any]:
    """Analyze multi-label classification task balance."""
    # Find all label columns matching the prefix
    label_cols = [c for c in df.columns if c.startswith(label_prefix)]
    
    if not label_cols:
        # Try finding columns with 'label_' prefix
        label_cols = [c for c in df.columns if c.startswith('label_')]
    
    if not label_cols:
        return {"error": f"No label columns found with prefix '{label_prefix}'"}
    
    total = len(df)
    if total == 0:
        return {"error": "No samples"}
    
    # Per-label statistics
    label_stats = {}
    positive_counts = []
    
    for col in sorted(label_cols):
        if col in df.columns:
            col_data = df[col].dropna()
            pos_count = (col_data == 1).sum()
            neg_count = (col_data == 0).sum()
            
            label_stats[col] = {
                "positive_count": int(pos_count),
                "positive_pct": round(100 * pos_count / len(col_data), 2) if len(col_data) > 0 else 0,
            }
            positive_counts.append(pos_count)
    
    # Labels per sample distribution
    label_matrix = df[label_cols].values
    labels_per_sample = np.sum(label_matrix == 1, axis=1)
    
    return {
        "total_samples": total,
        "num_labels": len(label_cols),
        "label_columns": label_cols[:10],  # First 10 for brevity
        "per_label_positive_rate": {
            "min_pct": round(100 * min(positive_counts) / total, 2) if positive_counts else 0,
            "max_pct": round(100 * max(positive_counts) / total, 2) if positive_counts else 0,
            "mean_pct": round(100 * np.mean(positive_counts) / total, 2) if positive_counts else 0,
        },
        "labels_per_sample": {
            "min": int(labels_per_sample.min()),
            "max": int(labels_per_sample.max()),
            "mean": round(labels_per_sample.mean(), 2),
            "std": round(labels_per_sample.std(), 2),
        },
        "detailed_label_stats": label_stats,
    }


def analyze_regression(df: pd.DataFrame, label_col: str) -> Dict[str, Any]:
    """Analyze regression task label distribution."""
    # Find regression label columns
    if label_col in df.columns:
        label_cols = [label_col]
    else:
        # Try common patterns
        label_cols = [c for c in df.columns if c.startswith('label_')]
    
    if not label_cols:
        return {"error": "No label columns found"}
    
    total = len(df)
    if total == 0:
        return {"error": "No samples"}
    
    stats = {"total_samples": total, "label_columns": label_cols}
    
    for col in label_cols:
        values = df[col].dropna()
        if len(values) == 0:
            continue
            
        stats[col] = {
            "count": int(len(values)),
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std()), 4),
            "median": round(float(values.median()), 4),
            "q25": round(float(values.quantile(0.25)), 4),
            "q75": round(float(values.quantile(0.75)), 4),
        }
    
    return stats


def detect_ml_form_from_columns(df: pd.DataFrame, task_id: str) -> str:
    """Try to detect ML form from column patterns."""
    cols = df.columns.tolist()
    label_cols = [c for c in cols if c.startswith('label_')]
    
    if not label_cols:
        return "unknown"
    
    # Check if multi-label (multiple label_ columns with place indices)
    if len(label_cols) > 1 and any('place_' in c for c in label_cols):
        return "multi_label_cls"
    
    # Check first label column values
    first_label = label_cols[0]
    unique_vals = df[first_label].dropna().unique()
    
    if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        return "binary_cls"
    
    if df[first_label].dtype in [np.float64, np.float32] and len(unique_vals) > 10:
        return "regression"
    
    if len(unique_vals) > 2 and len(unique_vals) <= 30:
        return "multi_cls"
    
    return "unknown"


def analyze_csv_file(csv_path: Path, task_def: Optional[Any] = None) -> Dict[str, Any]:
    """Analyze a single CSV file."""
    if not csv_path.exists():
        return {"error": f"File not found: {csv_path}"}
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"error": f"Failed to read CSV: {e}"}
    
    result = {
        "file": str(csv_path.name),
        "total_rows": len(df),
        "columns": list(df.columns),
    }
    
    # Determine ML form
    if task_def:
        ml_form = task_def.ml_form.value
        num_classes = task_def.num_classes
        label_field = task_def.label_field
    else:
        ml_form = detect_ml_form_from_columns(df, csv_path.stem)
        num_classes = None
        label_field = "label"
    
    result["ml_form"] = ml_form
    
    # Find the label column(s)
    label_cols = [c for c in df.columns if c.startswith('label_')]
    if label_cols:
        primary_label = label_cols[0]
    else:
        primary_label = None
    
    # Analyze based on ML form
    if ml_form == "binary_cls":
        if primary_label:
            result["label_stats"] = analyze_binary_cls(df, primary_label)
        else:
            result["label_stats"] = {"error": "No label column found"}
            
    elif ml_form == "multi_cls":
        if primary_label:
            result["label_stats"] = analyze_multi_cls(df, primary_label, num_classes)
        else:
            result["label_stats"] = {"error": "No label column found"}
            
    elif ml_form == "multi_label_cls":
        result["label_stats"] = analyze_multi_label_cls(df, "label_", num_classes)
        
    elif ml_form == "regression":
        result["label_stats"] = analyze_regression(df, primary_label or "label")
        
    else:
        result["label_stats"] = {"error": f"Unknown ML form: {ml_form}"}
    
    return result


def print_stats_summary(all_stats: Dict[str, Dict], verbose: bool = False):
    """Print formatted summary of all statistics."""
    print("\n" + "=" * 80)
    print("LABEL STATISTICS SUMMARY")
    print("=" * 80)
    
    # Group by ML form
    by_form = {}
    for task_id, stats in all_stats.items():
        ml_form = stats.get("ml_form", "unknown")
        if ml_form not in by_form:
            by_form[ml_form] = []
        by_form[ml_form].append((task_id, stats))
    
    for ml_form in ["binary_cls", "multi_cls", "multi_label_cls", "regression", "unknown"]:
        if ml_form not in by_form:
            continue
            
        tasks = by_form[ml_form]
        print(f"\n{'-' * 80}")
        print(f" {ml_form.upper()} ({len(tasks)} tasks)")
        print(f"{'-' * 80}")
        
        for task_id, stats in sorted(tasks, key=lambda x: x[0]):
            label_stats = stats.get("label_stats", {})
            
            if "error" in label_stats:
                print(f"\n  {task_id}: ERROR - {label_stats['error']}")
                continue
            
            print(f"\n  {task_id}:")
            print(f"    Samples: {stats.get('total_rows', 'N/A')}")
            
            if ml_form == "binary_cls":
                print(f"    Class 0: {label_stats.get('class_0_count', 'N/A')} ({label_stats.get('class_0_pct', 'N/A')}%)")
                print(f"    Class 1: {label_stats.get('class_1_count', 'N/A')} ({label_stats.get('class_1_pct', 'N/A')}%)")
                print(f"    Imbalance ratio: {label_stats.get('imbalance_ratio', 'N/A')}:1")
                
            elif ml_form == "multi_cls":
                print(f"    Classes observed: {label_stats.get('num_classes_observed', 'N/A')}")
                print(f"    Min/Max class count: {label_stats.get('min_class_count', 'N/A')} / {label_stats.get('max_class_count', 'N/A')}")
                print(f"    Imbalance ratio: {label_stats.get('imbalance_ratio', 'N/A')}:1")
                if verbose and "class_distribution" in label_stats:
                    for cls, info in label_stats["class_distribution"].items():
                        print(f"      Class {cls}: {info['count']} ({info['pct']}%)")
                        
            elif ml_form == "multi_label_cls":
                print(f"    Num labels: {label_stats.get('num_labels', 'N/A')}")
                per_label = label_stats.get("per_label_positive_rate", {})
                print(f"    Label positive rate: {per_label.get('min_pct', 'N/A')}% - {per_label.get('max_pct', 'N/A')}% (mean: {per_label.get('mean_pct', 'N/A')}%)")
                lps = label_stats.get("labels_per_sample", {})
                print(f"    Labels per sample: {lps.get('mean', 'N/A')} avg (range: {lps.get('min', 'N/A')}-{lps.get('max', 'N/A')})")
                
            elif ml_form == "regression":
                for col in label_stats.get("label_columns", []):
                    if col in label_stats:
                        col_stats = label_stats[col]
                        print(f"    {col}:")
                        print(f"      Range: [{col_stats.get('min', 'N/A')}, {col_stats.get('max', 'N/A')}]")
                        print(f"      Mean +/- Std: {col_stats.get('mean', 'N/A')} +/- {col_stats.get('std', 'N/A')}")
                        print(f"      Median [Q25, Q75]: {col_stats.get('median', 'N/A')} [{col_stats.get('q25', 'N/A')}, {col_stats.get('q75', 'N/A')}]")


def main():
    parser = argparse.ArgumentParser(description="Analyze label statistics for task CSVs")
    parser.add_argument("--labels_dir", type=str, default=None,
                        help="Directory containing label CSVs (default: DATA_BASE_PATH/labels/all_tasks)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for detailed stats (optional)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed per-class statistics")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Specific task IDs to analyze (default: all)")
    args = parser.parse_args()
    
    load_dotenv()
    
    DATA_BASE_PATH = os.getenv('DATA_BASE_PATH')
    if not DATA_BASE_PATH:
        print("ERROR: DATA_BASE_PATH environment variable not set")
        sys.exit(1)
    
    labels_dir = args.labels_dir
    if labels_dir is None:
        labels_dir = os.path.join(DATA_BASE_PATH, 'labels', 'all_tasks')
    
    labels_path = Path(labels_dir)
    
    if not labels_path.exists():
        print(f"ERROR: Labels directory not found: {labels_path}")
        sys.exit(1)
    
    # Load task definitions for metadata
    try:
        task_defs = load_task_definitions()
        task_def_map = {t.task_id: t for t in task_defs}
    except Exception as e:
        print(f"Warning: Could not load task definitions: {e}")
        task_def_map = {}
    
    print(f"Analyzing labels in: {labels_path}")
    
    # Find all CSV files
    csv_files = sorted(labels_path.glob("*.csv"))
    
    if args.tasks:
        csv_files = [f for f in csv_files if f.stem in args.tasks]
    
    if not csv_files:
        print("No CSV files found")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Analyze each file
    all_stats = {}
    
    for csv_file in csv_files:
        task_id = csv_file.stem
        task_def = task_def_map.get(task_id)
        
        print(f"  Analyzing: {csv_file.name}...")
        stats = analyze_csv_file(csv_file, task_def)
        all_stats[task_id] = stats
    
    # Print summary
    print_stats_summary(all_stats, verbose=args.verbose)
    
    # Save detailed stats if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(all_stats, f, indent=2, default=str)
        print(f"\nDetailed stats saved to: {output_path}")


if __name__ == "__main__":
    main()
