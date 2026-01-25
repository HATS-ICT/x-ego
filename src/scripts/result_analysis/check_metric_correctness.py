"""Check metric correctness by collecting all unique metrics per ml_form."""

import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

OUTPUT_BASE_PATH = Path(os.getenv("OUTPUT_BASE_PATH", "C:\\Users\\wangy\\projects\\x-ego\\output"))


def main():
    # Find all probe- folders
    probe_folders = sorted([
        f for f in OUTPUT_BASE_PATH.iterdir()
        if f.is_dir() and f.name.startswith("probe-")
    ])

    print(f"Found {len(probe_folders)} probe folders")

    # Track metrics per ml_form
    metrics_per_ml_form: dict[str, set[str]] = defaultdict(set)
    
    # Track folders with missing JSON files
    missing_both_json: list[str] = []
    missing_best_json: list[str] = []
    missing_last_json: list[str] = []
    
    # Track multi_label_cls folders with acc_subset metric
    multi_label_with_acc_subset: list[str] = []

    for folder in probe_folders:
        best_json = folder / "test_results_best.json"
        last_json = folder / "test_results_last.json"

        best_exists = best_json.exists()
        last_exists = last_json.exists()

        if not best_exists and not last_exists:
            missing_both_json.append(folder.name)
            continue
        
        if not best_exists:
            missing_best_json.append(folder.name)
        if not last_exists:
            missing_last_json.append(folder.name)

        # Read whichever JSON exists to get ml_form and metrics
        json_to_read = best_json if best_exists else last_json
        
        try:
            with open(json_to_read, "r") as f:
                data = json.load(f)
            
            ml_form = data.get("ml_form", "unknown")
            metrics = data.get("metrics", {})
            
            for metric_name in metrics.keys():
                metrics_per_ml_form[ml_form].add(metric_name)
            
            # Check for multi_label_cls with acc_subset
            if ml_form == "multi_label_cls" and "acc_subset" in metrics:
                multi_label_with_acc_subset.append(folder.name)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading {json_to_read}: {e}")

    # Print results
    print("\n" + "=" * 60)
    print("METRICS PER ML_FORM")
    print("=" * 60)
    
    for ml_form, metrics in sorted(metrics_per_ml_form.items()):
        print(f"\n{ml_form}:")
        for metric in sorted(metrics):
            print(f"  - {metric}")

    print("\n" + "=" * 60)
    print("MISSING JSON FILES")
    print("=" * 60)

    if missing_both_json:
        print(f"\nFolders missing BOTH test_results_best.json and test_results_last.json ({len(missing_both_json)}):")
        for folder in missing_both_json:
            print(f"  - {folder}")
    else:
        print("\nNo folders missing both JSON files.")

    if missing_best_json:
        print(f"\nFolders missing test_results_best.json only ({len(missing_best_json)}):")
        for folder in missing_best_json:
            print(f"  - {folder}")

    if missing_last_json:
        print(f"\nFolders missing test_results_last.json only ({len(missing_last_json)}):")
        for folder in missing_last_json:
            print(f"  - {folder}")

    print("\n" + "=" * 60)
    print("MULTI_LABEL_CLS WITH ACC_SUBSET METRIC")
    print("=" * 60)

    if multi_label_with_acc_subset:
        print(f"\nFolders with acc_subset metric ({len(multi_label_with_acc_subset)}):")
        for folder in multi_label_with_acc_subset:
            print(f"  - {folder}")
    else:
        print("\nNo multi_label_cls folders with acc_subset metric.")


if __name__ == "__main__":
    main()
