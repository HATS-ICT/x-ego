import json
from pathlib import Path
from collections import defaultdict
import numpy as np

output_dir = Path("/project2/ustun_1726/x-ego/output/main_exp")

results = defaultdict(list)

for exp_dir in sorted(output_dir.iterdir()):
    if exp_dir.is_dir():
        test_results = exp_dir / "test_analysis" / "enemy_location_best" / "test_results.json"
        if test_results.exists():
            parts = exp_dir.name.split("-")
            model = parts[0]
            task = parts[1]
            contra = "yes" if "yes" in exp_dir.name else "no"
            
            with open(test_results) as f:
                data = json.load(f)
                results[(model, contra, task)].append(data)

def format_val(vals, key):
    if not vals:
        return "_"
    nums = [v[key] for v in vals if key in v]
    if not nums:
        return "_"
    return f"{np.mean(nums) * 100:.2f}"

rows = [
    ("dinov2", "no"),
    ("vivit", "no"),
    ("dinov2", "yes"),
    ("vivit", "yes"),
]

metrics = ["subset_accuracy", "hamming_loss", "micro_f1", "macro_f1"]

print(f"{'Model':<20} {'EN SubAcc':<12} {'EN HamLoss':<12} {'EN MicroF1':<12} {'EN MacroF1':<12} {'TM SubAcc':<12} {'TM HamLoss':<12} {'TM MicroF1':<12} {'TM MacroF1':<12}")
print("-" * 140)

for model, contra in rows:
    row_name = f"{model} {contra}-contra"
    en_vals = results[(model, contra, "en")]
    tm_vals = results[(model, contra, "tm")]
    
    line = f"{row_name:<20}"
    for metric in metrics:
        line += f" {format_val(en_vals, metric):<12}"
    for metric in metrics:
        line += f" {format_val(tm_vals, metric):<12}"
    print(line)

