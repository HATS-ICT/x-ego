import json
from pathlib import Path
from collections import defaultdict
import numpy as np

output_dir = Path("/project2/ustun_1726/x-ego/output/main_exp")

results = defaultdict(dict)

for exp_dir in sorted(output_dir.iterdir()):
    if exp_dir.is_dir():
        test_results = exp_dir / "test_analysis" / "enemy_location_best" / "test_results.json"
        if test_results.exists():
            parts = exp_dir.name.split("-")
            model = parts[0]
            task = parts[1]
            contra = "yes" if "yes" in exp_dir.name else "no"
            pov_part = [p for p in parts if p.startswith("pov")]
            if pov_part:
                pov = int(pov_part[0][3:])
            else:
                continue
            
            with open(test_results) as f:
                data = json.load(f)
                results[(pov, model, contra, task)] = data

def get_val(pov, model, contra, task, key):
    data = results.get((pov, model, contra, task))
    if not data or key not in data:
        return None
    return data[key] * 100

def format_val(val, baseline_val=None, metric=None):
    if val is None:
        return "_"
    if baseline_val is None:
        return f"{val:.2f}"
    
    diff = val - baseline_val
    if metric == "hamming_loss":
        sign = "+" if diff < 0 else ""
    else:
        sign = "+" if diff > 0 else ""
    return f"{val:.2f} ({sign}{diff:.2f})"

metrics = ["subset_accuracy", "hamming_loss", "micro_f1", "macro_f1"]

for pov in range(1, 6):
    print(f"\n{'='*140}")
    print(f"POV {pov}")
    print(f"{'='*140}")
    print(f"{'Model':<25} {'EN SubAcc':<18} {'EN HamLoss':<18} {'EN MicroF1':<18} {'EN MacroF1':<18} {'TM SubAcc':<18} {'TM HamLoss':<18} {'TM MicroF1':<18} {'TM MacroF1':<18}")
    print("-" * 190)
    
    for model in ["dinov2", "vivit"]:
        no_contra_vals = {}
        for task in ["en", "tm"]:
            for metric in metrics:
                no_contra_vals[(task, metric)] = get_val(pov, model, "no", task, metric)
        
        line = f"{model} no-contra"
        line = f"{line:<25}"
        for task in ["en", "tm"]:
            for metric in metrics:
                val = no_contra_vals[(task, metric)]
                line += f" {format_val(val):<18}"
        print(line)
    
    for model in ["dinov2", "vivit"]:
        no_contra_vals = {}
        yes_contra_vals = {}
        for task in ["en", "tm"]:
            for metric in metrics:
                no_contra_vals[(task, metric)] = get_val(pov, model, "no", task, metric)
                yes_contra_vals[(task, metric)] = get_val(pov, model, "yes", task, metric)
        
        line = f"{model} yes-contra"
        line = f"{line:<25}"
        for task in ["en", "tm"]:
            for metric in metrics:
                val = yes_contra_vals[(task, metric)]
                baseline = no_contra_vals[(task, metric)]
                line += f" {format_val(val, baseline, metric):<18}"
        print(line)

