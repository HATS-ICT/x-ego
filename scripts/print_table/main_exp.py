import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

output_dir = Path(r"C:\Users\wangy\projects\x-ego\output\main_exp_no_ckpt\main_exp_no_ckpt")

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
    print(f"{'Model':<25} {'TM SubAcc':<18} {'TM HamLoss':<18} {'TM MicroF1':<18} {'TM MacroF1':<18} {'EN SubAcc':<18} {'EN HamLoss':<18} {'EN MicroF1':<18} {'EN MacroF1':<18}")
    print("-" * 190)
    
    for model in ["dinov2", "vivit", "siglip", "videomae"]:
        no_contra_vals = {}
        for task in ["en", "tm"]:
            for metric in metrics:
                no_contra_vals[(task, metric)] = get_val(pov, model, "no", task, metric)
        
        line = f"{model} no-contra"
        line = f"{line:<25}"
        for task in ["tm", "en"]:
            for metric in metrics:
                val = no_contra_vals[(task, metric)]
                line += f" {format_val(val):<18}"
        print(line)
    
    for model in ["dinov2", "vivit", "siglip", "videomae"]:
        no_contra_vals = {}
        yes_contra_vals = {}
        for task in ["en", "tm"]:
            for metric in metrics:
                no_contra_vals[(task, metric)] = get_val(pov, model, "no", task, metric)
                yes_contra_vals[(task, metric)] = get_val(pov, model, "yes", task, metric)
        
        line = f"{model} yes-contra"
        line = f"{line:<25}"
        for task in ["tm", "en"]:
            for metric in metrics:
                val = yes_contra_vals[(task, metric)]
                baseline = no_contra_vals[(task, metric)]
                line += f" {format_val(val, baseline, metric):<18}"
        print(line)

# Create visualization
print("\n\nGenerating plots...")

# Set professional style with seaborn
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.linewidth'] = 1.2

# Create figure with 8x4 subplots (4 models × 2 tasks × 4 metrics)
fig, axes = plt.subplots(8, 4, figsize=(16, 12))

# Define metrics and their display names
metric_info = {
    'subset_accuracy': 'Subset Accuracy',
    'hamming_accuracy': 'Hamming Accuracy',
    'micro_f1': 'Micro F1',
    'macro_f1': 'Macro F1'
}

# Define colors for contrastive condition (plasma scheme)
contra_colors = {
    'no': '#7201a8',   # purple from plasma
    'yes': '#fca636'   # orange from plasma
}

# Define line styles for contra
line_styles = {
    'no': '-',      # solid line for no contra
    'yes': '--'     # dashed line for yes contra
}

# Define markers for each model
model_markers = {
    'dinov2': 'o',      # circle
    'vivit': 's',       # square
    'siglip': '^',      # triangle up
    'videomae': 'v'     # triangle down
}

# POV range
povs = list(range(1, 6))

# Tasks
tasks = ['tm', 'en']
task_names = {'tm': 'Teammate', 'en': 'Opponent'}

# Models
models = ['dinov2', 'vivit', 'siglip', 'videomae']
model_display_names = {
    'dinov2': 'DINOV2',
    'vivit': 'VIVIT',
    'siglip': 'SigLIP',
    'videomae': 'VIDEOMAE'
}

# Plot for each model, task, and metric
row_idx = 0
for model in models:
    for task in tasks:
        for metric_idx, (metric_key, metric_name) in enumerate(metric_info.items()):
            ax = axes[row_idx, metric_idx]
            
            # Plot only this model's two lines (with/without contrastive)
            for contra in ['no', 'yes']:
                values = []
                for pov in povs:
                    if metric_key == 'hamming_accuracy':
                        # Convert hamming_loss to hamming_accuracy
                        ham_loss = get_val(pov, model, contra, task, 'hamming_loss')
                        if ham_loss is not None:
                            values.append(100 - ham_loss)
                        else:
                            values.append(None)
                    else:
                        values.append(get_val(pov, model, contra, task, metric_key))
                
                # Only plot if we have valid values
                if any(v is not None for v in values):
                    # Handle None values for plotting
                    plot_povs = [p for p, v in zip(povs, values) if v is not None]
                    plot_values = [v for v in values if v is not None]
                    
                    ax.plot(plot_povs, plot_values, 
                           color=contra_colors[contra],
                           linestyle=line_styles[contra],
                           marker=model_markers[model],
                           markersize=7,
                           linewidth=2.0,
                           alpha=0.85,
                           markeredgewidth=1.2,
                           markeredgecolor='white')
            
            # Styling
            # Only show metric name in top row
            if row_idx == 0:
                ax.set_title(f"{metric_name}", fontsize=12, fontweight='bold', pad=8)
            
            ax.set_xticks(povs)
            ax.set_xlim(0.8, 5.2)
            
            # Remove x and y labels
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            # Add model name and task on leftmost column only
            if metric_idx == 0:
                ax.set_ylabel(f"{model_display_names[model]}\n{task_names[task]}", 
                             fontsize=11, fontweight='bold')
            
            # Set y-axis limits based on actual data range for better separation
            # Collect all plotted values for this specific model and task
            all_values = []
            for contra in ['no', 'yes']:
                for pov in povs:
                    if metric_key == 'hamming_accuracy':
                        ham_loss = get_val(pov, model, contra, task, 'hamming_loss')
                        if ham_loss is not None:
                            all_values.append(100 - ham_loss)
                    else:
                        val = get_val(pov, model, contra, task, metric_key)
                        if val is not None:
                            all_values.append(val)
            
            if all_values:
                data_min = min(all_values)
                data_max = max(all_values)
                data_range = data_max - data_min
                # Add 10% padding on each side for clearer visualization
                padding = max(data_range * 0.1, 0.5)  # At least 0.5 padding
                ax.set_ylim(max(0, data_min - padding), min(100, data_max + padding))
            
            # Improve grid
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            ax.set_axisbelow(True)
        
        row_idx += 1

# Create custom legend (simplified since each subplot shows one model)
from matplotlib.lines import Line2D

legend_elements = []

# Add contrastive condition (colors and line styles)
legend_elements.append(Line2D([0], [0], color=contra_colors['no'], linestyle='-', 
                              linewidth=2.5, marker='o', markersize=7,
                              label='Without Contrastive'))
legend_elements.append(Line2D([0], [0], color=contra_colors['yes'], linestyle='--', 
                              linewidth=2.5, marker='o', markersize=7,
                              label='With Contrastive'))

# Place legend at the top
fig.legend(handles=legend_elements, loc='upper center', 
          bbox_to_anchor=(0.5, 0.995), frameon=False, 
          fancybox=False, shadow=False, fontsize=12, 
          ncol=2)

# Adjust layout to make room for legend at top
plt.tight_layout(rect=[0, 0, 1.0, 0.98])

# Save figure
save_path = Path("main_exp_metrics.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure saved to: {save_path}")

save_path_pdf = Path("main_exp_metrics.pdf")
plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')
print(f"Figure saved to: {save_path_pdf}")

plt.show()
