import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

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
    print(f"{'Model':<25} {'TM SubAcc':<18} {'TM HamLoss':<18} {'TM MicroF1':<18} {'TM MacroF1':<18} {'EN SubAcc':<18} {'EN HamLoss':<18} {'EN MicroF1':<18} {'EN MacroF1':<18}")
    print("-" * 190)
    
    for model in ["dinov2", "vivit", "siglip", "vjepa2", "videomae"]:
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
    
    for model in ["dinov2", "vivit", "siglip", "vjepa2", "videomae"]:
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

# Create figure with 2x4 subplots
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Define metrics and their display names
metric_info = {
    'subset_accuracy': 'Subset Accuracy',
    'hamming_accuracy': 'Hamming Accuracy',
    'micro_f1': 'Micro F1',
    'macro_f1': 'Macro F1'
}

# Define colors for contrastive condition
contra_colors = {
    'no': '#1f77b4',   # blue for without contrastive
    'yes': '#ff7f0e'   # orange for with contrastive
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
    'vjepa2': 'D',      # diamond
    'videomae': 'v'     # triangle down
}

# POV range
povs = list(range(1, 6))

# Tasks
tasks = ['tm', 'en']
task_names = {'tm': 'Teammate', 'en': 'Enemy'}

# Plot for each task and metric
for task_idx, task in enumerate(tasks):
    for metric_idx, (metric_key, metric_name) in enumerate(metric_info.items()):
        ax = axes[task_idx, metric_idx]
        
        # Plot each model and contra setting
        for model in ['dinov2', 'vivit', 'siglip', 'vjepa2', 'videomae']:
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
                    
                    # Only add label for legend (will be created separately)
                    ax.plot(plot_povs, plot_values, 
                           color=contra_colors[contra],
                           linestyle=line_styles[contra],
                           marker=model_markers[model],
                           markersize=8,
                           linewidth=2.5,
                           alpha=0.85,
                           markeredgewidth=1.5,
                           markeredgecolor='white')
        
        # Styling
        ax.set_title(f"{metric_name}", fontsize=13, fontweight='bold', pad=10)
        ax.set_xticks(povs)
        ax.set_xlim(0.8, 5.2)
        
        # Only show x-axis label on bottom row
        if task_idx == 1:
            ax.set_xlabel('', fontsize=12)
        else:
            ax.set_xlabel('', fontsize=12)
        
        # Add task name (Teammate/Enemy) as ylabel on leftmost column only
        if metric_idx == 0:
            ax.set_ylabel(f"{task_names[task]}", fontsize=14, fontweight='bold')
        else:
            ax.set_ylabel('')
        
        # Set y-axis limits based on actual data range for better separation
        # Collect all plotted values
        all_values = []
        for model in ['dinov2', 'vivit', 'siglip', 'vjepa2', 'videomae']:
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
            padding = data_range * 0.1
            ax.set_ylim(max(0, data_min - padding), min(100, data_max + padding))
        
        # Improve grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)

# Create custom legend (only once, no duplicates)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_elements = []

# Add contrastive condition (colors)
legend_elements.append(Line2D([0], [0], color=contra_colors['no'], linestyle='-', 
                              linewidth=2.5, label='Without Contrastive'))
legend_elements.append(Line2D([0], [0], color=contra_colors['yes'], linestyle='--', 
                              linewidth=2.5, label='With Contrastive'))

# Add a separator
legend_elements.append(Line2D([0], [0], color='none', label=''))

# Add model markers
for model in ['dinov2', 'vivit', 'siglip', 'vjepa2', 'videomae']:
    legend_elements.append(Line2D([0], [0], color='gray', marker=model_markers[model], 
                                  linestyle='', markersize=8, markeredgewidth=1.5,
                                  markeredgecolor='white',
                                  label=model.upper() if model != 'siglip' else 'SigLIP'))

# Place legend on the right side of the entire figure
fig.legend(handles=legend_elements, loc='center right', 
          bbox_to_anchor=(0.99, 0.5), frameon=True, 
          fancybox=True, shadow=False, fontsize=11, 
          title='Condition & Model', title_fontsize=12)

# Adjust layout
plt.tight_layout(rect=[0, 0, 0.92, 1.0])

# Save figure
save_path = Path("main_exp_metrics.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure saved to: {save_path}")

save_path_pdf = Path("main_exp_metrics.pdf")
plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')
print(f"Figure saved to: {save_path_pdf}")

plt.show()
