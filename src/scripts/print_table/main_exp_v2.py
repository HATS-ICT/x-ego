import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

output_dir = Path(r"C:\Users\wangy\projects\x-ego\output\results_archive")

# Store results keyed by (pov, model, contra, task, seed)
results = defaultdict(dict)

for exp_dir in sorted(output_dir.iterdir()):
    if exp_dir.is_dir():
        test_results = exp_dir / "test_analysis" / "enemy_location_best" / "test_results.json"
        if test_results.exists():
            parts = exp_dir.name.split("-")
            
            # Parse new naming format: main_exp_repeat_v1-{task}-nowcast-{model}-{pov}-{contra}-seed{N}-...
            # Find indices of key parts
            try:
                # Task is at index 1 (e.g., "en" or "tm")
                task = parts[1]
                
                # Model is at index 3
                model = parts[3]
                
                # POV is at index 4 (e.g., "pov2")
                pov = int(parts[4][3:])
                
                # Contra status: check if "no" is at index 5
                if parts[5] == "no":
                    # Format: pov1-no-contra-seed1
                    contra = "no"
                    seed_part = parts[7]  # parts[5]="no", parts[6]="contra", parts[7]="seed1"
                else:
                    # Format: pov1-contra-seed1
                    contra = "yes"
                    seed_part = parts[6]  # parts[5]="contra", parts[6]="seed1"
                
                # Extract seed number from "seed1", "seed2", etc.
                seed = int(seed_part[4:])
                
            except (IndexError, ValueError) as e:
                print(f"Skipping {exp_dir.name}: parsing error {e}")
                continue
            
            with open(test_results) as f:
                data = json.load(f)
                results[(pov, model, contra, task, seed)] = data

def get_val(pov, model, contra, task, key):
    """Get value for specific configuration and key."""
    data = results[(pov, model, contra, task, key)]
    if not data or key not in data:
        return None
    return data[key] * 100

def get_aggregated_val(pov, model, contra, task, key):
    """Get mean and std across all seeds for a configuration."""
    values = []
    for seed_key in results.keys():
        if seed_key[:4] == (pov, model, contra, task):
            data = results[seed_key]
            if data and key in data:
                values.append(data[key] * 100)
    
    if not values:
        return None, None
    
    return np.mean(values), np.std(values)

def format_val(mean, std, baseline_mean=None, baseline_std=None, metric=None):
    """Format value with mean ± std."""
    if mean is None:
        return "_"
    
    if baseline_mean is None:
        # Just show mean ± std
        if std is not None and std > 0:
            return f"{mean:.2f}±{std:.2f}"
        else:
            return f"{mean:.2f}"
    
    # Show mean ± std with difference from baseline
    diff = mean - baseline_mean
    if metric == "hamming_loss":
        sign = "+" if diff < 0 else ""
    else:
        sign = "+" if diff > 0 else ""
    
    if std is not None and std > 0:
        return f"{mean:.2f}±{std:.2f} ({sign}{diff:.2f})"
    else:
        return f"{mean:.2f} ({sign}{diff:.2f})"

metrics = ["subset_accuracy", "hamming_loss", "micro_f1", "macro_f1"]

# Print tables
for pov in range(1, 6):
    print(f"\n{'='*180}")
    print(f"POV {pov}")
    print(f"{'='*180}")
    print(f"{'Model':<30} {'TM SubAcc':<22} {'TM HamLoss':<22} {'TM MicroF1':<22} {'TM MacroF1':<22} {'EN SubAcc':<22} {'EN HamLoss':<22} {'EN MicroF1':<22} {'EN MacroF1':<22}")
    print("-" * 220)
    
    for model in ["dinov2", "vivit", "siglip", "videomae", "vjepa2"]:
        no_contra_vals = {}
        for task in ["en", "tm"]:
            for metric in metrics:
                mean, std = get_aggregated_val(pov, model, "no", task, metric)
                no_contra_vals[(task, metric)] = (mean, std)
        
        line = f"{model} no-contra"
        line = f"{line:<30}"
        for task in ["tm", "en"]:
            for metric in metrics:
                mean, std = no_contra_vals[(task, metric)]
                line += f" {format_val(mean, std):<22}"
        print(line)
    
    for model in ["dinov2", "vivit", "siglip", "videomae", "vjepa2"]:
        no_contra_vals = {}
        yes_contra_vals = {}
        for task in ["en", "tm"]:
            for metric in metrics:
                mean_no, std_no = get_aggregated_val(pov, model, "no", task, metric)
                no_contra_vals[(task, metric)] = (mean_no, std_no)
                mean_yes, std_yes = get_aggregated_val(pov, model, "yes", task, metric)
                yes_contra_vals[(task, metric)] = (mean_yes, std_yes)
        
        line = f"{model} yes-contra"
        line = f"{line:<30}"
        for task in ["tm", "en"]:
            for metric in metrics:
                mean, std = yes_contra_vals[(task, metric)]
                baseline_mean, baseline_std = no_contra_vals[(task, metric)]
                line += f" {format_val(mean, std, baseline_mean, baseline_std, metric):<22}"
        print(line)

# Calculate average differences across all models
print("\n\n" + "="*180)
print("AVERAGE DIFFERENCES: WITH CONTRA vs WITHOUT CONTRA (Averaged Across All Models)")
print("="*180)
print(f"{'POV':<10} {'TM SubAcc':<22} {'TM HamLoss':<22} {'TM MicroF1':<22} {'TM MacroF1':<22} {'EN SubAcc':<22} {'EN HamLoss':<22} {'EN MicroF1':<22} {'EN MacroF1':<22}")
print("-" * 220)

for pov in range(1, 6):
    line = f"POV {pov}"
    line = f"{line:<10}"
    
    for task in ["tm", "en"]:
        for metric in metrics:
            # Calculate difference for each model separately, then aggregate
            model_diffs = []
            
            for model in ["dinov2", "vivit", "siglip", "videomae", "vjepa2"]:
                # Get aggregated values for this model (across seeds)
                no_contra_values = []
                yes_contra_values = []
                
                for seed_key in results.keys():
                    if seed_key[:4] == (pov, model, "no", task):
                        data = results[seed_key]
                        if data and metric in data:
                            no_contra_values.append(data[metric] * 100)
                
                for seed_key in results.keys():
                    if seed_key[:4] == (pov, model, "yes", task):
                        data = results[seed_key]
                        if data and metric in data:
                            yes_contra_values.append(data[metric] * 100)
                
                # Calculate difference for this model
                if no_contra_values and yes_contra_values:
                    avg_no = np.mean(no_contra_values)
                    avg_yes = np.mean(yes_contra_values)
                    model_diffs.append(avg_yes - avg_no)
            
            # Calculate mean and std across models
            if model_diffs:
                mean_diff = np.mean(model_diffs)
                std_diff = np.std(model_diffs)
                
                if metric == "hamming_loss":
                    # For hamming loss, negative diff is better
                    sign = "+" if mean_diff < 0 else ""
                else:
                    # For other metrics, positive diff is better
                    sign = "+" if mean_diff > 0 else ""
                
                line += f" {sign}{mean_diff:.2f}±{std_diff:.2f}"
                line = f"{line:<22}"
            else:
                line += f" {'_':<22}"
    
    print(line)

# Calculate average differences across all models EXCLUDING SIGLIP
print("\n\n" + "="*180)
print("AVERAGE DIFFERENCES: WITH CONTRA vs WITHOUT CONTRA (Averaged Across All Models, EXCLUDING SIGLIP)")
print("="*180)
print(f"{'POV':<10} {'TM SubAcc':<22} {'TM HamLoss':<22} {'TM MicroF1':<22} {'TM MacroF1':<22} {'EN SubAcc':<22} {'EN HamLoss':<22} {'EN MicroF1':<22} {'EN MacroF1':<22}")
print("-" * 220)

for pov in range(1, 6):
    line = f"POV {pov}"
    line = f"{line:<10}"
    
    for task in ["tm", "en"]:
        for metric in metrics:
            # Calculate difference for each model separately, then aggregate
            model_diffs = []
            
            for model in ["dinov2", "vivit", "videomae", "vjepa2"]:  # Exclude siglip
                # Get aggregated values for this model (across seeds)
                no_contra_values = []
                yes_contra_values = []
                
                for seed_key in results.keys():
                    if seed_key[:4] == (pov, model, "no", task):
                        data = results[seed_key]
                        if data and metric in data:
                            no_contra_values.append(data[metric] * 100)
                
                for seed_key in results.keys():
                    if seed_key[:4] == (pov, model, "yes", task):
                        data = results[seed_key]
                        if data and metric in data:
                            yes_contra_values.append(data[metric] * 100)
                
                # Calculate difference for this model
                if no_contra_values and yes_contra_values:
                    avg_no = np.mean(no_contra_values)
                    avg_yes = np.mean(yes_contra_values)
                    model_diffs.append(avg_yes - avg_no)
            
            # Calculate mean and std across models
            if model_diffs:
                mean_diff = np.mean(model_diffs)
                std_diff = np.std(model_diffs)
                
                if metric == "hamming_loss":
                    # For hamming loss, negative diff is better
                    sign = "+" if mean_diff < 0 else ""
                else:
                    # For other metrics, positive diff is better
                    sign = "+" if mean_diff > 0 else ""
                
                line += f" {sign}{mean_diff:.2f}±{std_diff:.2f}"
                line = f"{line:<22}"
            else:
                line += f" {'_':<22}"
    
    print(line)

# Create visualization
print("\n\nGenerating plots...")

# Set professional style with seaborn
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.linewidth'] = 1.2

# Create figure with 10x4 subplots (5 models × 2 tasks × 4 metrics)
fig, axes = plt.subplots(10, 4, figsize=(16, 15))

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
    'vivit': 'o',       # square
    'siglip': 'o',      # triangle up
    'videomae': 'o',     # triangle down
    'vjepa2': 'o'     # diamond
}

# POV range
povs = list(range(1, 6))

# Tasks
tasks = ['tm', 'en']
task_names = {'tm': 'Teammate', 'en': 'Opponent'}

# Models
models = ['dinov2', 'vivit', 'siglip', 'videomae', 'vjepa2']
model_display_names = {
    'dinov2': 'DINOV2',
    'vivit': 'VIVIT',
    'siglip': 'SigLIP',
    'videomae': 'VIDEOMAE',
    'vjepa2': 'V-JEPA2'
}

def get_metric_aggregated(pov, model, contra, task, metric_key):
    """Get aggregated metric values across seeds."""
    values = []
    for seed_key in results.keys():
        if seed_key[:4] == (pov, model, contra, task):
            data = results[seed_key]
            if data and metric_key in data:
                values.append(data[metric_key] * 100)
    
    if not values:
        return None, None
    
    return np.mean(values), np.std(values)

# Plot for each model, task, and metric
row_idx = 0
for model in models:
    for task in tasks:
        for metric_idx, (metric_key, metric_name) in enumerate(metric_info.items()):
            ax = axes[row_idx, metric_idx]
            
            # Plot only this model's two lines (with/without contrastive)
            for contra in ['no', 'yes']:
                means = []
                stds = []
                for pov in povs:
                    if metric_key == 'hamming_accuracy':
                        # Convert hamming_loss to hamming_accuracy
                        ham_loss_mean, ham_loss_std = get_metric_aggregated(pov, model, contra, task, 'hamming_loss')
                        if ham_loss_mean is not None:
                            means.append(100 - ham_loss_mean)
                            stds.append(ham_loss_std if ham_loss_std is not None else 0)
                        else:
                            means.append(None)
                            stds.append(None)
                    else:
                        mean, std = get_metric_aggregated(pov, model, contra, task, metric_key)
                        means.append(mean)
                        stds.append(std if std is not None else 0)
                
                # Only plot if we have valid values
                if any(m is not None for m in means):
                    # Handle None values for plotting
                    plot_povs = [p for p, m in zip(povs, means) if m is not None]
                    plot_means = [m for m in means if m is not None]
                    plot_stds = [s for m, s in zip(means, stds) if m is not None]
                    
                    # Plot line with error band
                    line = ax.plot(plot_povs, plot_means, 
                           color=contra_colors[contra],
                           linestyle=line_styles[contra],
                           marker=model_markers[model],
                           markersize=7,
                           linewidth=2.0,
                           alpha=0.85,
                           markeredgewidth=1.2,
                           markeredgecolor='white')[0]
                    
                    # Add error band (mean ± std)
                    if plot_stds and any(s > 0 for s in plot_stds):
                        lower = [m - s for m, s in zip(plot_means, plot_stds)]
                        upper = [m + s for m, s in zip(plot_means, plot_stds)]
                        ax.fill_between(plot_povs, lower, upper,
                                       color=contra_colors[contra],
                                       alpha=0.15)
            
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
                        ham_loss_mean, _ = get_metric_aggregated(pov, model, contra, task, 'hamming_loss')
                        if ham_loss_mean is not None:
                            all_values.append(100 - ham_loss_mean)
                    else:
                        mean, _ = get_metric_aggregated(pov, model, contra, task, metric_key)
                        if mean is not None:
                            all_values.append(mean)
            
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
save_path = Path("main_exp_metrics_v2.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure saved to: {save_path}")

save_path_pdf = Path("main_exp_metrics_v2.pdf")
plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')
print(f"Figure saved to: {save_path_pdf}")

plt.show()

# Create aggregate plot showing average differences
print("\n\nGenerating aggregate difference plot...")

fig_agg, axes_agg = plt.subplots(2, 4, figsize=(16, 8))

for task_idx, task in enumerate(tasks):
    for metric_idx, (metric_key, metric_name) in enumerate(metric_info.items()):
        ax = axes_agg[task_idx, metric_idx]
        
        # Calculate average differences across all models for each POV
        pov_diffs = []
        pov_stds = []
        
        for pov in povs:
            # Calculate difference for each model separately, then aggregate
            model_diffs = []
            
            for model in models:
                # Get aggregated values for this model (across seeds)
                no_contra_values = []
                yes_contra_values = []
                
                for seed_key in results.keys():
                    if seed_key[:4] == (pov, model, "no", task):
                        data = results[seed_key]
                        if data and metric_key in data:
                            if metric_key == 'hamming_accuracy':
                                no_contra_values.append(100 - data['hamming_loss'] * 100)
                            else:
                                no_contra_values.append(data[metric_key] * 100)
                
                for seed_key in results.keys():
                    if seed_key[:4] == (pov, model, "yes", task):
                        data = results[seed_key]
                        if data and metric_key in data:
                            if metric_key == 'hamming_accuracy':
                                yes_contra_values.append(100 - data['hamming_loss'] * 100)
                            else:
                                yes_contra_values.append(data[metric_key] * 100)
                
                # Calculate difference for this model
                if no_contra_values and yes_contra_values:
                    avg_no = np.mean(no_contra_values)
                    avg_yes = np.mean(yes_contra_values)
                    model_diffs.append(avg_yes - avg_no)
            
            # Calculate mean and std across models
            if model_diffs:
                pov_diffs.append(np.mean(model_diffs))
                pov_stds.append(np.std(model_diffs))
            else:
                pov_diffs.append(None)
                pov_stds.append(None)
        
        # Plot the differences with error bands
        valid_povs = [p for p, d in zip(povs, pov_diffs) if d is not None]
        valid_diffs = [d for d in pov_diffs if d is not None]
        valid_stds = [s for d, s in zip(pov_diffs, pov_stds) if d is not None]
        
        if valid_povs and valid_diffs:
            ax.plot(valid_povs, valid_diffs, 
                   color='#0d7f99',  # Teal color
                   marker='o',
                   markersize=9,
                   linewidth=2.5,
                   alpha=0.9,
                   markeredgewidth=1.5,
                   markeredgecolor='white')
            
            # Add error band (mean ± std across models)
            if valid_stds and any(s is not None and s > 0 for s in valid_stds):
                lower = [m - s for m, s in zip(valid_diffs, valid_stds)]
                upper = [m + s for m, s in zip(valid_diffs, valid_stds)]
                ax.fill_between(valid_povs, lower, upper,
                               color='#0d7f99',
                               alpha=0.2)
            
            # Add horizontal line at y=0
            ax.axhline(y=0, color='#666666', linestyle='--', linewidth=1.5, alpha=0.7)
            
            # Styling
            if task_idx == 0:
                ax.set_title(f"{metric_name}", fontsize=13, fontweight='bold', pad=10)
            
            ax.set_xticks(povs)
            ax.set_xlim(0.8, 5.2)
            
            # Add task label on left column
            if metric_idx == 0:
                ax.set_ylabel(f"{task_names[task]}\nDifference (Δ)", 
                             fontsize=12, fontweight='bold')
            
            # Improve grid
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            ax.set_axisbelow(True)
            
            # Set y-axis to show 0 in the middle
            if valid_diffs:
                max_abs_diff = max(abs(d) for d in valid_diffs)
                padding = max(max_abs_diff * 0.2, 0.5)
                ax.set_ylim(-max_abs_diff - padding, max_abs_diff + padding)

# Add overall title
fig_agg.suptitle('Average Effect of Contrastive Learning Across All Models\n(With Contra - Without Contra)', 
                 fontsize=14, fontweight='bold', y=0.995)

# Add x-axis label
fig_agg.text(0.5, 0.02, 'POV', ha='center', fontsize=12, fontweight='bold')

# Adjust layout
plt.tight_layout(rect=[0, 0.04, 1.0, 0.97])

# Save aggregate figure
save_path_agg = Path("main_exp_aggregate_diff.png")
plt.savefig(save_path_agg, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Aggregate figure saved to: {save_path_agg}")

save_path_agg_pdf = Path("main_exp_aggregate_diff.pdf")
plt.savefig(save_path_agg_pdf, bbox_inches='tight', facecolor='white')
print(f"Aggregate figure saved to: {save_path_agg_pdf}")

plt.show()
