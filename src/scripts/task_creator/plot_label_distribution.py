"""
Plot label distribution for each task.

This script reads task definitions and generates visualization plots
for each task's label distribution, including time distribution and
other relevant statistics.

Usage:
    python -m src.scripts.task_creator.plot_label_distribution [--labels_dir LABELS_DIR] [--output_dir OUTPUT_DIR]
"""

import os
import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dotenv import load_dotenv

from .task_definitions import (
    load_task_definitions,
    TaskDefinition,
    MLForm,
    IDX_TO_PLACE,
)


def setup_style():
    """Set up matplotlib style for consistent plots."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })


def plot_binary_cls(
    df: pd.DataFrame,
    task_def: TaskDefinition,
    output_path: Path,
):
    """
    Plot binary classification task distribution.
    
    Creates a figure with:
    - Class distribution bar chart
    - Distribution by pov_side (CT vs T)
    - Distribution by round_num (if available)
    - Temporal distribution (start_tick)
    """
    # Find label column
    label_col = "label" if "label" in df.columns else None
    if label_col is None:
        label_cols = [c for c in df.columns if c.startswith("label")]
        if label_cols:
            label_col = label_cols[0]
    
    if label_col is None:
        print(f"  Warning: No label column found for {task_def.task_id}")
        return
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle(
        f"{task_def.task_name} ({task_def.task_id})\n{task_def.description}",
        fontsize=14,
        fontweight="bold",
    )
    
    # 1. Class distribution
    ax1 = fig.add_subplot(gs[0, 0])
    labels = df[label_col].dropna()
    counts = labels.value_counts().sort_index()
    colors = ["#3498db", "#e74c3c"]
    bars = ax1.bar(
        [f"Class {int(i)}" for i in counts.index],
        counts.values,
        color=colors[: len(counts)],
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_title("Class Distribution")
    ax1.set_ylabel("Count")
    
    # Add percentage labels
    total = len(labels)
    for bar, count in zip(bars, counts.values):
        pct = 100 * count / total
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    
    # Add imbalance ratio
    if len(counts) == 2:
        ratio = max(counts.values) / min(counts.values)
        ax1.text(
            0.95,
            0.95,
            f"Imbalance: {ratio:.2f}:1",
            transform=ax1.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    
    # 2. Distribution by side (CT vs T)
    ax2 = fig.add_subplot(gs[0, 1])
    if "pov_side" in df.columns:
        side_label_counts = df.groupby(["pov_side", label_col]).size().unstack(fill_value=0)
        side_label_counts.plot(kind="bar", ax=ax2, color=colors, edgecolor="black", linewidth=0.5)
        ax2.set_title("Distribution by POV Side")
        ax2.set_xlabel("Side")
        ax2.set_ylabel("Count")
        ax2.legend(title="Class", labels=[f"Class {int(c)}" for c in side_label_counts.columns])
        ax2.tick_params(axis="x", rotation=0)
    else:
        ax2.text(0.5, 0.5, "No pov_side column", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Distribution by POV Side")
    
    # 3. Distribution by partition
    ax3 = fig.add_subplot(gs[0, 2])
    if "partition" in df.columns:
        partition_label_counts = df.groupby(["partition", label_col]).size().unstack(fill_value=0)
        partition_label_counts.plot(kind="bar", ax=ax3, color=colors, edgecolor="black", linewidth=0.5)
        ax3.set_title("Distribution by Partition")
        ax3.set_xlabel("Partition")
        ax3.set_ylabel("Count")
        ax3.legend(title="Class", labels=[f"Class {int(c)}" for c in partition_label_counts.columns])
        ax3.tick_params(axis="x", rotation=0)
    else:
        ax3.text(0.5, 0.5, "No partition column", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Distribution by Partition")
    
    # 4. Temporal distribution (by round_num)
    ax4 = fig.add_subplot(gs[1, 0])
    if "round_num" in df.columns:
        round_counts = df.groupby(["round_num", label_col]).size().unstack(fill_value=0)
        round_counts.plot(kind="bar", ax=ax4, color=colors, edgecolor="black", linewidth=0.5, width=0.8)
        ax4.set_title("Distribution by Round Number")
        ax4.set_xlabel("Round")
        ax4.set_ylabel("Count")
        ax4.legend(title="Class", labels=[f"Class {int(c)}" for c in round_counts.columns])
        ax4.tick_params(axis="x", rotation=45)
    else:
        ax4.text(0.5, 0.5, "No round_num column", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("Distribution by Round Number")
    
    # 5. Start tick distribution (time within round)
    ax5 = fig.add_subplot(gs[1, 1])
    if "start_tick" in df.columns:
        for cls in sorted(labels.unique()):
            subset = df[df[label_col] == cls]["start_tick"]
            ax5.hist(
                subset,
                bins=30,
                alpha=0.6,
                label=f"Class {int(cls)}",
                color=colors[int(cls) % len(colors)],
                edgecolor="black",
                linewidth=0.3,
            )
        ax5.set_title("Distribution by Start Tick (Time in Round)")
        ax5.set_xlabel("Start Tick")
        ax5.set_ylabel("Count")
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, "No start_tick column", ha="center", va="center", transform=ax5.transAxes)
        ax5.set_title("Distribution by Start Tick")
    
    # 6. Stats summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    
    stats_text = f"""Summary Statistics
─────────────────────
Total Samples: {total:,}
Class 0: {int(counts.get(0, 0)):,} ({100*counts.get(0, 0)/total:.1f}%)
Class 1: {int(counts.get(1, 0)):,} ({100*counts.get(1, 0)/total:.1f}%)

Task Info
─────────────────────
Category: {task_def.category.value}
Temporal: {task_def.temporal_type.value}
Horizon: {task_def.horizon_sec}s
Data Source: {task_def.primary_data_source.value}
"""
    ax6.text(
        0.1,
        0.9,
        stats_text,
        transform=ax6.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3),
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_multi_cls(
    df: pd.DataFrame,
    task_def: TaskDefinition,
    output_path: Path,
):
    """
    Plot multi-class classification task distribution.
    
    Creates a figure with:
    - Class distribution bar chart
    - Distribution by pov_side
    - Distribution by partition
    - Temporal distribution
    """
    # Find label column
    label_col = "label" if "label" in df.columns else None
    if label_col is None:
        label_cols = [c for c in df.columns if c.startswith("label")]
        if label_cols:
            label_col = label_cols[0]
    
    if label_col is None:
        print(f"  Warning: No label column found for {task_def.task_id}")
        return
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    fig.suptitle(
        f"{task_def.task_name} ({task_def.task_id})\n{task_def.description}",
        fontsize=14,
        fontweight="bold",
    )
    
    labels = df[label_col].dropna()
    total = len(labels)
    counts = labels.value_counts().sort_index()
    num_classes = len(counts)
    
    # Use a colormap for multi-class
    cmap_name = "tab20" if num_classes <= 20 else "viridis"
    cmap = plt.colormaps.get_cmap(cmap_name)
    colors = [cmap(i / num_classes) for i in range(num_classes)]
    
    # 1. Class distribution bar chart
    ax1 = fig.add_subplot(gs[0, 0:2])
    
    # Create labels for x-axis (use place names for location tasks)
    if "location" in task_def.task_id and task_def.num_classes == 23:
        x_labels = [IDX_TO_PLACE.get(int(i), f"Class {int(i)}") for i in counts.index]
    else:
        x_labels = [f"{int(i)}" for i in counts.index]
    
    bars = ax1.bar(
        range(len(counts)),
        counts.values,
        color=colors[: len(counts)],
        edgecolor="black",
        linewidth=0.3,
    )
    ax1.set_xticks(range(len(counts)))
    ax1.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax1.set_title(f"Class Distribution ({num_classes} classes)")
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Class")
    
    # Add min/max/mean annotation
    ax1.axhline(
        y=np.mean(counts.values),
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Mean: {np.mean(counts.values):.0f}",
    )
    ax1.legend(loc="upper right")
    
    # Imbalance stats
    imbalance_ratio = max(counts.values) / min(counts.values) if min(counts.values) > 0 else float("inf")
    ax1.text(
        0.02,
        0.95,
        f"Imbalance: {imbalance_ratio:.2f}:1\nMin: {min(counts.values):,}\nMax: {max(counts.values):,}",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    
    # 2. Distribution by side (stacked)
    ax2 = fig.add_subplot(gs[0, 2])
    if "pov_side" in df.columns:
        side_counts = df.groupby("pov_side")[label_col].value_counts().unstack(fill_value=0)
        side_counts.T.plot(kind="bar", ax=ax2, stacked=True, edgecolor="black", linewidth=0.3, width=0.8)
        ax2.set_title("Distribution by POV Side")
        ax2.set_xlabel("Class")
        ax2.set_ylabel("Count")
        ax2.tick_params(axis="x", rotation=45)
        ax2.legend(title="Side")
    else:
        ax2.text(0.5, 0.5, "No pov_side column", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Distribution by POV Side")
    
    # 3. Distribution by partition
    ax3 = fig.add_subplot(gs[1, 0])
    if "partition" in df.columns:
        partition_counts = df.groupby("partition")[label_col].value_counts().unstack(fill_value=0)
        partition_counts.plot(kind="bar", ax=ax3, edgecolor="black", linewidth=0.3, width=0.8)
        ax3.set_title("Distribution by Partition")
        ax3.set_xlabel("Partition")
        ax3.set_ylabel("Count")
        ax3.tick_params(axis="x", rotation=0)
        ax3.legend(title="Class", fontsize=6, ncol=2)
    else:
        ax3.text(0.5, 0.5, "No partition column", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Distribution by Partition")
    
    # 4. Temporal distribution (start tick histogram per class)
    ax4 = fig.add_subplot(gs[1, 1])
    if "start_tick" in df.columns:
        # Show distribution of start_tick (aggregate, not per-class for clarity)
        ax4.hist(
            df["start_tick"].dropna(),
            bins=40,
            color="steelblue",
            edgecolor="black",
            linewidth=0.3,
            alpha=0.7,
        )
        ax4.set_title("Temporal Distribution (Start Tick)")
        ax4.set_xlabel("Start Tick")
        ax4.set_ylabel("Count")
    else:
        ax4.text(0.5, 0.5, "No start_tick column", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("Temporal Distribution")
    
    # 5. Stats summary
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    
    stats_text = f"""Summary Statistics
─────────────────────────
Total Samples: {total:,}
Num Classes: {num_classes}
Expected Classes: {task_def.num_classes}

Class Counts
─────────────────────────
Min: {min(counts.values):,}
Max: {max(counts.values):,}
Mean: {np.mean(counts.values):,.1f}
Std: {np.std(counts.values):,.1f}

Task Info
─────────────────────────
Category: {task_def.category.value}
Temporal: {task_def.temporal_type.value}
Horizon: {task_def.horizon_sec}s
Data Source: {task_def.primary_data_source.value}
"""
    ax5.text(
        0.1,
        0.95,
        stats_text,
        transform=ax5.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3),
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_multi_label_cls(
    df: pd.DataFrame,
    task_def: TaskDefinition,
    output_path: Path,
):
    """
    Plot multi-label classification task distribution.
    
    Creates a figure with:
    - Per-label positive rate
    - Labels per sample distribution
    - Label co-occurrence heatmap
    - Distribution by pov_side
    - Temporal distribution
    """
    # Find label columns
    label_cols = sorted([c for c in df.columns if c.startswith("label_")])
    
    if not label_cols:
        print(f"  Warning: No label columns found for {task_def.task_id}")
        return
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    fig.suptitle(
        f"{task_def.task_name} ({task_def.task_id})\n{task_def.description}",
        fontsize=14,
        fontweight="bold",
    )
    
    total = len(df)
    label_matrix = df[label_cols].values
    
    # 1. Per-label positive rate
    ax1 = fig.add_subplot(gs[0, 0:2])
    positive_counts = (label_matrix == 1).sum(axis=0)
    positive_rates = 100 * positive_counts / total
    
    # Create labels (use place names for location tasks)
    if "location" in task_def.task_id and len(label_cols) == 23:
        x_labels = [IDX_TO_PLACE.get(i, f"Label {i}") for i in range(len(label_cols))]
    else:
        x_labels = [f"{i}" for i in range(len(label_cols))]
    
    cmap = plt.colormaps.get_cmap("viridis")
    colors = [cmap(r / 100) for r in positive_rates]
    
    bars = ax1.bar(
        range(len(label_cols)),
        positive_rates,
        color=colors,
        edgecolor="black",
        linewidth=0.3,
    )
    ax1.set_xticks(range(len(label_cols)))
    ax1.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax1.set_title(f"Per-Label Positive Rate ({len(label_cols)} labels)")
    ax1.set_ylabel("Positive Rate (%)")
    ax1.set_xlabel("Label Index")
    ax1.axhline(y=np.mean(positive_rates), color="red", linestyle="--", linewidth=1, label=f"Mean: {np.mean(positive_rates):.1f}%")
    ax1.legend(loc="upper right")
    
    # 2. Labels per sample distribution
    ax2 = fig.add_subplot(gs[0, 2])
    labels_per_sample = np.sum(label_matrix == 1, axis=1)
    unique_counts, freq = np.unique(labels_per_sample, return_counts=True)
    
    ax2.bar(unique_counts, freq, color="steelblue", edgecolor="black", linewidth=0.5)
    ax2.set_title("Labels per Sample Distribution")
    ax2.set_xlabel("Number of Active Labels")
    ax2.set_ylabel("Count")
    ax2.text(
        0.95,
        0.95,
        f"Mean: {np.mean(labels_per_sample):.2f}\nStd: {np.std(labels_per_sample):.2f}",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    
    # 3. Label co-occurrence heatmap (subset if too many labels)
    ax3 = fig.add_subplot(gs[1, 0])
    n_labels = len(label_cols)
    if n_labels <= 25:
        co_occurrence = np.dot(label_matrix.T, label_matrix)
        # Normalize by diagonal (self-occurrence)
        diag = np.diag(co_occurrence).copy()
        diag[diag == 0] = 1  # Avoid division by zero
        co_occurrence_norm = co_occurrence / diag[:, None]
        
        im = ax3.imshow(co_occurrence_norm, cmap="YlOrRd", aspect="auto")
        ax3.set_title("Label Co-occurrence (Normalized)")
        ax3.set_xlabel("Label Index")
        ax3.set_ylabel("Label Index")
        plt.colorbar(im, ax=ax3, shrink=0.8)
    else:
        ax3.text(0.5, 0.5, f"Too many labels ({n_labels})\nfor co-occurrence plot", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Label Co-occurrence")
    
    # 4. Distribution by pov_side
    ax4 = fig.add_subplot(gs[1, 1])
    if "pov_side" in df.columns:
        side_groups = df.groupby("pov_side")
        side_stats = {}
        for side, group in side_groups:
            side_matrix = group[label_cols].values
            side_labels_per_sample = np.sum(side_matrix == 1, axis=1)
            side_stats[side] = {
                "count": len(group),
                "mean_labels": np.mean(side_labels_per_sample),
            }
        
        sides = list(side_stats.keys())
        means = [side_stats[s]["mean_labels"] for s in sides]
        counts = [side_stats[s]["count"] for s in sides]
        
        x_pos = np.arange(len(sides))
        width = 0.35
        
        ax4_twin = ax4.twinx()
        bars1 = ax4.bar(x_pos - width / 2, counts, width, label="Sample Count", color="#3498db", edgecolor="black", linewidth=0.5)
        bars2 = ax4_twin.bar(x_pos + width / 2, means, width, label="Avg Labels/Sample", color="#e74c3c", edgecolor="black", linewidth=0.5)
        
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(sides)
        ax4.set_xlabel("POV Side")
        ax4.set_ylabel("Sample Count", color="#3498db")
        ax4_twin.set_ylabel("Avg Labels per Sample", color="#e74c3c")
        ax4.set_title("Distribution by POV Side")
        ax4.legend(loc="upper left")
        ax4_twin.legend(loc="upper right")
    else:
        ax4.text(0.5, 0.5, "No pov_side column", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("Distribution by POV Side")
    
    # 5. Stats summary
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    
    stats_text = f"""Summary Statistics
─────────────────────────────
Total Samples: {total:,}
Num Labels: {len(label_cols)}

Per-Label Positive Rate
─────────────────────────────
Min: {min(positive_rates):.2f}%
Max: {max(positive_rates):.2f}%
Mean: {np.mean(positive_rates):.2f}%

Labels per Sample
─────────────────────────────
Min: {int(labels_per_sample.min())}
Max: {int(labels_per_sample.max())}
Mean: {labels_per_sample.mean():.2f}
Std: {labels_per_sample.std():.2f}

Task Info
─────────────────────────────
Category: {task_def.category.value}
Temporal: {task_def.temporal_type.value}
Horizon: {task_def.horizon_sec}s
"""
    ax5.text(
        0.05,
        0.95,
        stats_text,
        transform=ax5.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3),
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_regression(
    df: pd.DataFrame,
    task_def: TaskDefinition,
    output_path: Path,
):
    """
    Plot regression task distribution.
    
    Creates a figure with:
    - Value histogram
    - Box plot
    - Distribution by pov_side
    - Temporal distribution (value vs start_tick)
    """
    # Find label column(s)
    if "label" in df.columns:
        label_cols = ["label"]
    else:
        label_cols = sorted([c for c in df.columns if c.startswith("label_")])
    
    if not label_cols:
        print(f"  Warning: No label columns found for {task_def.task_id}")
        return
    
    # For multi-output regression, we'll plot the first few outputs
    n_outputs = min(len(label_cols), 4)
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle(
        f"{task_def.task_name} ({task_def.task_id})\n{task_def.description}",
        fontsize=14,
        fontweight="bold",
    )
    
    total = len(df)
    
    # 1. Value histogram (first output or all combined)
    ax1 = fig.add_subplot(gs[0, 0])
    primary_label = label_cols[0]
    values = df[primary_label].dropna()
    
    ax1.hist(
        values,
        bins=50,
        color="steelblue",
        edgecolor="black",
        linewidth=0.3,
        alpha=0.7,
    )
    ax1.axvline(values.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {values.mean():.2f}")
    ax1.axvline(values.median(), color="orange", linestyle="-.", linewidth=1.5, label=f"Median: {values.median():.2f}")
    ax1.set_title(f"Value Distribution ({primary_label})")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=8)
    
    # 2. Box plot for all outputs
    ax2 = fig.add_subplot(gs[0, 1])
    if n_outputs > 1:
        box_data = [df[col].dropna().values for col in label_cols[:n_outputs]]
        bp = ax2.boxplot(box_data, tick_labels=[f"Out {i}" for i in range(n_outputs)], patch_artist=True)
        colors = plt.cm.tab10.colors[:n_outputs]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax2.set_title(f"Output Distribution (first {n_outputs})")
    else:
        bp = ax2.boxplot([values.values], tick_labels=[primary_label], patch_artist=True)
        bp["boxes"][0].set_facecolor("steelblue")
        bp["boxes"][0].set_alpha(0.6)
        ax2.set_title("Value Box Plot")
    ax2.set_ylabel("Value")
    
    # 3. Distribution by pov_side
    ax3 = fig.add_subplot(gs[0, 2])
    if "pov_side" in df.columns:
        sides = df["pov_side"].unique()
        box_data = [df[df["pov_side"] == side][primary_label].dropna().values for side in sorted(sides)]
        bp = ax3.boxplot(box_data, tick_labels=sorted(sides), patch_artist=True)
        colors = ["#3498db", "#e74c3c"]
        for patch, color in zip(bp["boxes"], colors[: len(sides)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax3.set_title("Distribution by POV Side")
        ax3.set_xlabel("Side")
        ax3.set_ylabel("Value")
    else:
        ax3.text(0.5, 0.5, "No pov_side column", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Distribution by POV Side")
    
    # 4. Temporal distribution (scatter of value vs start_tick)
    ax4 = fig.add_subplot(gs[1, 0])
    if "start_tick" in df.columns:
        sample_size = min(5000, len(df))
        sample_df = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
        ax4.scatter(
            sample_df["start_tick"],
            sample_df[primary_label],
            alpha=0.3,
            s=5,
            c="steelblue",
        )
        ax4.set_title(f"Value vs Time (n={sample_size:,})")
        ax4.set_xlabel("Start Tick")
        ax4.set_ylabel("Value")
    else:
        ax4.text(0.5, 0.5, "No start_tick column", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("Temporal Distribution")
    
    # 5. Distribution by partition
    ax5 = fig.add_subplot(gs[1, 1])
    if "partition" in df.columns:
        partitions = df["partition"].unique()
        box_data = [df[df["partition"] == p][primary_label].dropna().values for p in sorted(partitions)]
        bp = ax5.boxplot(box_data, tick_labels=sorted(partitions), patch_artist=True)
        colors = ["#2ecc71", "#9b59b6", "#f39c12"]
        for patch, color in zip(bp["boxes"], colors[: len(partitions)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax5.set_title("Distribution by Partition")
        ax5.set_xlabel("Partition")
        ax5.set_ylabel("Value")
    else:
        ax5.text(0.5, 0.5, "No partition column", ha="center", va="center", transform=ax5.transAxes)
        ax5.set_title("Distribution by Partition")
    
    # 6. Stats summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    
    stats_text = f"""Summary Statistics
─────────────────────────
Total Samples: {total:,}
Output Dimensions: {len(label_cols)}

{primary_label} Statistics
─────────────────────────
Min: {values.min():.4f}
Max: {values.max():.4f}
Mean: {values.mean():.4f}
Std: {values.std():.4f}
Median: {values.median():.4f}
Q25: {values.quantile(0.25):.4f}
Q75: {values.quantile(0.75):.4f}

Task Info
─────────────────────────
Category: {task_def.category.value}
Temporal: {task_def.temporal_type.value}
Horizon: {task_def.horizon_sec}s
"""
    ax6.text(
        0.05,
        0.95,
        stats_text,
        transform=ax6.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3),
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_task_distribution(
    csv_path: Path,
    task_def: TaskDefinition,
    output_path: Path,
):
    """
    Plot label distribution for a single task.
    
    Dispatches to the appropriate plotting function based on ml_form.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  Error reading {csv_path}: {e}")
        return
    
    ml_form = task_def.ml_form
    
    if ml_form == MLForm.BINARY_CLS:
        plot_binary_cls(df, task_def, output_path)
    elif ml_form == MLForm.MULTI_CLS:
        plot_multi_cls(df, task_def, output_path)
    elif ml_form == MLForm.MULTI_LABEL_CLS:
        plot_multi_label_cls(df, task_def, output_path)
    elif ml_form == MLForm.REGRESSION:
        plot_regression(df, task_def, output_path)
    else:
        print(f"  Warning: Unknown ml_form '{ml_form}' for {task_def.task_id}")


def main():
    parser = argparse.ArgumentParser(description="Plot label distribution for each task")
    parser.add_argument(
        "--labels_dir",
        type=str,
        default=None,
        help="Directory containing label CSVs (default: DATA_BASE_PATH/labels/all_tasks)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: artifacts/label_distribution)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Specific task IDs to plot (default: all)",
    )
    args = parser.parse_args()
    
    load_dotenv()
    
    DATA_BASE_PATH = os.getenv("DATA_BASE_PATH")
    if not DATA_BASE_PATH:
        print("ERROR: DATA_BASE_PATH environment variable not set")
        sys.exit(1)
    
    # Set up paths
    labels_dir = args.labels_dir
    if labels_dir is None:
        labels_dir = Path(DATA_BASE_PATH) / "labels" / "all_tasks"
    else:
        labels_dir = Path(labels_dir)
    
    output_dir = args.output_dir
    if output_dir is None:
        # Use workspace artifacts folder
        output_dir = Path(__file__).parents[3] / "artifacts" / "label_distribution"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not labels_dir.exists():
        print(f"ERROR: Labels directory not found: {labels_dir}")
        sys.exit(1)
    
    # Load task definitions
    try:
        task_defs = load_task_definitions()
        task_def_map = {t.task_id: t for t in task_defs}
    except Exception as e:
        print(f"ERROR: Could not load task definitions: {e}")
        sys.exit(1)
    
    print(f"Labels directory: {labels_dir}")
    print(f"Output directory: {output_dir}")
    
    # Set up matplotlib style
    setup_style()
    
    # Find all CSV files
    csv_files = sorted(labels_dir.glob("*.csv"))
    
    if args.tasks:
        csv_files = [f for f in csv_files if f.stem in args.tasks]
    
    if not csv_files:
        print("No CSV files found")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV files")
    print()
    
    # Process each task
    for csv_file in csv_files:
        task_id = csv_file.stem
        task_def = task_def_map.get(task_id)
        
        if task_def is None:
            print(f"  Warning: No task definition for {task_id}, skipping")
            continue
        
        output_path = output_dir / f"{task_id}.png"
        print(f"  Plotting: {task_id} ({task_def.ml_form.value})...")
        
        plot_task_distribution(csv_file, task_def, output_path)
    
    print()
    print(f"Done! Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
