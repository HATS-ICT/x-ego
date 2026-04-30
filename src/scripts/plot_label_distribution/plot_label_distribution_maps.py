"""Plot benchmark task label distributions, one image per map."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("artifacts") / "label_distribition_maps"
METADATA_COLUMNS = {
    "idx",
    "partition",
    "pov_steamid",
    "pov_side",
    "seg_duration_sec",
    "horizon_sec",
    "start_tick",
    "end_tick",
    "prediction_tick",
    "start_tick_norm",
    "end_tick_norm",
    "prediction_tick_norm",
    "match_id",
    "round_num",
    "map_name",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one label-distribution image per map for benchmark tasks."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Directory containing map folders. Default: data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output PNGs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--maps",
        nargs="*",
        help="Optional list of map folder names to plot. Defaults to all maps with labels/task_definitions.csv.",
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=30,
        help="Maximum number of categorical bars to show per subplot.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=30,
        help="Number of bins for regression histograms.",
    )
    return parser.parse_args()


def discover_map_dirs(data_root: Path, map_names: list[str] | None) -> list[Path]:
    if map_names:
        candidates = [data_root / map_name for map_name in map_names]
    else:
        candidates = sorted(path for path in data_root.iterdir() if path.is_dir())

    return [
        path
        for path in candidates
        if (path / "labels" / "task_definitions.csv").is_file()
        and (path / "labels" / "all_tasks").is_dir()
    ]


def load_benchmark_tasks(task_definitions_path: Path) -> pd.DataFrame:
    task_defs = pd.read_csv(task_definitions_path)
    use_col = task_defs["use_in_benchmark"].fillna("").astype(str).str.lower()
    return task_defs.loc[use_col.eq("yes")].copy()


def label_columns(df: pd.DataFrame) -> list[str]:
    if "label" in df.columns:
        return ["label"]

    columns = [col for col in df.columns if col.startswith("label_")]
    if columns:
        return sorted(columns, key=lambda col: int(col.split("_", 1)[1]))

    return [col for col in df.columns if col not in METADATA_COLUMNS]


def format_count(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}m"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(int(value))


def annotate_bars(ax: plt.Axes, bars, total: float) -> None:
    if total <= 0:
        return

    ymax = max((bar.get_height() for bar in bars), default=0)
    for bar in bars:
        height = bar.get_height()
        pct = 100 * height / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(ymax * 0.015, 0.5),
            f"{pct:.0f}%",
            ha="center",
            va="bottom",
            fontsize=6,
            rotation=90,
        )


def plot_categorical(
    ax: plt.Axes,
    series: pd.Series,
    task_id: str,
    ml_form: str,
    max_categories: int,
) -> None:
    values = series.dropna()
    counts = values.value_counts().sort_index()
    if len(counts) > max_categories:
        counts = counts.sort_values(ascending=False).head(max_categories)

    labels = [str(int(idx)) if isinstance(idx, (int, np.integer, float)) and float(idx).is_integer() else str(idx) for idx in counts.index]
    bars = ax.bar(labels, counts.values, color="#4c78a8", linewidth=0)
    annotate_bars(ax, bars, float(counts.sum()))
    ax.set_title(f"{task_id}\n{ml_form} n={format_count(len(values))}", fontsize=8)
    ax.set_ylabel("count", fontsize=7)
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    ax.tick_params(axis="y", labelsize=6)

    if len(counts) > 1 and counts.min() > 0:
        ratio = counts.max() / counts.min()
        ax.text(
            0.98,
            0.92,
            f"{ratio:.1f}:1",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "linewidth": 0},
        )


def plot_multilabel(
    ax: plt.Axes,
    df: pd.DataFrame,
    label_cols: list[str],
    task_id: str,
    max_categories: int,
) -> None:
    positives = df[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum()
    positives.index = [col.removeprefix("label_") for col in positives.index]
    positives = positives.sort_index(key=lambda idx: idx.astype(int))
    if len(positives) > max_categories:
        positives = positives.sort_values(ascending=False).head(max_categories)

    bars = ax.bar(positives.index.astype(str), positives.values, color="#59a14f", linewidth=0)
    total_assignments = float(positives.sum())
    annotate_bars(ax, bars, total_assignments)
    per_sample = total_assignments / len(df) if len(df) else 0
    ax.set_title(
        f"{task_id}\nmulti_label_cls n={format_count(len(df))} avg_pos={per_sample:.2f}",
        fontsize=8,
    )
    ax.set_ylabel("positive count", fontsize=7)
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    ax.tick_params(axis="y", labelsize=6)


def plot_regression(
    ax: plt.Axes,
    df: pd.DataFrame,
    label_cols: list[str],
    task_id: str,
    bins: int,
) -> None:
    values = (
        df[label_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy()
        .reshape(-1)
    )
    values = values[~np.isnan(values)]
    ax.hist(values, bins=bins, color="#f28e2b", edgecolor="white", linewidth=0.4)
    ax.set_title(f"{task_id}\nregression n={format_count(len(values))}", fontsize=8)
    ax.set_ylabel("count", fontsize=7)
    ax.tick_params(axis="x", labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    if len(values):
        ax.text(
            0.98,
            0.92,
            f"mean={values.mean():.2f}\nstd={values.std():.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "linewidth": 0},
        )


def plot_task(
    ax: plt.Axes,
    labels_dir: Path,
    task: pd.Series,
    max_categories: int,
    hist_bins: int,
) -> None:
    task_id = str(task["task_id"])
    ml_form = str(task["ml_form"])
    csv_path = labels_dir / f"{task_id}.csv"

    if not csv_path.is_file():
        ax.set_axis_off()
        ax.set_title(f"{task_id}\nmissing csv", fontsize=8)
        return

    df = pd.read_csv(csv_path)
    cols = label_columns(df)
    if not cols:
        ax.set_axis_off()
        ax.set_title(f"{task_id}\nno label columns", fontsize=8)
        return

    if ml_form == "multi_label_cls":
        plot_multilabel(ax, df, cols, task_id, max_categories)
    elif ml_form == "regression":
        plot_regression(ax, df, cols, task_id, hist_bins)
    else:
        label_values = df[cols[0]]
        numeric_values = pd.to_numeric(label_values, errors="coerce")
        if numeric_values.notna().sum() == label_values.notna().sum():
            label_values = numeric_values
        plot_categorical(ax, label_values, task_id, ml_form, max_categories)


def plot_map_distribution(
    map_dir: Path,
    output_dir: Path,
    max_categories: int,
    hist_bins: int,
) -> Path | None:
    labels_root = map_dir / "labels"
    labels_dir = labels_root / "all_tasks"
    tasks = load_benchmark_tasks(labels_root / "task_definitions.csv")
    if tasks.empty:
        return None

    ncols = 4
    nrows = math.ceil(len(tasks) / ncols)
    fig_width = 4.2 * ncols
    fig_height = 3.0 * nrows + 1.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
    fig.suptitle(
        f"{map_dir.name} benchmark label distributions ({len(tasks)} tasks)",
        fontsize=16,
        fontweight="bold",
    )

    for ax, (_, task) in zip(axes.flat, tasks.iterrows()):
        plot_task(ax, labels_dir, task, max_categories, hist_bins)

    for ax in axes.flat[len(tasks) :]:
        ax.set_axis_off()

    fig.tight_layout(rect=(0, 0, 1, 0.985))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{map_dir.name}_label_distribution.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    map_dirs = discover_map_dirs(args.data_root, args.maps)
    if not map_dirs:
        raise FileNotFoundError(f"No map label directories found under {args.data_root}")

    for map_dir in map_dirs:
        output_path = plot_map_distribution(
            map_dir=map_dir,
            output_dir=args.output_dir,
            max_categories=args.max_categories,
            hist_bins=args.hist_bins,
        )
        if output_path is not None:
            print(f"saved {output_path}")


if __name__ == "__main__":
    main()
