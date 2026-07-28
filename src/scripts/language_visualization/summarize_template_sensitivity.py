"""
Summarize the prompt-template sensitivity sweep.

Consumes the per-map CSVs written by language_visualization.py under
<sweep-dir>/<map>/template_sensitivity/ and answers, quantitatively, how sensitive the
semantic-drift ranking trajectories are to the prompt template:

1. Headline robustness table: per group, the distribution of the rank change across the
   43 singleton templates, with sign-consistency and how many arms clear the
   permutation null.
2. Per-family breakdown, so sensitivity attributable to one odd prompt family is visible
   rather than averaged away.
3. The five rebuttal-named templates as a labelled subset, in the Appendix Table 18 layout.
4. Variance decomposition: how much of the group-change variance is template choice vs
   map choice vs sampling noise.
5. Spearman rho distribution over all template pairs.
6. A box/strip figure of the per-template changes with the null band shaded.

Usage:
    python src/scripts/language_visualization/summarize_template_sensitivity.py \
        --sweep-dir artifacts/lv_tmpl_sweep
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.scripts.language_visualization.concept_vocabulary import GROUP_COLORS
from src.scripts.language_visualization.prompt_templates import (
    ENSEMBLE_TEMPLATE_KEYS,
    FAMILY_LABELS,
    REBUTTAL_TEMPLATES,
    SINGLETON_TEMPLATE_KEYS,
)

GROUP_ORDER = ["egocentric", "teammate", "enemy", "global", "spatial"]
ALPHA = 0.05


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--sweep-dir", default=str(Path("artifacts") / "lv_tmpl_sweep"))
    parser.add_argument("--out-dir", default=None,
                        help="Where to write the summary. Defaults to <sweep-dir>/summary.")
    parser.add_argument("--maps", nargs="*", default=None,
                        help="Restrict to these maps. Default: every map found.")
    return parser.parse_args()


def load_sweep(sweep_dir: Path, wanted_maps=None):
    """Concatenate the per-map CSVs. Returns (group, concept, agreement, trajectory)."""
    frames = {"group_changes": [], "concept_changes": [],
              "template_agreement": [], "trajectory_changes": []}

    map_dirs = sorted(p for p in sweep_dir.iterdir() if (p / "template_sensitivity").is_dir())
    if wanted_maps:
        map_dirs = [p for p in map_dirs if p.name in wanted_maps]
    if not map_dirs:
        raise FileNotFoundError(
            f"No <map>/template_sensitivity/ directories under {sweep_dir}. "
            "Run scripts/run_template_sensitivity.sh first."
        )

    for map_dir in map_dirs:
        ts = map_dir / "template_sensitivity"
        for name in frames:
            path = ts / f"{name}.csv"
            if path.exists():
                frames[name].append(pl.read_csv(path))
            elif name in ("group_changes", "concept_changes"):
                raise FileNotFoundError(f"Missing required {path}")

    print(f"Loaded {len(map_dirs)} map(s): {', '.join(p.name for p in map_dirs)}")
    return tuple(
        pl.concat(frames[name], how="vertical_relaxed") if frames[name] else None
        for name in ("group_changes", "concept_changes", "template_agreement", "trajectory_changes")
    )


def singleton_only(df: pl.DataFrame) -> pl.DataFrame:
    """Restrict to the 43 singleton arms; ensembles are reported separately."""
    return df.filter(pl.col("template").is_in(SINGLETON_TEMPLATE_KEYS))


def headline_table(group_df: pl.DataFrame, by_map: bool) -> pl.DataFrame:
    """
    Distribution of the group rank change across templates.

    'change' is baseline mean rank minus final mean rank, so positive = the group rose
    (became more similar to the video embeddings).
    """
    keys = ["map", "group"] if by_map else ["group"]
    agg = (
        singleton_only(group_df)
        .group_by(keys)
        .agg(
            pl.len().alias("n_templates"),
            pl.col("change").mean().alias("mean_change"),
            pl.col("change").std().alias("sd_change"),
            pl.col("change").min().alias("min_change"),
            pl.col("change").max().alias("max_change"),
            (pl.col("change") > 0).sum().alias("n_rose"),
            (pl.col("perm_p") < ALPHA).sum().alias("n_sig"),
            pl.col("n_samples").min().alias("n_samples"),
        )
    )
    order = pl.DataFrame({"group": GROUP_ORDER, "_order": range(len(GROUP_ORDER))})
    agg = agg.join(order, on="group").sort((["map"] if by_map else []) + ["_order"]).drop("_order")
    return agg.with_columns(
        pl.when(pl.col("n_rose") == pl.col("n_templates")).then(pl.lit("rose"))
        .when(pl.col("n_rose") == 0).then(pl.lit("fell"))
        .otherwise(pl.lit("MIXED")).alias("consistency")
    )


def family_table(group_df: pl.DataFrame) -> pl.DataFrame:
    order = pl.DataFrame({"group": GROUP_ORDER, "_order": range(len(GROUP_ORDER))})
    return (
        singleton_only(group_df)
        .group_by(["group", "family"])
        .agg(
            pl.len().alias("n"),
            pl.col("change").mean().alias("mean_change"),
            pl.col("change").std().alias("sd_change"),
            (pl.col("change") > 0).sum().alias("n_rose"),
        )
        .join(order, on="group")
        .sort(["_order", "family"])
        .drop("_order")
    )


def variance_decomposition(group_df: pl.DataFrame) -> pl.DataFrame:
    """
    Split the variance of the group-change statistic into template, map, and residual.

    Two-way ANOVA-style decomposition per group over the template x map grid, plus the
    mean within-cell bootstrap variance (sampling noise) for scale. The question the
    reviewer is really asking is whether prompt choice moves the result more than the
    other sources of variation, and this answers it directly.
    """
    df = singleton_only(group_df)
    rows = []
    for group in GROUP_ORDER:
        g = df.filter(pl.col("group") == group)
        if g.height == 0:
            continue

        grand = g["change"].mean()
        total_ss = float(((g["change"] - grand) ** 2).sum())

        def factor_ss(col):
            means = g.group_by(col).agg(
                pl.col("change").mean().alias("m"), pl.len().alias("n")
            )
            return float((means["n"] * (means["m"] - grand) ** 2).sum())

        tmpl_ss = factor_ss("template")
        map_ss = factor_ss("map") if g["map"].n_unique() > 1 else 0.0
        resid_ss = max(total_ss - tmpl_ss - map_ss, 0.0)

        boot_var = None
        if "boot_sd" in g.columns and g["boot_sd"].null_count() < g.height:
            boot_var = float(np.nanmean(g["boot_sd"].to_numpy().astype(float) ** 2))

        rows.append({
            "group": group,
            "total_sd": float(np.sqrt(total_ss / max(g.height - 1, 1))),
            "template_share": tmpl_ss / total_ss if total_ss > 0 else float("nan"),
            "map_share": map_ss / total_ss if total_ss > 0 else float("nan"),
            "residual_share": resid_ss / total_ss if total_ss > 0 else float("nan"),
            "sampling_sd": float(np.sqrt(boot_var)) if boot_var is not None else None,
        })
    return pl.DataFrame(rows)


def agreement_summary(agreement_df) -> pl.DataFrame:
    if agreement_df is None or agreement_df.height == 0:
        return pl.DataFrame()
    df = agreement_df.filter(
        pl.col("template_a").is_in(SINGLETON_TEMPLATE_KEYS)
        & pl.col("template_b").is_in(SINGLETON_TEMPLATE_KEYS)
    )
    return (
        df.group_by("map")
        .agg(
            pl.len().alias("n_pairs"),
            pl.col("spearman_rho").median().alias("median_rho"),
            pl.col("spearman_rho").quantile(0.25).alias("q25_rho"),
            pl.col("spearman_rho").quantile(0.75).alias("q75_rho"),
            pl.col("spearman_rho").min().alias("min_rho"),
        )
        .sort("map")
    )


def rebuttal_subset(group_df: pl.DataFrame) -> pl.DataFrame:
    order = pl.DataFrame({"group": GROUP_ORDER, "_order": range(len(GROUP_ORDER))})
    tmpl_order = pl.DataFrame(
        {"template": REBUTTAL_TEMPLATES, "_torder": range(len(REBUTTAL_TEMPLATES))}
    )
    return (
        group_df.filter(pl.col("template").is_in(REBUTTAL_TEMPLATES))
        .join(order, on="group").join(tmpl_order, on="template")
        .sort(["map", "_torder", "_order"])
        .select(["map", "template", "group", "n_samples", "baseline_mean", "final_mean",
                 "change", "ci_lo", "ci_hi", "perm_p", "direction"])
    )


def ensemble_subset(group_df: pl.DataFrame) -> pl.DataFrame:
    order = pl.DataFrame({"group": GROUP_ORDER, "_order": range(len(GROUP_ORDER))})
    return (
        group_df.filter(pl.col("template").is_in(ENSEMBLE_TEMPLATE_KEYS))
        .join(order, on="group")
        .sort(["map", "template", "_order"])
        .select(["map", "template", "group", "change", "ci_lo", "ci_hi", "perm_p", "direction"])
    )


# The paper's textwidth is 5.5in. Build the figure at that size with print-scale fonts so
# \includegraphics[width=\linewidth] is a 1:1 placement. Emitting a 15in-wide figure and
# letting LaTeX shrink it by 2.8x is what made the existing appendix figures illegible.
PRINT_WIDTH_IN = 5.5
MAP_HATCHES = {0: "", 1: "///", 2: "..."}


def plot_box(group_df: pl.DataFrame, save_path: Path) -> None:
    """
    Per-group distribution of the change across templates, maps side by side.

    One panel rather than one panel per map: with three panels at 5.5in total each axis is
    under 1.9in and the tick labels stop being readable. Grouping the maps within each
    concept group keeps a single legible axis and makes the map comparison direct.
    """
    df = singleton_only(group_df)
    maps = sorted(df["map"].unique().to_list())
    n_maps = len(maps)

    fig, ax = plt.subplots(figsize=(PRINT_WIDTH_IN, 2.9))
    rng = np.random.default_rng(0)
    slot = 0.8 / n_maps  # horizontal room per map within a group

    for gi, group in enumerate(GROUP_ORDER):
        for mi, map_name in enumerate(maps):
            cell = df.filter((pl.col("group") == group) & (pl.col("map") == map_name))
            if cell.height == 0:
                continue
            values = cell["change"].to_numpy()
            pos = gi + (mi - (n_maps - 1) / 2) * slot

            # Shade the permutation null band: a distribution inside it is within chance.
            if "null_lo" in cell.columns and cell["null_lo"].null_count() < cell.height:
                lo = float(np.nanmean(cell["null_lo"].to_numpy().astype(float)))
                hi = float(np.nanmean(cell["null_hi"].to_numpy().astype(float)))
                ax.fill_between([pos - slot * 0.46, pos + slot * 0.46], lo, hi,
                                color="0.78", alpha=0.6, linewidth=0, zorder=1)

            bp = ax.boxplot([values], positions=[pos], widths=slot * 0.72,
                            patch_artist=True, manage_ticks=False,
                            medianprops=dict(color="black", linewidth=1.0),
                            boxprops=dict(linewidth=0.6),
                            whiskerprops=dict(linewidth=0.6),
                            capprops=dict(linewidth=0.6),
                            flierprops=dict(marker="", markersize=0), zorder=2)
            bp["boxes"][0].set_facecolor(GROUP_COLORS[group])
            bp["boxes"][0].set_alpha(0.5)
            bp["boxes"][0].set_hatch(MAP_HATCHES.get(mi, ""))

            # Every template as a point, so n is visible rather than implied.
            jitter = rng.uniform(-slot * 0.2, slot * 0.2, size=len(values))
            ax.scatter(pos + jitter, values, s=3.0, color=GROUP_COLORS[group],
                       edgecolor="black", linewidth=0.15, alpha=0.85, zorder=3)

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.7)
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels([g.title() for g in GROUP_ORDER], fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlim(-0.5, len(GROUP_ORDER) - 0.5)
    ax.set_ylabel("Rank change\n(positive = rose)", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Map legend by hatch, colour already encodes the concept group.
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                      linewidth=0.6, hatch=MAP_HATCHES.get(i, ""), label=m)
        for i, m in enumerate(maps)
    ]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="0.78", edgecolor="none",
                                 label="chance band"))
    ax.legend(handles=handles, fontsize=6.5, ncol=len(handles), loc="upper center",
              bbox_to_anchor=(0.5, 1.16), frameon=False, handlelength=1.4,
              columnspacing=1.2, handletextpad=0.5)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(save_path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {save_path.with_suffix('.pdf')} "
          f"({PRINT_WIDTH_IN}in wide, sized for 1:1 placement)")


def latex_headline(pooled: pl.DataFrame, n_templates: int, maps: list) -> str:
    lines = [
        "\\begin{table}[!ht]",
        "\\centering",
        f"\\caption{{Prompt-template sensitivity of the group-level semantic ranking change. "
        f"Each group's change is measured under {n_templates} prompt templates spanning seven "
        f"families, on {len(maps)} maps ({', '.join(maps)}). Positive change means the group "
        f"rose in ranking. Consistency counts the templates in which the group moved in the "
        f"majority direction; significance is against a permutation null over "
        f"concept-to-group assignment at $\\alpha={ALPHA}$.}}",
        "\\label{tab:prompt_sensitivity}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "\\textbf{Group} & \\textbf{Mean $\\Delta$} & \\textbf{SD} & \\textbf{[Min, Max]} & "
        "\\textbf{Consistent} & \\textbf{Sig.} \\\\",
        "\\midrule",
    ]
    for row in pooled.iter_rows(named=True):
        n = row["n_templates"]
        n_dir = row["n_rose"] if row["mean_change"] > 0 else n - row["n_rose"]
        arrow = "\\uparrow" if row["mean_change"] > 0 else "\\downarrow"
        mean = f"${arrow}$ {row['mean_change']:+.1f}".replace("+", "$+$").replace("-", "$-$")
        lines.append(
            f"{row['group'].title()} & {mean} & {row['sd_change']:.1f} & "
            f"[{row['min_change']:+.1f}, {row['max_change']:+.1f}] & "
            f"{n_dir}/{n} & {row['n_sig']}/{n} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def fmt(df: pl.DataFrame) -> str:
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200,
                   tbl_formatting="ASCII_FULL_CONDENSED", tbl_hide_dataframe_shape=True,
                   float_precision=3):
        return str(df)


def main():
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    out_dir = Path(args.out_dir) if args.out_dir else sweep_dir / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    group_df, concept_df, agreement_df, traj_df = load_sweep(sweep_dir, args.maps)
    maps = sorted(group_df["map"].unique().to_list())
    n_templates = singleton_only(group_df)["template"].n_unique()

    per_map = headline_table(group_df, by_map=True)
    pooled = headline_table(group_df, by_map=False)
    families = family_table(group_df)
    variance = variance_decomposition(group_df)
    agreement = agreement_summary(agreement_df)
    rebuttal = rebuttal_subset(group_df)
    ensembles = ensemble_subset(group_df)

    for name, df in [
        ("headline_per_map.csv", per_map),
        ("headline_pooled.csv", pooled),
        ("by_family.csv", families),
        ("variance_decomposition.csv", variance),
        ("agreement_summary.csv", agreement),
        ("rebuttal_templates.csv", rebuttal),
        ("ensembles.csv", ensembles),
    ]:
        if df is not None and df.height:
            df.write_csv(out_dir / name)

    plot_box(group_df, out_dir / "template_sensitivity_box")

    latex = latex_headline(pooled, n_templates, maps)
    (out_dir / "headline_table.tex").write_text(latex, encoding="utf-8")

    report = [
        "# Prompt-template sensitivity",
        "",
        f"Maps: {', '.join(maps)}. Singleton templates: {n_templates} "
        f"(+{len(ENSEMBLE_TEMPLATE_KEYS)} ensembles reported separately). "
        f"Clips per map: {group_df['n_samples'].min()}-{group_df['n_samples'].max()}.",
        "",
        "Change = baseline mean rank minus final mean rank, so positive means the group",
        "rose (became more similar to the video embeddings). Ranks run 1-250.",
        "",
        "## Headline, pooled over maps",
        "",
        fmt(pooled),
        "",
        "## Per map",
        "",
        fmt(per_map),
        "",
        "## By prompt family",
        "",
        fmt(families),
        "",
        "## Variance decomposition of the group change",
        "",
        "template_share / map_share / residual_share partition the variance of the change",
        "across the template x map grid. sampling_sd is the mean within-cell bootstrap SD",
        "over clips, for scale.",
        "",
        fmt(variance),
        "",
        "## Per-concept agreement between templates (Spearman rho on change vectors)",
        "",
        fmt(agreement),
        "",
        "## The five rebuttal-named templates",
        "",
        fmt(rebuttal),
        "",
        "## Ensembles",
        "",
        fmt(ensembles),
        "",
        "## LaTeX",
        "",
        "```latex",
        latex,
        "```",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report[:1] + report[2:26]))
    print(f"\nWrote summary to {out_dir}")


if __name__ == "__main__":
    main()
