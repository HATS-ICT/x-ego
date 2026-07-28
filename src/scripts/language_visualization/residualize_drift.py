"""
Baseline-conditioned control for the semantic-drift group statistic.

The group change used in the paper is change = baseline_rank - final_rank, aggregated
over a group of concepts. Two properties of that statistic make a raw group change hard
to interpret:

1. It is exactly zero-sum. Ranks are a permutation of 1..N, so the mean rank over all
   concepts is fixed and the five group changes must sum to zero. Some group has to fall.
2. It regresses to the mean. Concepts that start well-ranked have room only to fall and
   concepts that start poorly-ranked have room only to rise, which induces a strong
   positive corr(baseline_rank, change) purely mechanically.

Together these mean a group's raw change is largely predicted by its baseline rank, with
no content-specific reorganization required. This script quantifies how much of the
group effect survives once that is removed, two ways:

  residual   Regress change on baseline_rank across all concepts within each
             (map, template) cell, then report group means of the residuals. This is the
             group's movement relative to concepts that started equally well-ranked.
  strat_p    Permutation test that shuffles group labels only WITHIN baseline-rank
             strata, so the null preserves each group's baseline-rank profile.

Usage:
    python src/scripts/language_visualization/residualize_drift.py \
        --sweep-dir artifacts/lv_tmpl_sweep
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import polars as pl

from src.scripts.language_visualization.prompt_templates import SINGLETON_TEMPLATE_KEYS

GROUP_ORDER = ["egocentric", "teammate", "enemy", "global", "spatial"]
ALPHA = 0.05


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--sweep-dir", default=str(Path("artifacts") / "lv_tmpl_sweep"))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--strata", type=int, default=5,
                        help="Number of equal-count baseline-rank strata for the null.")
    parser.add_argument("--degree", type=int, default=2,
                        help="Polynomial degree for the change-on-baseline_rank fit.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_concepts(sweep_dir: Path) -> pl.DataFrame:
    frames = []
    for map_dir in sorted(p for p in sweep_dir.iterdir() if (p / "template_sensitivity").is_dir()):
        path = map_dir / "template_sensitivity" / "concept_changes.csv"
        if path.exists():
            frames.append(pl.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No concept_changes.csv under {sweep_dir}")
    df = pl.concat(frames, how="vertical_relaxed")
    # Singletons only; ensembles are averages of these and would double-count.
    return df.filter(pl.col("template").is_in(SINGLETON_TEMPLATE_KEYS))


def stratify(baseline: np.ndarray, n_strata: int) -> np.ndarray:
    """Equal-count strata by baseline rank."""
    order = np.argsort(baseline, kind="stable")
    strata = np.empty(len(baseline), dtype=int)
    for s, chunk in enumerate(np.array_split(order, n_strata)):
        strata[chunk] = s
    return strata


def analyze_cell(baseline: np.ndarray, change: np.ndarray, groups: np.ndarray,
                 degree: int, n_perm: int, n_strata: int,
                 rng: np.random.Generator) -> list:
    """Residualized group effect and stratified permutation p-value for one cell."""
    # Remove the baseline-rank trend. A quadratic captures the compression toward the
    # middle better than a line, since the pull is strongest at both extremes.
    coeffs = np.polyfit(baseline, change, degree)
    residual = change - np.polyval(coeffs, baseline)
    r2 = 1.0 - residual.var() / change.var() if change.var() > 0 else float("nan")

    strata = stratify(baseline, n_strata)
    rows = []
    for group in GROUP_ORDER:
        mask = groups == group
        observed = float(residual[mask].mean())
        raw = float(change[mask].mean())

        # Null: reshuffle group membership within baseline-rank strata, so a drawn
        # pseudo-group has the same baseline-rank profile as the real one.
        draws = np.empty(n_perm)
        stratum_counts = [(s, int((strata[mask] == s).sum())) for s in range(n_strata)]
        stratum_pool = {s: np.flatnonzero(strata == s) for s in range(n_strata)}
        for b in range(n_perm):
            picks = [
                rng.choice(stratum_pool[s], size=k, replace=False)
                for s, k in stratum_counts if k > 0
            ]
            draws[b] = residual[np.concatenate(picks)].mean()

        centre = draws.mean()
        extreme = np.abs(draws - centre) >= abs(observed - centre)
        rows.append({
            "group": group,
            "raw_change": raw,
            "baseline_mean": float(baseline[mask].mean()),
            "residual_change": observed,
            "strat_p": float((extreme.sum() + 1) / (n_perm + 1)),
            "null_sd": float(draws.std(ddof=1)),
            "baseline_r2": float(r2),
        })
    return rows


def main():
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    out_dir = Path(args.out_dir) if args.out_dir else sweep_dir / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_concepts(sweep_dir)
    rng = np.random.default_rng(args.seed)

    rows = []
    cells = df.group_by(["map", "template"], maintain_order=True)
    for (map_name, template), cell in cells:
        cell = cell.sort("concept")
        for row in analyze_cell(
            cell["baseline_rank"].to_numpy().astype(float),
            cell["change"].to_numpy().astype(float),
            cell["group"].to_numpy(),
            args.degree, args.permutations, args.strata, rng,
        ):
            row["map"] = map_name
            row["template"] = template
            rows.append(row)

    per_cell = pl.DataFrame(rows)
    per_cell.write_csv(out_dir / "residualized_per_cell.csv")

    order = pl.DataFrame({"group": GROUP_ORDER, "_o": range(len(GROUP_ORDER))})
    summary = (
        per_cell.group_by("group")
        .agg(
            pl.len().alias("n_cells"),
            pl.col("raw_change").mean().alias("raw_mean"),
            pl.col("residual_change").mean().alias("resid_mean"),
            pl.col("residual_change").std().alias("resid_sd"),
            (pl.col("residual_change") < 0).sum().alias("n_resid_fell"),
            (pl.col("residual_change") > 0).sum().alias("n_resid_rose"),
            (pl.col("strat_p") < ALPHA).sum().alias("n_sig"),
        )
        .join(order, on="group").sort("_o").drop("_o")
    )
    summary.write_csv(out_dir / "residualized_summary.csv")

    per_map = (
        per_cell.group_by(["map", "group"])
        .agg(
            pl.col("raw_change").mean().alias("raw_mean"),
            pl.col("residual_change").mean().alias("resid_mean"),
            (pl.col("residual_change") < 0).sum().alias("n_resid_fell"),
            (pl.col("strat_p") < ALPHA).sum().alias("n_sig"),
            pl.len().alias("n_cells"),
        )
        .join(order, on="group").sort(["map", "_o"]).drop("_o")
    )
    per_map.write_csv(out_dir / "residualized_per_map.csv")

    r2 = per_cell["baseline_r2"].mean()
    print(f"Baseline rank alone explains {r2:.1%} of the per-concept change variance "
          f"(degree-{args.degree} fit).\n")
    print("Group effect before and after removing the baseline-rank trend:")
    print("  raw_mean   = mean group change, as reported in the paper")
    print("  resid_mean = mean group change relative to equally-well-ranked concepts")
    print("  n_sig      = cells clearing a baseline-stratified permutation null\n")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200,
                   tbl_formatting="ASCII_FULL_CONDENSED", tbl_hide_dataframe_shape=True,
                   float_precision=2):
        print(summary)
        print()
        print(per_map)
    print(f"\nWrote {out_dir/'residualized_summary.csv'}, "
          f"{out_dir/'residualized_per_map.csv'}, {out_dir/'residualized_per_cell.csv'}")


if __name__ == "__main__":
    main()
