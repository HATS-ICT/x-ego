#!/usr/bin/env python3
"""
Paper-facing summary of the P1-P4 results -> _analysis/summary.md

Every number is read from saved artifacts (aggregates.csv, contrasts.csv,
arm_scores.csv, per-task probe JSONs, retrieval_diagnostic.csv); nothing is
recomputed or selected here.

    uv run python -m src.scripts.iclr2027.make_tables --out-tag p2
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import polars as pl
from dotenv import load_dotenv

from .split_common import COHORT

ARMS = ["frozen", "cecl", "single", "shuffle"]
LABEL = {"frozen": "F frozen", "cecl": "C CECL", "single": "S single-view", "shuffle": "H shuffled"}


def fmt(v, d=3):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def ci(r):
    return f"{r['value']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-tag", default="p2")
    args = ap.parse_args()
    root = Path(os.environ["OUTPUT_BASE_PATH"]) / "iclr2027_probes" / COHORT / args.out_tag
    an = root / "_analysis"
    L = [f"# ICLR-2027 follow-up results (cohort `{COHORT}`, Mirage, SigLIP2 frame-mean)\n",
         "Primary checkpoint: `last.ckpt`. Intervals: 95% paired bootstrap over held-out matches x encoder seeds; "
         "scope = this split, these seeds, these 9 test matches.\n"]

    agg = pl.read_csv(an / "aggregates.csv") if (an / "aggregates.csv").exists() else None
    if agg is not None:
        seeds = agg["seeds"][0]
        L.append(f"Seeds included: {seeds}.\n")
        for probe in ("converged", "historical_best"):
            L.append(f"\n## Task-group aggregates — probe `{probe}`\n")
            L.append("| group | metric | " + " | ".join(LABEL[a] for a in ARMS) + " | C−F | C−S | C−H |")
            L.append("|---|---|" + "---|" * (len(ARMS) + 3))
            sub = agg.filter(pl.col("probe") == probe)
            for (g, m), blk in sub.group_by(["group", "metric"], maintain_order=True):
                arm = {r["name"]: r["value"] for r in blk.filter(pl.col("kind") == "arm").iter_rows(named=True)}
                con = {r["name"]: r for r in blk.filter(pl.col("kind") == "contrast").iter_rows(named=True)}
                L.append(f"| {g} | {m} | " + " | ".join(fmt(arm.get(a)) for a in ARMS) + " | "
                         + " | ".join(ci(con[c]) if c in con else "—" for c in ("cecl-frozen", "cecl-single", "cecl-shuffle")) + " |")

    con = pl.read_csv(an / "contrasts.csv") if (an / "contrasts.csv").exists() else None
    scores = pl.read_csv(an / "arm_scores.csv") if (an / "arm_scores.csv").exists() else None
    if scores is not None:
        L.append("\n## Per-task scores (converged probe)\n")
        L.append("| task | metric | " + " | ".join(LABEL[a] for a in ARMS) + " | C−F | C−S | C−H | test rows |")
        L.append("|---|---|" + "---|" * (len(ARMS) + 4))
        sc = scores.filter(pl.col("probe") == "converged")
        for (t, m), blk in sc.group_by(["task", "metric"], maintain_order=True):
            arm = {r["arm"]: r["value"] for r in blk.iter_rows(named=True)}
            cc = {r["contrast"]: r for r in con.filter((pl.col("task") == t) & (pl.col("metric") == m) & (pl.col("probe") == "converged")).iter_rows(named=True)}
            cell = lambda c: f"{cc[c]['diff']:+.3f} [{cc[c]['ci_lo']:+.3f}, {cc[c]['ci_hi']:+.3f}]" if c in cc else "—"
            L.append(f"| {t} | {m} | " + " | ".join(fmt(arm.get(a)) for a in ARMS) + " | "
                     + " | ".join(cell(c) for c in ("cecl-frozen", "cecl-single", "cecl-shuffle")) + f" | {blk['test_rows'][0]} |")

    # P4 gate: 20 s teammate occupancy, practical metrics + trivial predictors on identical rows
    task = "teammate_location_20s"
    recs = {Path(f).parent.name: json.loads(Path(f).read_text()) for f in glob.glob(str(root / "*" / f"{task}.json"))}
    if recs:
        L.append(f"\n## P4 — `{task}`: practical prediction quality on identical test rows\n")
        L.append("| encoder | macro-AUROC | macro-AP | F1@0.5 | F1@val-thr (thr) | pred. card. @thr | true card. |")
        L.append("|---|---|---|---|---|---|---|")
        for e in sorted(recs):
            t = recs[e]["probes"]["converged"]["test"]
            L.append(f"| {e} | {fmt(t['macro_auroc'])} | {fmt(t['macro_ap'])} | {fmt(t['f1_macro_at_0.5'])} | "
                     f"{fmt(t['f1_macro_at_val_thr'])} ({t['val_threshold']}) | {fmt(t['cardinality_pred_at_val_thr'], 2)} | {fmt(t['cardinality_true'], 2)} |")
        b = next(iter(recs.values()))["baselines"]
        r0 = next(iter(recs.values()))
        L.append(f"| always-positive | — | — | — | {fmt(b['always_positive']['f1_macro'])} | {b['always_positive']['cardinality_pred']} | |")
        L.append(f"| always-negative | — | — | — | {fmt(b['always_negative']['f1_macro'])} | 0 | |")
        L.append(f"| train prevalence | 0.500 | {fmt(b['train_prevalence']['macro_ap'])} | — | "
                 f"{fmt(b['train_prevalence']['f1_macro_at_val_thr'])} ({b['train_prevalence']['val_threshold']}) | | |")
        L.append(f"\nRows: {r0['rows']}; test matches: {r0['test_matches']}; evaluable labels "
                 f"{len(r0['evaluable_labels'])}/23 (omitted: {r0['omitted_labels']}); per-region test support: {r0['test_support']}.")

    # solver audit
    bad = []
    for f in glob.glob(str(root / "*" / "*.json")):
        r = json.loads(Path(f).read_text())
        fit = r["probes"]["converged"]["fit"]
        if fit.get("boundary_optimum") or fit.get("selected", {}).get("converged") is False:
            bad.append(f"{r['encoder']}/{r['task']} (lambda={fit['lambda']:.0e}, boundary={fit.get('boundary_optimum')}, "
                       f"converged={fit.get('selected', {}).get('converged', 'n/a')})")
    L.append(f"\n## Solver audit\n\n{len(bad)} converged-probe fits at a grid boundary or not converged at the selected lambda.")
    L += [f"- {b}" for b in sorted(bad)[:40]]

    rs, ra = an / "restrictions.csv", an / "restrictions_audit.csv"
    if ra.exists():
        L.append("\n## P5 restrictions (converged probe; identical rows across arms)\n")
        aud = pl.read_csv(ra)
        L.append("| restriction | task | kept rows / test rows | matches | classes left |")
        L.append("|---|---|---|---|---|")
        for x in aud.iter_rows(named=True):
            L.append(f"| {x['restriction']} | {x['task']} | {x['test_rows_kept']} / {x['test_rows_total']} | {x['test_matches_kept']} | {x['classes_remaining']} |")
        if rs.exists():
            r = pl.read_csv(rs).filter(pl.col("probe") == "converged")
            L.append("\n| restriction | task | metric | contrast | diff [95% CI] | A | B |")
            L.append("|---|---|---|---|---|---|---|")
            for x in r.filter(pl.col("contrast").is_in(["cecl-frozen", "cecl-single", "cecl-shuffle"])).iter_rows(named=True):
                L.append(f"| {x['restriction']} | {x['task']} | {x['metric']} | {x['contrast']} | {x['diff']:+.3f} [{x['ci_lo']:+.3f}, {x['ci_hi']:+.3f}] | {x['value_a']:.3f} | {x['value_b']:.3f} |")

    rp = root.parent / "retrieval_diagnostic.csv"
    if rp.exists():
        r = pl.read_csv(rp).group_by(["pool", "arm"]).agg(pl.col("mAP").mean(), pl.col("mAP").std().alias("sd"),
                                                         pl.col("top1").mean(), pl.col("chance_mAP").first(), pl.col("chance_top1").first()).sort(["pool", "arm"])
        L.append("\n## Teammate-retrieval diagnostic on h (test matches)\n")
        L.append("| pool | arm | mAP (seed sd) | top-1 | chance mAP | chance top-1 |")
        L.append("|---|---|---|---|---|---|")
        for x in r.iter_rows(named=True):
            L.append(f"| {x['pool']} | {x['arm']} | {fmt(x['mAP'])} ({fmt(x['sd'])}) | {fmt(x['top1'])} | {fmt(x['chance_mAP'])} | {fmt(x['chance_top1'])} |")

    out = an / "summary.md"
    an.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
