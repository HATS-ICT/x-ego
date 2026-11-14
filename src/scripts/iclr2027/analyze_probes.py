#!/usr/bin/env python3
"""
P3 -- paired uncertainty and the primary C-F / C-S / C-H contrasts.

Seed inclusion follows the pre-declared rule: seeds 1-3 are primary; seed k in
{4, 5} is added only if all three arms have results for it (checked per task).
Frozen (F) is one encoder shared across seeds.

Bootstrap: resamples of held-out MATCHES with replacement (all rows of a
sampled match kept; same draw for every arm), crossed with a resample of
encoder seeds (same seed ids for every arm). Each resample recomputes the full
metric from exported test scores. Task-group aggregates (e.g. the 8 other-player
occupancy tasks) share ONE match draw across all member tasks per resample.
Intervals cover this split, these seeds and these held-out matches only.

    uv run python -m src.scripts.iclr2027.analyze_probes --out-tag p2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl
from dotenv import load_dotenv

from .split_common import COHORT

ARMS = ("cecl", "single", "shuffle")
CONTRASTS = (("cecl", "frozen"), ("cecl", "single"), ("cecl", "shuffle"), ("single", "frozen"), ("shuffle", "frozen"))
PRIMARY = {"multi_label_cls": ["macro_auroc", "macro_ap"], "binary_cls": ["auroc", "ap"],
           "multi_cls": ["top1", "macro_recall", "macro_ovr_auroc"], "regression": ["r2"]}


# --------------------------------------------------------------------------- vectorized metrics

def _ranks(s):
    order = np.argsort(s, axis=0, kind="mergesort")
    r = np.empty(order.shape, dtype=np.float64)
    np.put_along_axis(r, order, np.broadcast_to(np.arange(1, s.shape[0] + 1, dtype=np.float64)[:, None], order.shape), axis=0)
    return r


def auroc_cols(y, s):
    y = y.astype(bool)
    P = y.sum(0).astype(np.float64)
    N = y.shape[0] - P
    denom = np.where(P * N > 0, P * N, np.nan)
    return ((_ranks(s) * y).sum(0) - P * (P + 1) / 2) / denom


def ap_cols(y, s):
    order = np.argsort(-s, axis=0, kind="mergesort")
    ys = np.take_along_axis(y.astype(np.float64), order, axis=0)
    prec = np.cumsum(ys, axis=0) / np.arange(1, ys.shape[0] + 1)[:, None]
    P = ys.sum(0)
    return np.where(P > 0, (prec * ys).sum(0) / np.where(P > 0, P, 1), np.nan)


def metric(form, y, s, name):
    if form == "multi_label_cls":
        ok = ~np.isnan(auroc_cols(y, s))  # evaluable in THIS resample; identical rows => identical mask across arms
        cols = auroc_cols(y, s) if name == "macro_auroc" else ap_cols(y, s)
        return float(np.mean(cols[ok]))
    if form == "binary_cls":
        f = auroc_cols if name == "auroc" else ap_cols
        return float(f(y.reshape(-1, 1), s.reshape(-1, 1))[0])
    if form == "multi_cls":
        yi = y.astype(int)
        if name == "top1":
            return float((s.argmax(1) == yi).mean())
        if name == "macro_recall":
            pred = s.argmax(1)
            return float(np.mean([(pred[yi == c] == c).mean() for c in np.unique(yi)]))
        return float(np.nanmean(auroc_cols(np.eye(s.shape[1])[yi], s)))
    y2, p2 = y.reshape(len(y), -1), s.reshape(len(s), -1)
    return float(np.mean(1 - ((y2 - p2) ** 2).sum(0) / ((y2 - y2.mean(0)) ** 2).sum(0)))


# --------------------------------------------------------------------------- per-task worker

def load_task(root: Path, task: str, enc_list, probes):
    data = {e: np.load(root / e / f"{task}.npz", allow_pickle=False) for e in enc_list}
    z0 = data["frozen"]
    te = z0["partition"] == "test"
    wid = z0["window_id"][te]
    for e in enc_list:  # identical rows across arms, or nothing is comparable
        if not np.array_equal(data[e]["window_id"][data[e]["partition"] == "test"], wid):
            raise ValueError(f"{task}: {e} test rows differ from frozen")
    S = {pr: {e: data[e][f"{pr}_test"] for e in enc_list} for pr in probes if f"{pr}_test" in z0.files}
    return z0["y"][te], z0["match_id"][te], S


def process_task(job):
    task, form, seeds, root, probes, B = job
    rng = np.random.default_rng(zlib.crc32(task.encode()))
    enc_list = ["frozen"] + [f"{a}_s{s}" for a in ARMS for s in seeds]
    y, matches, S_all = load_task(root, task, enc_list, probes)
    uniq = np.unique(matches)
    by_match = [np.flatnonzero(matches == m) for m in uniq]
    rows, contrast_rows = [], []
    for probe, S in S_all.items():
        for mname in PRIMARY[form]:
            point = {e: metric(form, y, S[e], mname) for e in enc_list}
            arm_point = {"frozen": point["frozen"], **{a: float(np.mean([point[f"{a}_s{s}"] for s in seeds])) for a in ARMS}}
            boots = {c: [] for c in CONTRASTS}
            for _ in range(B):
                idx = np.concatenate([by_match[i] for i in rng.integers(0, len(uniq), len(uniq))])
                ss = rng.choice(seeds, size=len(seeds), replace=True)
                cache = {}

                def val(e):
                    if e not in cache:
                        cache[e] = metric(form, y[idx], S[e][idx], mname)
                    return cache[e]
                v = {"frozen": val("frozen"), **{a: float(np.mean([val(f"{a}_s{s}") for s in ss])) for a in ARMS}}
                for a, b in CONTRASTS:
                    boots[(a, b)].append(v[a] - v[b])
            for arm, v in arm_point.items():
                rows.append({"task": task, "probe": probe, "metric": mname, "arm": arm, "value": v,
                             "per_seed": json.dumps([point[f"{arm}_s{s}"] for s in seeds]) if arm != "frozen" else "",
                             "seeds": ",".join(map(str, seeds)), "test_rows": int(len(y)), "test_matches": int(len(uniq))})
            for (a, b), bs in boots.items():
                bs = np.asarray(bs)
                ref = (lambda s: point["frozen"]) if b == "frozen" else (lambda s: point[f"{b}_s{s}"])
                contrast_rows.append({"task": task, "probe": probe, "metric": mname, "contrast": f"{a}-{b}",
                                      "diff": arm_point[a] - arm_point[b],
                                      "ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5)),
                                      "p_le_0": float((bs <= 0).mean()),
                                      "seed_wise": json.dumps([point[f"{a}_s{s}"] - ref(s) for s in seeds]),
                                      "seeds": ",".join(map(str, seeds)), "test_matches": int(len(uniq)), "B": B})
    return rows, contrast_rows


def process_group(job):
    gname, members, forms, seeds, root, probes, B = job
    rng = np.random.default_rng(zlib.crc32(gname.encode()))
    form = forms[members[0]]
    enc_list = ["frozen"] + [f"{a}_s{s}" for a in ARMS for s in seeds]
    loaded = {t: load_task(root, t, enc_list, probes) for t in members}
    all_matches = np.unique(np.concatenate([loaded[t][1] for t in members]))
    out = []
    for probe in probes:
        if not all(probe in loaded[t][2] for t in members):
            continue
        for mname in PRIMARY[form]:
            def arm_vals(match_draw, seed_draw):
                vals = {}
                for arm in ("frozen",) + ARMS:
                    encs = ["frozen"] if arm == "frozen" else [f"{arm}_s{s}" for s in seed_draw]
                    per_task = []
                    for t in members:
                        y_t, m_t, S_t = loaded[t]
                        idx = slice(None) if match_draw is None else np.concatenate([np.flatnonzero(m_t == m) for m in match_draw])
                        memo = {}
                        per_task.append(np.mean([memo.setdefault(e, metric(form, y_t[idx], S_t[probe][e][idx], mname)) for e in encs]))
                    vals[arm] = float(np.mean(per_task))
                return vals
            point = arm_vals(None, seeds)
            boots = {c: [] for c in CONTRASTS}
            for _ in range(B):
                v = arm_vals(rng.choice(all_matches, size=len(all_matches), replace=True),
                             rng.choice(seeds, size=len(seeds), replace=True))
                for a, b in CONTRASTS:
                    boots[(a, b)].append(v[a] - v[b])
            common = {"group": gname, "tasks": len(members), "probe": probe, "metric": mname,
                      "seeds": ",".join(map(str, seeds)), "test_matches": int(len(all_matches)), "B": B}
            out += [{**common, "kind": "arm", "name": a, "value": v, "ci_lo": None, "ci_hi": None} for a, v in point.items()]
            out += [{**common, "kind": "contrast", "name": f"{a}-{b}", "value": point[a] - point[b],
                     "ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5))}
                    for (a, b), bs in ((k, np.asarray(v)) for k, v in boots.items())]
    return gname, out


GROUPS = {
    "other_location_8": [f"{w}_location_{h}s" for w in ("teammate", "enemy") for h in (0, 5, 10, 20)],
    "teammate_location_4": [f"teammate_location_{h}s" for h in (0, 5, 10, 20)],
    "enemy_location_4": [f"enemy_location_{h}s" for h in (0, 5, 10, 20)],
    **{f"matched_{p}_region_3": [f"{p}_region_{h}s__matched" for h in (0, 5, 10)] for p in ("self", "teammate", "enemy")},
}


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-tag", default="p2")
    ap.add_argument("--probes", nargs="+", default=["converged", "historical_best"])
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--B-agg", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    ap.add_argument("--task-regex", default=None)
    ap.add_argument("--no-aggregates", action="store_true")
    args = ap.parse_args()

    root = Path(os.environ["OUTPUT_BASE_PATH"]) / "iclr2027_probes" / COHORT / args.out_tag
    forms, have = {}, {}
    for f in glob.glob(str(root / "*" / "*.json")):
        r = json.loads(Path(f).read_text())
        forms[r["task"]] = r["ml_form"]
        have.setdefault(r["task"], set()).add(r["encoder"])
    tasks = sorted(have)
    if args.task_regex:
        tasks = [t for t in tasks if re.search(args.task_regex, t)]
    seeds_of = {}
    for t in tasks:
        seeds = [s for s in (1, 2, 3, 4, 5) if all(f"{a}_s{s}" in have[t] for a in ARMS)]
        if {1, 2, 3} <= set(seeds) and "frozen" in have[t]:
            seeds_of[t] = seeds
        else:
            print(f"skip {t}: incomplete encoders {sorted(have[t])}")

    rows, contrast_rows, agg_rows = [], [], []
    with Pool(args.workers) as pool:
        for r_, c_ in pool.imap_unordered(process_task, [(t, forms[t], seeds_of[t], root, args.probes, args.B) for t in seeds_of]):
            rows += r_
            contrast_rows += c_
            if r_:
                print(f"done {r_[0]['task']}")
        if not args.no_aggregates:
            gjobs = []
            for g, members in GROUPS.items():
                members = [t for t in members if t in seeds_of]
                if not members:
                    continue
                seeds = sorted(set.intersection(*[set(seeds_of[t]) for t in members]))
                gjobs.append((g, members, forms, seeds, root, args.probes, args.B_agg))
            for g, out in pool.imap_unordered(process_group, gjobs):
                agg_rows += out
                print(f"aggregate {g} done")

    an = root / "_analysis"
    an.mkdir(exist_ok=True)
    pl.DataFrame(rows).write_csv(an / "arm_scores.csv")
    pl.DataFrame(contrast_rows).write_csv(an / "contrasts.csv")
    if agg_rows:
        pl.DataFrame(agg_rows, infer_schema_length=None).write_csv(an / "aggregates.csv")
    print(f"wrote {an}")


if __name__ == "__main__":
    main()
