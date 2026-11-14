#!/usr/bin/env python3
"""
P5 -- supporting-evidence restrictions on Mirage, re-scored from saved test
predictions of every arm (no refitting). Restrictions (fixed before viewing):

  R1 disjoint_region  {teammate,enemy}_location_{5,10,20}s, moved_disjoint == 1
  R2 plant_free       global_bombPlanted,                  plant_inside_window == 0
  R3 death_free       {enemy,teammate}_aliveCount,          no_death_in_window == 1

Flags come from src/scripts/eval_subsets (ground-truth trajectories/events),
joined on the task CSV `idx`, so every arm is scored on identical rows.
Paired match x seed bootstrap as in analyze_probes. For R3, remaining classes
are reported; present-class metrics are used (fixed-class AUROC is undefined
for absent classes).

    uv run python -m src.scripts.iclr2027.analyze_restrictions --out-tag p2
"""
import argparse
import json
import os
import zlib
from pathlib import Path

import numpy as np
import polars as pl
from dotenv import load_dotenv

from .analyze_probes import ARMS, CONTRASTS, PRIMARY, load_task, metric
from .split_common import COHORT, cohort_dir, ensure_tick_norm, tasks_folder, window_id

RESTRICTIONS = [
    *[("disjoint_region", f"{w}_location_{h}s", "moved", "moved_disjoint", 1) for w in ("teammate", "enemy") for h in (5, 10, 20)],
    ("plant_free", "global_bombPlanted", "bomb", "plant_inside_window", 0),
    ("death_free", "enemy_aliveCount", "alive", "no_death_in_window", 1),
    ("death_free", "teammate_aliveCount", "alive", "no_death_in_window", 1),
]


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-tag", default="p2")
    ap.add_argument("--probes", nargs="+", default=["converged", "historical_best"])
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()
    data_root = Path(os.environ["DATA_BASE_PATH"])
    root = Path(os.environ["OUTPUT_BASE_PATH"]) / "iclr2027_probes" / COHORT / args.out_tag
    sub_dir = cohort_dir(data_root, "mirage") / "eval_subsets"
    rows, audit = [], []
    for rname, task, builder, flag, keep in RESTRICTIONS:
        fpath = sub_dir / f"mirage-{task}-{builder}.csv"
        if not fpath.exists():
            print(f"skip {task}: no flags at {fpath}")
            continue
        have = {Path(p).parent.name for p in root.glob(f"*/{task}.json")}
        seeds = [s for s in (1, 2, 3, 4, 5) if all(f"{a}_s{s}" in have for a in ARMS)]
        if not {1, 2, 3} <= set(seeds) or "frozen" not in have:
            print(f"skip {task}: incomplete encoders")
            continue
        form = json.loads((root / "frozen" / f"{task}.json").read_text())["ml_form"]
        flags = pl.read_csv(fpath).select("idx", flag)
        lab, _ = ensure_tick_norm(pl.read_csv(data_root / "mirage" / "labels" / tasks_folder() / f"{task}.csv", null_values=[]), data_root, "mirage")
        lab = lab.join(flags, on="idx", how="left")
        wid = [window_id(r["match_id"], r["round_num"], str(int(r["pov_steamid"])), int(round(r["start_tick_norm"])), int(round(r["end_tick_norm"])))
               for r in lab.select("match_id", "round_num", "pov_steamid", "start_tick_norm", "end_tick_norm").iter_rows(named=True)]
        keep_ids = {w for w, f in zip(wid, lab[flag].to_list()) if f is not None and int(f) == keep}
        unflagged = int(lab[flag].null_count())

        enc_list = ["frozen"] + [f"{a}_s{s}" for a in ARMS for s in seeds]
        y_all, m_all, S_all = load_task(root, task, enc_list, args.probes)
        te_ids = np.load(root / "frozen" / f"{task}.npz")["window_id"][np.load(root / "frozen" / f"{task}.npz")["partition"] == "test"]
        mask = np.array([w in keep_ids for w in te_ids.tolist()])
        y, matches = y_all[mask], m_all[mask]
        uniq = np.unique(matches)
        by_match = [np.flatnonzero(matches == m) for m in uniq]
        classes = sorted(set(np.asarray(y).astype(int).ravel().tolist())) if form in ("multi_cls", "binary_cls") else None
        audit.append({"restriction": rname, "task": task, "flag": f"{flag}=={keep}", "test_rows_total": int(len(y_all)),
                      "test_rows_kept": int(mask.sum()), "test_matches_kept": int(len(uniq)), "rows_without_flag": unflagged,
                      "classes_remaining": json.dumps(classes)})
        if mask.sum() < 30 or len(uniq) < 3:
            print(f"{task}: only {mask.sum()} rows / {len(uniq)} matches after restriction; reported in audit, not scored")
            continue
        rng = np.random.default_rng(zlib.crc32((rname + task).encode()))
        for probe, S in S_all.items():
            S = {e: v[mask] for e, v in S.items()}
            names = PRIMARY[form]
            if form == "binary_cls" and len(classes) < 2:
                names = []  # AUROC/AP undefined with one class; reported in audit only
            for mname in names:
                point = {e: metric(form, y, S[e], mname) for e in enc_list}
                arm = {"frozen": point["frozen"], **{a: float(np.mean([point[f"{a}_s{s}"] for s in seeds])) for a in ARMS}}
                boots = {c: [] for c in CONTRASTS}
                for _ in range(args.B):
                    idx = np.concatenate([by_match[i] for i in rng.integers(0, len(uniq), len(uniq))])
                    ss = rng.choice(seeds, size=len(seeds), replace=True)
                    memo = {}
                    val = lambda e: memo.setdefault(e, metric(form, y[idx], S[e][idx], mname))
                    v = {"frozen": val("frozen"), **{a: float(np.mean([val(f"{a}_s{s}") for s in ss])) for a in ARMS}}
                    for a, b in CONTRASTS:
                        boots[(a, b)].append(v[a] - v[b])
                for a, b in CONTRASTS:
                    bs = np.asarray(boots[(a, b)])
                    rows.append({"restriction": rname, "task": task, "probe": probe, "metric": mname, "contrast": f"{a}-{b}",
                                 "diff": arm[a] - arm[b], "ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5)),
                                 "value_a": arm[a], "value_b": arm[b], "test_rows": int(mask.sum()), "test_matches": int(len(uniq)),
                                 "seeds": ",".join(map(str, seeds))})
        print(f"done {rname} {task}: {mask.sum()} rows, {len(uniq)} matches")
    an = root / "_analysis"
    an.mkdir(exist_ok=True)
    pl.DataFrame(audit).write_csv(an / "restrictions_audit.csv")
    if rows:
        pl.DataFrame(rows).write_csv(an / "restrictions.csv")
    print(f"wrote {an}/restrictions*.csv")


if __name__ == "__main__":
    main()
