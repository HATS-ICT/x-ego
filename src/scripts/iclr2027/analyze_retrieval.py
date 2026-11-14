#!/usr/bin/env python3
"""
Tier-2 diagnostic: synchronized-teammate retrieval on encoder features h.

Replaces the in-training val retrieval metric, whose unshuffled batches made the
negatives near-duplicate moments of the same round. Two candidate pools, both
always containing the anchor's true synchronized teammates:
  cross_match  negatives = clips from OTHER test matches      (team/game-state structure)
  same_round   negatives = clips from the same match+round at other times (synchrony)
Cosine similarity on L2-normalized h; identical pools for every encoder.
Chance = expected metric under a random ranking (top-1: P/(P+N); AP: Monte Carlo).

    uv run python -m src.scripts.iclr2027.analyze_retrieval
"""
import glob
import os
from pathlib import Path

import numpy as np
import polars as pl
from dotenv import load_dotenv

from .split_common import COHORT, cohort_dir


def ap_of(sims, is_pos):
    order = np.argsort(-sims, kind="mergesort")
    hits = is_pos[order]
    cum = np.cumsum(hits)
    return float((cum[hits] / (np.flatnonzero(hits) + 1)).mean())


def main():
    load_dotenv()
    data_root = Path(os.environ["DATA_BASE_PATH"])
    feat_root = Path(os.environ["OUTPUT_BASE_PATH"]) / "iclr2027_features" / COHORT
    meta = pl.read_csv(cohort_dir(data_root, "mirage") / "retrieval_windows.csv", schema_overrides={"pov_steamid": pl.Utf8})
    feats = {}
    for f in glob.glob(str(feat_root / "*" / "retr_shard000of001.npz")):
        z = np.load(f, allow_pickle=False)
        order = {w: i for i, w in enumerate(z["window_id"].tolist())}
        idx = np.array([order[w] for w in meta["window_id"].to_list()])
        for e in z["encoders"].tolist():
            feats[e] = z[f"h__{e}"][idx].astype(np.float64)
    group = meta["group_id"].to_numpy()
    match = meta["match_id"].to_numpy()
    mround = np.array([f"{m}|{r}" for m, r in zip(match, meta["round_num"].to_numpy())])
    n = len(meta)
    rng = np.random.default_rng(0)
    pools = {}
    for i in range(n):
        pos = np.flatnonzero((group == group[i]) & (np.arange(n) != i))
        if len(pos) == 0:
            continue
        cross = np.flatnonzero(match != match[i])
        same = np.flatnonzero((mround == mround[i]) & (group != group[i]))
        pools.setdefault("cross_match", []).append((i, pos, cross))
        if len(same):
            pools.setdefault("same_round", []).append((i, pos, same))
    rows = []
    for pool, items in pools.items():
        chance_top1 = np.mean([len(p) / (len(p) + len(q)) for _, p, q in items])
        chance_ap = np.mean([ap_of(rng.random(len(p) + len(q)), np.r_[np.ones(len(p), bool), np.zeros(len(q), bool)]) for _, p, q in items[:2000]])
        for e, H in sorted(feats.items()):
            Hn = H / np.linalg.norm(H, axis=1, keepdims=True)
            top1, aps, per_match = [], [], {}
            for i, p, q in items:
                cand = np.r_[p, q]
                sims = Hn[cand] @ Hn[i]
                is_pos = np.r_[np.ones(len(p), bool), np.zeros(len(q), bool)]
                t1 = float(is_pos[np.argmax(sims)])
                a = ap_of(sims, is_pos)
                top1.append(t1); aps.append(a)
                per_match.setdefault(match[i], []).append(a)
            arm = "frozen" if e == "frozen" else e.rsplit("_s", 1)[0]
            rows.append({"pool": pool, "encoder": e, "arm": arm, "anchors": len(items),
                         "top1": float(np.mean(top1)), "mAP": float(np.mean(aps)),
                         "chance_top1": float(chance_top1), "chance_mAP": float(chance_ap),
                         "mAP_match_sd": float(np.std([np.mean(v) for v in per_match.values()]))})
    df = pl.DataFrame(rows).sort(["pool", "arm", "encoder"])
    out = Path(os.environ["OUTPUT_BASE_PATH"]) / "iclr2027_probes" / COHORT / "retrieval_diagnostic.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out)
    summary = df.group_by(["pool", "arm"]).agg(pl.col("mAP").mean(), pl.col("mAP").std().alias("mAP_seed_sd"),
                                              pl.col("top1").mean(), pl.col("chance_mAP").first(), pl.col("chance_top1").first(),
                                              pl.len().alias("encoders")).sort(["pool", "arm"])
    with pl.Config(tbl_rows=40, tbl_cols=20):
        print(summary)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
