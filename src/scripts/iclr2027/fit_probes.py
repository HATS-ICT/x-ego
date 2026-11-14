#!/usr/bin/env python3
"""
P2 / P4 probes on cached encoder features h.

For one encoder, fits every requested task with two probe families on
IDENTICAL rows:

  historical  nn.Linear on raw h, AdamW lr 4e-4 wd 0, 20 epochs, batch 16 x
              accumulate 2, val-loss checkpoint selection (reports best + last).
              Features are deterministic, so this replays the paper's protocol.
  converged   training-only standardization; ridge (exact, averaged objective,
              unpenalized intercept) for regression; full-batch float64 L-BFGS
              logistic / softmax (averaged loss + lambda/2 ||W||^2, bias
              unpenalized) run to a gradient tolerance for classification.
              Regularization selected on validation; boundary optima trigger a
              validation-only grid expansion. Solver status is recorded.

P4 outputs per task: AUROC / AP (macro + per class, on an evaluable-label mask
that depends only on the shared test rows), F1 at 0.5 and at a validation-
selected scalar threshold, predicted vs true cardinality, and trivial
baselines (always-positive / always-negative / training prevalence / majority
class / training mean) on the same rows. Validation and test scores are
exported with window ids and match ids for the paired bootstrap.

    uv run python -m src.scripts.iclr2027.fit_probes --encoder cecl_s1 --tier core
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from rich import print as rprint
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, r2_score, roc_auc_score

from src.utils.config_utils import load_task_config

from .build_window_manifest import CORE_TASKS
from .split_common import COHORT, cohort_dir, ensure_tick_norm, tasks_folder, window_id

LAMBDA_GRID_RIDGE = [10.0 ** k for k in range(-6, 7)]
LAMBDA_GRID_LOGIT = [10.0 ** k for k in range(-6, 3)]
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
MATCHED_REGION = [f"{p}_region_{h}s" for p in ("self", "teammate", "enemy") for h in (0, 5, 10)]
MATCHED_SPEED = ["self_speed_0s", "teammate_speed_0s"]
NUM_REGIONS = 23
LOCK_STALE_SEC = 1800


# --------------------------------------------------------------------------- data

def load_features(feat_dir: Path, encoder: str, tiers: list[str], wait_sec: int = 0) -> tuple[dict, np.ndarray]:
    """Union of the requested tiers' shards. Optionally waits for shards still being extracted."""
    ids, blocks = [], []
    for tier in tiers:
        deadline = time.time() + wait_sec
        while True:
            shards = sorted(glob.glob(str(feat_dir / f"{tier}_shard*of*.npz")))
            shards = [s for s in shards if "_pilot" not in s and not s.endswith(".tmp.npz")]
            n_expected = int(Path(shards[0]).stem.split("of")[-1]) if shards else None
            if shards and len(shards) == n_expected:
                break
            if time.time() > deadline:
                raise FileNotFoundError(f"{tier}: {len(shards)}/{n_expected or '?'} shards present in {feat_dir}")
            rprint(f"[yellow]waiting[/yellow] for {tier} shards ({len(shards)}/{n_expected or '?'})")
            time.sleep(60)
        for s in shards:
            with np.load(s, allow_pickle=False) as z:
                if f"h__{encoder}" not in z.files:
                    raise KeyError(f"{Path(s).name} has no features for {encoder}")
                ids.append(z["window_id"])
                blocks.append(z[f"h__{encoder}"].astype(np.float32))
    all_ids = np.concatenate(ids)
    index = {w: i for i, w in enumerate(all_ids.tolist())}
    if len(index) != len(all_ids):
        raise ValueError("duplicate window ids across shards")
    return index, np.concatenate(blocks)


def historical_task(data_root: Path, task: str):
    cfg = load_task_config(task, data_root, "mirage")
    cols = cfg["label_column"].split(";") if isinstance(cfg["label_column"], str) else list(cfg["label_column"])
    df, derived = ensure_tick_norm(pl.read_csv(data_root / "mirage" / "labels" / tasks_folder() / f"{task}.csv", null_values=[]), data_root, "mirage")
    wids = [
        window_id(r["match_id"], r["round_num"], str(int(r["pov_steamid"])), int(round(r["start_tick_norm"])), int(round(r["end_tick_norm"])))
        for r in df.select("match_id", "round_num", "pov_steamid", "start_tick_norm", "end_tick_norm").iter_rows(named=True)
    ]
    y = df.select(cols).to_numpy().astype(np.float64)
    if cfg["ml_form"] in ("multi_cls", "binary_cls"):
        y = y[:, 0]
    elif cfg["ml_form"] == "regression" and y.shape[1] == 1:
        y = y[:, 0]
    return {
        "task": task, "family": "historical", "ml_form": cfg["ml_form"], "tick_norm_derived": derived,
        "num_classes": int(cfg["num_classes"]) if cfg["num_classes"] not in (None, "") and not (isinstance(cfg["num_classes"], float) and math.isnan(cfg["num_classes"])) else None,
        "window_id": np.array(wids), "match_id": df["match_id"].to_numpy(), "partition": df["partition"].to_numpy(), "y": y,
    }


def matched_tasks(data_root: Path):
    df = pl.read_csv(data_root / "mirage" / "labels" / f"matched_suite_{COHORT}.csv", null_values=[""], schema_overrides={"pov_steamid": pl.Utf8})
    out = []
    for task in MATCHED_REGION + MATCHED_SPEED:
        is_speed = task in MATCHED_SPEED
        h = "speed" if is_speed else task.rsplit("_", 1)[-1]
        for pop, mask in (("matched", pl.col(f"matched_valid_{h}")), ("eligible", pl.col(task).is_not_null())):
            sub = df.filter(mask & pl.col(task).is_not_null())
            out.append({
                "task": f"{task}__{pop}", "family": "matched", "ml_form": "regression" if is_speed else "multi_cls",
                "num_classes": None if is_speed else NUM_REGIONS,
                "window_id": sub["window_id"].to_numpy(), "match_id": sub["match_id"].to_numpy(),
                "partition": sub["partition"].to_numpy(),
                "y": sub[task].to_numpy().astype(np.float64),
                "candidates": {p: int(df.filter(pl.col("partition") == p).height) for p in ("train", "val", "test")},
            })
    return out


# --------------------------------------------------------------------------- probes

def standardize(Xtr, *others):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    constant = sd < 1e-8
    sd = np.where(constant, 1.0, sd)
    return [(X - mu) / sd for X in (Xtr, *others)], {"n_constant_features": int(constant.sum())}


def ridge_path(Xtr, ytr, Xva, yva, grid):
    """Exact ridge for every lambda via one eigendecomposition; X already standardized."""
    n = Xtr.shape[0]
    ym = ytr.mean(0)
    evals, evecs = np.linalg.eigh(Xtr.T @ Xtr / n)
    proj = evecs.T @ (Xtr.T @ (ytr - ym) / n)
    fits = []
    for lam in grid:
        W = evecs @ (proj / (evals + lam)[:, None] if proj.ndim == 2 else proj / (evals + lam))
        pred = Xva @ W + ym
        fits.append((float(np.mean((pred - yva) ** 2)), lam, W, ym))
    return fits


def fit_ridge(Xtr, ytr, Xva, yva):
    grid = list(LAMBDA_GRID_RIDGE)
    expansions = 0
    while True:
        fits = ridge_path(Xtr, ytr, Xva, yva, grid)
        best = min(range(len(fits)), key=lambda i: fits[i][0])
        at_edge = best in (0, len(grid) - 1)
        if not at_edge or expansions >= 2:
            break
        grid = ([grid[0] / 100, grid[0] / 10] + grid) if best == 0 else (grid + [grid[-1] * 10, grid[-1] * 100])
        expansions += 1
    mse, lam, W, b = fits[best]
    return (lambda X: X @ W + b), {"lambda": lam, "val_mse": mse, "grid_expansions": expansions,
                                   "boundary_optimum": best in (0, len(grid) - 1)}


def lbfgs_fit(X, Y, ml_form, lam, k, W0=None, tol=1e-6, max_rounds=40):
    """
    Full-batch float64 L-BFGS. Only outputs that are identifiable from the
    training labels are optimized: a softmax class never seen in training, or
    a multi-label column with no positives (or no negatives), has an MLE at
    probability 0 (or 1) with an unbounded bias. Those outputs are pinned to
    weight 0 and bias -/+PIN and reported, instead of letting the bias diverge.
    Convergence is judged on a parameter-scale tolerance tol * max(1, lambda).
    """
    PIN = 30.0
    Xt = torch.from_numpy(X).double()
    d = X.shape[1]
    if ml_form == "multi_cls":
        present = np.unique(Y.astype(int))
        active = np.zeros(k, bool); active[present] = True
        remap = -np.ones(k, int); remap[present] = np.arange(len(present))
        Yt = torch.from_numpy(remap[Y.astype(int)]).long()
        pinned_bias = np.full(k, -PIN)
    else:
        Ym = Y.reshape(len(Y), -1)
        pos = Ym.sum(0)
        active = (pos > 0) & (pos < len(Ym))
        Yt = torch.from_numpy(Ym[:, active]).double()
        pinned_bias = np.where(pos == 0, -PIN, PIN)
    ka = int(active.sum())
    W = torch.zeros(d + 1, ka, dtype=torch.float64)
    if W0 is not None:
        W = torch.from_numpy(W0[:, active].numpy().copy()).double()
    W.requires_grad_(True)
    opt = torch.optim.LBFGS([W], lr=1, max_iter=200, history_size=50, line_search_fn="strong_wolfe",
                            tolerance_grad=tol * max(1.0, lam), tolerance_change=1e-14)

    def loss_fn():
        logits = Xt @ W[:-1] + W[-1]
        loss = F.cross_entropy(logits, Yt) if ml_form == "multi_cls" else F.binary_cross_entropy_with_logits(logits, Yt)
        return loss + 0.5 * lam * (W[:-1] ** 2).sum()

    def closure():
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    thresh = tol * max(1.0, lam)
    for rounds in range(1, max_rounds + 1):
        opt.step(closure)
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        gmax = W.grad.abs().max().item()
        if gmax < thresh:
            break
    full = torch.zeros(d + 1, k, dtype=torch.float64)
    full[-1] = torch.from_numpy(pinned_bias).double()
    full[:, torch.from_numpy(active)] = W.detach()
    return full, {"converged": gmax < thresh, "grad_max": gmax, "grad_tol": thresh, "lbfgs_rounds": rounds,
                  "train_objective": loss.item(), "pinned_outputs": np.where(~active)[0].tolist()}


def logits_of(W, X):
    return (torch.from_numpy(X).double() @ W[:-1] + W[-1]).numpy()


def val_loss(logits, y, ml_form):
    lt = torch.from_numpy(logits)
    if ml_form == "multi_cls":
        return F.cross_entropy(lt, torch.from_numpy(y).long()).item()
    return F.binary_cross_entropy_with_logits(lt, torch.from_numpy(y.reshape(len(y), -1)).double()).item()


def fit_logistic(Xtr, ytr, Xva, yva, ml_form, k):
    grid = sorted(LAMBDA_GRID_LOGIT, reverse=True)  # large -> small, warm-started path
    expansions, results, W = 0, {}, None
    while True:
        for lam in grid:
            if lam in results:
                W = results[lam][1]
                continue
            W, info = lbfgs_fit(Xtr, ytr, ml_form, lam, k, W0=W)
            results[lam] = (val_loss(logits_of(W, Xva), yva, ml_form), W, info)
        order = sorted(results)
        best = min(order, key=lambda l: results[l][0])
        edge = best in (order[0], order[-1])
        if not edge or expansions >= 2:
            break
        grid = [order[0] / 10, order[0] / 100] if best == order[0] else [order[-1] * 10, order[-1] * 100]
        grid = sorted(set(grid), reverse=True)
        W = results[best][1]
        expansions += 1
    vl, W, info = results[best]
    nonconv = [l for l, (_, _, i) in results.items() if not i["converged"]]
    return (lambda X: logits_of(W, X)), {"lambda": best, "val_loss": vl, "grid_expansions": expansions,
                                         "boundary_optimum": edge, "selected": info,
                                         "nonconverged_lambdas": nonconv}


def fit_historical(Xtr, ytr, Xva, yva, ml_form, k, seed=42, epochs=20, batch=16, accum=2):
    """Replay of the paper's probe: nn.Linear on raw h, AdamW, val-loss selection."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    d = Xtr.shape[1]
    head = torch.nn.Linear(d, k)
    opt = torch.optim.AdamW(head.parameters(), lr=4e-4, weight_decay=0.0)
    Xt, Xv = torch.from_numpy(Xtr).float(), torch.from_numpy(Xva).float()
    yt, yv = torch.from_numpy(ytr), torch.from_numpy(yva)

    def loss_of(logits, y):
        if ml_form == "binary_cls":
            return F.binary_cross_entropy_with_logits(logits.squeeze(-1), y.float())
        if ml_form == "multi_cls":
            return F.cross_entropy(logits, y.long())
        if ml_form == "multi_label_cls":
            return F.binary_cross_entropy_with_logits(logits, y.float())
        out = logits.squeeze(-1) if logits.shape[-1] == 1 else logits
        return F.mse_loss(out, y.float())

    best = (float("inf"), None, -1)
    n = len(Xt)
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=gen)
        nb = n // batch  # drop_last=True, as in the historical train loader
        opt.zero_grad()
        for b in range(nb):
            idx = perm[b * batch:(b + 1) * batch]
            (loss_of(head(Xt[idx]), yt[idx]) / accum).backward()
            if (b + 1) % accum == 0 or b + 1 == nb:
                opt.step()
                opt.zero_grad()
        with torch.no_grad():
            vl = loss_of(head(Xv), yv).item()
        if vl < best[0]:
            best = (vl, {k_: v.clone() for k_, v in head.state_dict().items()}, epoch)
    last_state = {k_: v.clone() for k_, v in head.state_dict().items()}

    def predictor(state):
        def f(X):
            h = torch.nn.Linear(d, k)
            h.load_state_dict(state)
            with torch.no_grad():
                return h(torch.from_numpy(X).float()).double().numpy()
        return f

    return {"best": predictor(best[1]), "last": predictor(last_state)}, {"best_epoch": best[2], "best_val_loss": best[0]}


# --------------------------------------------------------------------------- metrics

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def pick_threshold(yv, pv, multilabel):
    def f1(t):
        pred = (pv >= t).astype(int)
        return f1_score(yv, pred, average="macro" if multilabel else "binary", zero_division=0)
    scores = [f1(t) for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(scores))])


def metrics_multilabel(y, p, thr, evaluable):
    ye, pe = y[:, evaluable], p[:, evaluable]
    per = {"auroc": [], "ap": []}
    for j in range(ye.shape[1]):
        per["auroc"].append(float(roc_auc_score(ye[:, j], pe[:, j])))
        per["ap"].append(float(average_precision_score(ye[:, j], pe[:, j])))
    out = {
        "macro_auroc": float(np.mean(per["auroc"])), "macro_ap": float(np.mean(per["ap"])),
        "f1_macro_at_0.5_all_labels": float(f1_score(y, (p >= 0.5).astype(int), average="macro", zero_division=0)),
        "f1_macro_at_0.5": float(f1_score(ye, (pe >= 0.5).astype(int), average="macro", zero_division=0)),
        "f1_macro_at_val_thr": float(f1_score(ye, (pe >= thr).astype(int), average="macro", zero_division=0)),
        "val_threshold": thr,
        "cardinality_true": float(y.sum(1).mean()),
        "cardinality_pred_at_0.5": float((p >= 0.5).sum(1).mean()),
        "cardinality_pred_at_val_thr": float((p >= thr).sum(1).mean()),
        "per_label_auroc": per["auroc"], "per_label_ap": per["ap"],
    }
    return out


def metrics_multiclass(y, p, k):
    y = y.astype(int)
    present = sorted(set(y.tolist()))
    pred = p.argmax(1)
    recalls = [float((pred[y == c] == c).mean()) for c in present]
    aurocs, aps = [], []
    for c in present:
        yc = (y == c).astype(int)
        if 0 < yc.sum() < len(yc):
            aurocs.append(float(roc_auc_score(yc, p[:, c])))
            aps.append(float(average_precision_score(yc, p[:, c])))
    return {"top1": float((pred == y).mean()), "macro_recall": float(np.mean(recalls)),
            "macro_ovr_auroc": float(np.mean(aurocs)), "macro_ovr_ap": float(np.mean(aps)),
            "classes_present": len(present)}


def metrics_binary(y, p, thr):
    return {"auroc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)),
            "f1_at_0.5": float(f1_score(y, p >= 0.5, zero_division=0)),
            "f1_at_val_thr": float(f1_score(y, p >= thr, zero_division=0)), "val_threshold": thr,
            "balanced_acc_at_0.5": float(0.5 * (((p >= 0.5) & (y == 1)).sum() / max((y == 1).sum(), 1)
                                          + ((p < 0.5) & (y == 0)).sum() / max((y == 0).sum(), 1)))}


def metrics_regression(y, pred):
    y2, p2 = y.reshape(len(y), -1), pred.reshape(len(pred), -1)
    return {"r2": float(r2_score(y2, p2)), "mae": float(mean_absolute_error(y2, p2)),
            "rmse": float(np.sqrt(np.mean((y2 - p2) ** 2))),
            "r2_per_output": [float(r2_score(y2[:, j], p2[:, j])) for j in range(y2.shape[1])]}


def evaluate(task, split_scores, ytr, yva, yte):
    """Metrics for one probe on the shared test rows, plus threshold chosen on val."""
    form = task["ml_form"]
    sv, st = split_scores
    if form == "multi_label_cls":
        pv, pt = sigmoid(sv), sigmoid(st)
        thr = pick_threshold(yva, pv, multilabel=True)
        return metrics_multilabel(yte, pt, thr, task["evaluable"])
    if form == "binary_cls":
        pv, pt = sigmoid(sv).ravel(), sigmoid(st).ravel()
        return metrics_binary(yte, pt, pick_threshold(yva, pv, multilabel=False))
    if form == "multi_cls":
        return metrics_multiclass(yte, softmax(st), st.shape[1])
    return metrics_regression(yte, st)


def trivial_baselines(task, ytr, yva, yte):
    form = task["ml_form"]
    if form == "multi_label_cls":
        prev = ytr.mean(0)
        ev = task["evaluable"]
        pv = np.tile(prev, (len(yva), 1))
        pt = np.tile(prev, (len(yte), 1))
        thr = pick_threshold(yva, pv, multilabel=True)
        return {
            "always_positive": {"f1_macro": float(f1_score(yte[:, ev], np.ones_like(yte[:, ev]), average="macro", zero_division=0)),
                                "cardinality_pred": float(yte.shape[1])},
            "always_negative": {"f1_macro": float(f1_score(yte[:, ev], np.zeros_like(yte[:, ev]), average="macro", zero_division=0)),
                                "cardinality_pred": 0.0},
            "train_prevalence": {"macro_auroc": 0.5, "macro_ap": float(np.mean(yte[:, ev].mean(0))),
                                 "f1_macro_at_val_thr": float(f1_score(yte[:, ev], (pt[:, ev] >= thr).astype(int), average="macro", zero_division=0)),
                                 "val_threshold": thr},
        }
    if form == "binary_cls":
        prev = float(ytr.mean())
        return {"always_positive_f1": float(f1_score(yte, np.ones_like(yte), zero_division=0)),
                "train_prevalence": {"auroc": 0.5, "ap": float(yte.mean()), "value": prev}}
    if form == "multi_cls":
        maj = int(np.bincount(ytr.astype(int)).argmax())
        present = sorted(set(yte.astype(int).tolist()))
        return {"majority_class": {"class": maj, "top1": float((yte == maj).mean()), "macro_recall": 1.0 / len(present)}}
    mean = ytr.mean(0)
    return {"train_mean": metrics_regression(yte, np.tile(mean, (len(yte), 1)) if np.ndim(mean) else np.full(len(yte), mean))}


# --------------------------------------------------------------------------- main

def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--tier", choices=["core", "rest", "all"], default="core")
    parser.add_argument("--feat-tag", default="p1")
    parser.add_argument("--out-tag", default="p2")
    parser.add_argument("--out-name", default=None, help="result folder name (default: --encoder); P1b maps orig_cecl_s1 -> cecl_s1")
    parser.add_argument("--tasks", nargs="*", default=None, help="restrict to these task names (pilot)")
    parser.add_argument("--skip-historical-protocol", action="store_true")
    parser.add_argument("--wait-minutes", type=int, default=150, help="wait this long for shards still being extracted")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    data_root = Path(os.environ["DATA_BASE_PATH"])
    out_root = Path(os.environ["OUTPUT_BASE_PATH"]) / "iclr2027_probes" / COHORT / args.out_tag / (args.out_name or args.encoder)
    out_root.mkdir(parents=True, exist_ok=True)
    feat_dir = Path(os.environ["OUTPUT_BASE_PATH"]) / "iclr2027_features" / COHORT / args.feat_tag
    # Rest tasks can reuse windows that were encoded in the core tier (each window is
    # encoded once), so the rest tier always loads core + rest features.
    tiers = ["core"] if args.tier == "core" else ["core", "rest"]
    index, H = load_features(feat_dir, args.encoder, tiers, wait_sec=args.wait_minutes * 60)
    rprint(f"[blue]->[/blue] {args.encoder}: {len(index):,} cached windows")

    defs = pl.read_csv(data_root / "mirage" / "labels" / "task_definitions.csv")
    defined = set(defs["task_id"].to_list())
    csvs = sorted(p.stem for p in (data_root / "mirage" / "labels" / tasks_folder()).glob("*.csv"))
    undefined = [t for t in csvs if t not in defined]
    if undefined:
        # CSVs in the task folder that task_definitions.csv does not define are not benchmark tasks.
        rprint(f"[yellow]skip[/yellow] task CSVs absent from task_definitions.csv: {undefined}")
    all_hist = [t for t in csvs if t in defined]
    hist_names = [t for t in all_hist if (t in CORE_TASKS) == (args.tier == "core") or args.tier == "all"]
    tasks = [historical_task(data_root, t) for t in hist_names]
    use_flag = dict(zip(defs["task_id"].to_list(), defs["use_in_benchmark"].to_list()))
    if args.tier in ("core", "all"):
        tasks += matched_tasks(data_root)
    if args.tasks:
        tasks = [t for t in tasks if t["task"] in args.tasks or t["task"].split("__")[0] in args.tasks]

    for task in tasks:
        name = task["task"]
        out_json = out_root / f"{name}.json"
        if out_json.exists():
            rprint(f"[green]SKIP[/green] {name}")
            continue
        # Cooperative locking: several jobs on one encoder split its tasks instead of
        # duplicating them (and never write the same task concurrently). A lock older
        # than LOCK_STALE_SEC belongs to a dead job and is taken over.
        lock = out_root / f"{name}.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, os.environ.get("SLURM_JOB_ID", str(os.getpid())).encode())
            os.close(fd)
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
            except FileNotFoundError:
                age = LOCK_STALE_SEC + 1
            if age < LOCK_STALE_SEC:
                rprint(f"[cyan]LOCKED[/cyan] {name} (another job, {age:.0f}s)")
                continue
            lock.touch()
            rprint(f"[yellow]TAKEOVER[/yellow] stale lock {name} ({age:.0f}s)")
        if out_json.exists():  # finished by another job between the two checks
            lock.unlink(missing_ok=True)
            continue
        t0 = time.time()
        rows = np.array([index.get(w, -1) for w in task["window_id"]])
        missing = int((rows < 0).sum())
        if missing:
            raise KeyError(f"{name}: {missing} rows have no cached features")
        X = H[rows].astype(np.float64)
        y = task["y"]
        split = task["partition"]
        tr, va, te = (split == "train"), (split == "val"), (split == "test")
        Xtr, Xva, Xte = X[tr], X[va], X[te]
        ytr, yva, yte = y[tr], y[va], y[te]
        form = task["ml_form"]
        if form == "multi_label_cls":
            task["evaluable"] = np.where((yte.sum(0) > 0) & (yte.sum(0) < len(yte)))[0]
        k = (1 if form == "binary_cls" else ytr.shape[1] if form == "multi_label_cls"
             else NUM_REGIONS if form == "multi_cls" and task["family"] == "matched"
             else (task["num_classes"] or int(y.max()) + 1) if form == "multi_cls"
             else (ytr.shape[1] if ytr.ndim == 2 else 1))

        (Str, Sva, Ste), std_info = standardize(Xtr, Xva, Xte)
        record = {
            "encoder": args.out_name or args.encoder, "source_encoder": args.encoder, "task": name, "family": task["family"], "ml_form": form,
            "rows": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
            "test_matches": int(len(set(task["match_id"][te].tolist()))),
            "candidates": task.get("candidates"),
            "standardization": std_info, "probes": {}, "tick_norm_derived": task.get("tick_norm_derived", False),
            "use_in_benchmark": use_flag.get(name.split("__")[0], "matched") if task["family"] == "historical" else "matched",
            "baselines": trivial_baselines(task, ytr, yva, yte),
        }
        if form == "multi_label_cls":
            record["evaluable_labels"] = task["evaluable"].tolist()
            record["omitted_labels"] = sorted(set(range(ytr.shape[1])) - set(task["evaluable"].tolist()))
            record["test_support"] = yte.sum(0).astype(int).tolist()
        exports = {"window_id": task["window_id"], "match_id": task["match_id"], "partition": split, "y": y}

        if form == "regression":
            f, info = fit_ridge(Str, ytr, Sva, yva)
        else:
            f, info = fit_logistic(Str, ytr, Sva, yva, form, k)
        sv, st = f(Sva), f(Ste)
        record["probes"]["converged"] = {"fit": info, "test": evaluate(task, (sv, st), ytr, yva, yte)}
        exports["converged_val"], exports["converged_test"] = sv, st

        if not args.skip_historical_protocol:
            preds, hinfo = fit_historical(Xtr, ytr, Xva, yva, form, k)
            for which in ("best", "last"):
                sv, st = preds[which](Xva), preds[which](Xte)
                record["probes"][f"historical_{which}"] = {"fit": hinfo, "test": evaluate(task, (sv, st), ytr, yva, yte)}
                exports[f"historical_{which}_val"], exports[f"historical_{which}_test"] = sv, st

        np.savez_compressed(out_root / f"{name}.npz.tmp{os.getpid()}.npz", **{k_: (v.astype("U64") if v.dtype.kind in "OU" else v)
                                                          for k_, v in exports.items()})
        os.replace(out_root / f"{name}.npz.tmp{os.getpid()}.npz", out_root / f"{name}.npz")
        record["seconds"] = round(time.time() - t0, 1)
        tmp = out_json.with_suffix(f".json.tmp{os.getpid()}")
        tmp.write_text(json.dumps(record, indent=1, default=float))
        os.replace(tmp, out_json)
        lock.unlink(missing_ok=True)

        c = record["probes"]["converged"]["test"]
        headline = (c.get("macro_auroc") or c.get("auroc") or c.get("top1") or c.get("r2"))
        rprint(f"  {name:36s} conv={headline:.4f} lam={record['probes']['converged']['fit']['lambda']:.0e} "
               f"({record['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
