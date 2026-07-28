"""
Why does a dumped run not reproduce its reported metric?

Prints, for one run, every plausible reading of the dumped logits and targets
next to the reference value, so the mismatch is identified rather than guessed
at. Covers the two candidates that matter here:

  average    torchmetrics MulticlassAccuracy defaults to average='macro', so the
             published "acc" for multi_cls is per-class recall averaged over
             classes, not plain accuracy.
  coverage   whether the dump holds every test row, or the metric was computed
             over a different population than the one dumped.

Usage:
    python -m src.scripts.eval_subsets.diagnose_metric \
        probe-dust2-siglip2-enemy_aliveCount-all-260502-013848-p62n
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DEFAULT_OUTPUT_BASE = '/project2/ustun_1726/x-ego/output'


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--output-base', default=None)
    ap.add_argument('--reference', default='src/scripts/eval_subsets/reference_metrics.tsv')
    args = ap.parse_args()

    base = Path(args.output_base or os.environ.get('OUTPUT_BASE_PATH', DEFAULT_OUTPUT_BASE))
    d = base / args.run_dir

    pq = next((d / n for n in ('test_predictions_last.parquet',
                               'test_predictions_best.parquet') if (d / n).exists()), None)
    if pq is None:
        raise SystemExit(f'no dump under {d}')
    frame = pd.read_parquet(pq)

    hp = yaml.safe_load((d / 'hparam.yaml').read_text(encoding='utf-8'))
    task = hp.get('task') or {}
    ml_form = task.get('ml_form')

    lcols = sorted([c for c in frame.columns if c.startswith('logit_')],
                   key=lambda c: int(c.split('_')[1]))
    tcols = sorted([c for c in frame.columns if c.startswith('target_')],
                   key=lambda c: int(c.split('_')[1]))
    logits = frame[lcols].to_numpy(dtype=np.float64)
    targets = frame[tcols].to_numpy(dtype=np.float64)

    print(f'run        {args.run_dir}')
    print(f'dump       {pq.name}')
    print(f'ml_form    {ml_form}')
    print(f'task_id    {task.get("task_id")}')
    print(f'rows       {len(frame)}')
    print(f'logit cols {len(lcols)}   target cols {len(tcols)}')
    print(f'columns    {[c for c in frame.columns if not c.startswith(("logit_", "target_"))]}')
    print(f'unique idx {frame["original_csv_idx"].nunique()}')

    ref = {}
    rp = Path(args.reference)
    if rp.exists():
        r = pd.read_csv(rp, sep='\t')
        ref = {(x['run_dir'], x['which']): (x['metric'], float(x['value']))
               for x in r.to_dict('records')}
    for which in ('last', 'best'):
        if (args.run_dir, which) in ref:
            m, v = ref[(args.run_dir, which)]
            print(f'reference  {which:5s} {m} = {v:.6f}')

    for which in ('last', 'best'):
        f = d / f'test_results_{which}.json'
        if f.exists():
            j = json.loads(f.read_text(encoding='utf-8'))
            print(f'json       {which:5s} {j.get("metrics")}')
            if 'num_test_samples' in j:
                print(f'           num_test_samples = {j["num_test_samples"]}')

    print()
    if ml_form == 'multi_cls':
        y = targets.reshape(len(targets), -1)
        print(f'target col 0 uniques  {np.unique(y[:, 0])[:12]}')
        if y.shape[1] > 1:
            print(f'target rowsums uniques {np.unique(y.sum(axis=1))[:6]}  '
                  f'(all 1.0 means the target is one hot, not a class index)')
        true = y[:, 0].astype(int) if y.shape[1] == 1 else y.argmax(axis=1)
        pred = logits.argmax(axis=1)
        n_cls = len(lcols)
        print(f'\nnum_classes from logits {n_cls}, true label range '
              f'[{true.min()}, {true.max()}]')

        micro = float(np.mean(pred == true))
        rec, sup = [], []
        for c in range(n_cls):
            m = true == c
            sup.append(int(m.sum()))
            rec.append(float(np.mean(pred[m] == c)) if m.any() else None)
        macro_all = float(np.mean([0.0 if r is None else r for r in rec]))
        present = [r for r in rec if r is not None]
        macro_present = float(np.mean(present)) if present else float('nan')

        print(f'\nmicro accuracy                      {micro:.6f}')
        print(f'macro recall, empty classes as 0    {macro_all:.6f}   <- torchmetrics default')
        print(f'macro recall, empty classes skipped {macro_present:.6f}')
        print(f'\nper class support and recall')
        for c, (s, r) in enumerate(zip(sup, rec)):
            print(f'  class {c}  n={s:6d}  recall={"n/a" if r is None else f"{r:.4f}"}')
    elif ml_form == 'multi_label_cls':
        pred = sigmoid(logits) > 0.5
        true = targets > 0.5
        f1s, f1s_present = [], []
        for j in range(true.shape[1]):
            tp = int(np.sum(pred[:, j] & true[:, j]))
            fp = int(np.sum(pred[:, j] & ~true[:, j]))
            fn = int(np.sum(~pred[:, j] & true[:, j]))
            denom = 2 * tp + fp + fn
            f1 = 0.0 if denom == 0 else 2 * tp / denom
            f1s.append(f1)
            if true[:, j].any():
                f1s_present.append(f1)
        print(f'macro F1, empty labels as 0    {float(np.mean(f1s)):.6f}   <- torchmetrics default')
        print(f'macro F1, empty labels skipped '
              f'{float(np.mean(f1s_present)) if f1s_present else float("nan"):.6f}')
        print(f'labels with no positives      '
              f'{int(sum(1 for j in range(true.shape[1]) if not true[:, j].any()))} '
              f'of {true.shape[1]}')
    elif ml_form == 'binary_cls':
        p = sigmoid(logits.reshape(len(logits), -1)[:, 0]) > 0.5
        t = targets.reshape(len(targets), -1)[:, 0] > 0.5
        print(f'accuracy {float(np.mean(p == t)):.6f}')

    print('\nCompare each candidate against the reference above. The one that '
          'matches is the definition the training code used.')


if __name__ == '__main__':
    main()
