"""
Compute CECL-minus-baseline deltas on evaluation subsets.

Reads the dumped per-sample logits, joins them to the subset flags, and reports
the delta on the full test split alongside each restricted condition. Metrics
mirror the training code exactly so the full-split numbers are directly
comparable to the published ones:

    multi_label_cls  macro F1 over labels, sigmoid at threshold 0.5
    binary_cls       accuracy, sigmoid at threshold 0.5
    multi_cls        accuracy, argmax over logits
    regression       R^2

--verify recomputes the full-split metric from the dumped logits and compares it
against that run's test_results_best.json. Run it first. If those agree, the
subset slices provably come from the same predictions behind the published
deltas; if they do not, nothing downstream is trustworthy.

Usage:
    python -m src.scripts.eval_subsets.analyze_subsets --verify
    python -m src.scripts.eval_subsets.analyze_subsets --out reviews/subset-results.md
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# task -> (subset file suffix, [(label, query) ...])
CONDITIONS = {
    'enemy_location_10s': ('moved', [('target changed region', 'moved_disjoint == 1')]),
    'enemy_location_5s': ('moved', [('target changed region', 'moved_disjoint == 1')]),
    'teammate_location_10s': ('moved', [('target changed region', 'moved_disjoint == 1')]),
    'global_bombPlanted': ('bomb', [('plant before window', 'plant_before_window == 1')]),
    'enemy_aliveCount': ('alive', [('deaths precede window', 'unobservable_count == 1')]),
    'teammate_aliveCount': ('alive', [('deaths precede window', 'unobservable_count == 1')]),
}
JACCARD_BINS = [(-0.01, 0.0), (0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def metric(ml_form: str, logits: np.ndarray, targets: np.ndarray) -> float:
    if ml_form == 'multi_label_cls':
        pred = sigmoid(logits) > 0.5
        true = targets > 0.5
        f1s = []
        for j in range(true.shape[1]):
            tp = int(np.sum(pred[:, j] & true[:, j]))
            fp = int(np.sum(pred[:, j] & ~true[:, j]))
            fn = int(np.sum(~pred[:, j] & true[:, j]))
            denom = 2 * tp + fp + fn
            f1s.append(0.0 if denom == 0 else 2 * tp / denom)
        return float(np.mean(f1s))
    if ml_form == 'binary_cls':
        pred = sigmoid(logits.reshape(len(logits), -1)[:, 0]) > 0.5
        return float(np.mean(pred == (targets.reshape(len(targets), -1)[:, 0] > 0.5)))
    if ml_form == 'multi_cls':
        pred = logits.argmax(axis=1)
        return float(np.mean(pred == targets.reshape(len(targets), -1)[:, 0].astype(int)))
    if ml_form == 'regression':
        y = targets.reshape(len(targets), -1)
        p = logits.reshape(len(logits), -1)
        ss_res = float(np.sum((y - p) ** 2))
        ss_tot = float(np.sum((y - y.mean(axis=0)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    raise ValueError(ml_form)


def load_run(output_base: Path, run_dir: str):
    d = output_base / run_dir
    pq = d / 'test_predictions_best.parquet'
    if not pq.exists():
        alt = d / 'test_predictions_best.csv'
        if not alt.exists():
            return None
        frame = pd.read_csv(alt)
    else:
        frame = pd.read_parquet(pq)
    hp = yaml.safe_load((d / 'hparam.yaml').read_text(encoding='utf-8'))
    ml_form = (hp.get('task') or {}).get('ml_form')
    lcols = sorted([c for c in frame.columns if c.startswith('logit_')],
                   key=lambda c: int(c.split('_')[1]))
    tcols = sorted([c for c in frame.columns if c.startswith('target_')],
                   key=lambda c: int(c.split('_')[1]))
    return {
        'idx': frame['original_csv_idx'].to_numpy(),
        'logits': frame[lcols].to_numpy(dtype=float),
        'targets': frame[tcols].to_numpy(dtype=float),
        'ml_form': ml_form,
        'reported': _reported(d),
    }


def _reported(run_path: Path):
    f = run_path / 'test_results_best.json'
    if not f.exists():
        return None
    j = json.loads(f.read_text(encoding='utf-8'))
    key = {'binary_cls': 'acc', 'multi_cls': 'acc',
           'multi_label_cls': 'f1', 'regression': 'r2'}.get(j.get('ml_form'))
    return (j.get('metrics') or {}).get(key)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', default='src/scripts/eval_subsets/dump_manifest.tsv')
    ap.add_argument('--output-base', default=None,
                    help='dir holding the probe run dirs (default: $OUTPUT_BASE_PATH or output)')
    ap.add_argument('--subset-dir', default='output/eval_subsets')
    ap.add_argument('--verify', action='store_true',
                    help='only check dumped logits reproduce test_results_best.json')
    ap.add_argument('--out', default=None, help='write a markdown report here')
    args = ap.parse_args()

    import os
    base = Path(args.output_base or os.environ.get('OUTPUT_BASE_PATH', 'output'))
    man = pd.read_csv(args.manifest, sep='\t')

    runs, missing = {}, []
    for r in man.to_dict('records'):
        loaded = load_run(base, r['run_dir'])
        if loaded is None:
            missing.append(r['run_dir'])
            continue
        runs[(r['arm'], r['map'], r['enc'], r['task'], r['seed'])] = loaded

    print(f'loaded {len(runs)} runs, {len(missing)} missing')
    if missing:
        for m in missing[:10]:
            print(f'  MISSING {m}')
        if len(missing) > 10:
            print(f'  ... and {len(missing) - 10} more')

    # ---- verification -----------------------------------------------------
    diffs = []
    for k, v in runs.items():
        if v['reported'] is None:
            continue
        got = metric(v['ml_form'], v['logits'], v['targets'])
        diffs.append((abs(got - v['reported']), k, got, v['reported']))
    diffs.sort(reverse=True)
    if diffs:
        worst = diffs[0]
        med = statistics.median(d[0] for d in diffs)
        print(f'\nverification over {len(diffs)} runs: median |diff| {med:.5f}, max {worst[0]:.5f}')
        for d, k, got, rep in diffs[:5]:
            flag = 'OK  ' if d < 1e-3 else 'BAD '
            print(f'  {flag}{d:.5f}  recomputed {got:.4f} vs reported {rep:.4f}  {k}')
        if worst[0] >= 1e-3:
            print('\n  WARNING: at least one run does not reproduce its reported metric.')
            print('  Do not trust the subset numbers until this is resolved.')
    if args.verify:
        return

    # ---- subset deltas ----------------------------------------------------
    lines = ['# Subset analysis', '',
             'Delta is CECL minus baseline, seeds averaged within each arm before',
             'differencing. `n` is the number of test rows in the condition.', '']
    cells = sorted({(m, e) for (_, m, e, _, _) in runs})

    for task, (suffix, conds) in CONDITIONS.items():
        task_keys = sorted({(m, e) for (a, m, e, t, s) in runs if t == task})
        if not task_keys:
            continue
        lines += [f'## {task}', '',
                  '| condition | mean delta | cells positive | median n |', '|---|---|---|---|']

        for label, query in [('full test split', None)] + conds:
            per_cell, ns = [], []
            for (mp, enc) in task_keys:
                mask_idx = None
                if query is not None:
                    sf = Path(args.subset_dir) / f'{mp}-{task}-{suffix}.csv'
                    if not sf.exists():
                        continue
                    flags = pd.read_csv(sf)
                    mask_idx = set(flags.query(query)['idx'].tolist())
                    if not mask_idx:
                        continue

                arm_vals = {}
                for arm in ('BASE', 'CECL'):
                    seed_vals = []
                    for seed in (1, 2):
                        v = runs.get((arm, mp, enc, task, seed))
                        if v is None:
                            continue
                        if mask_idx is None:
                            sel = np.ones(len(v['idx']), dtype=bool)
                        else:
                            sel = np.isin(v['idx'], list(mask_idx))
                        if sel.sum() < 10:
                            continue
                        seed_vals.append(metric(v['ml_form'], v['logits'][sel], v['targets'][sel]))
                        ns.append(int(sel.sum()))
                    if seed_vals:
                        arm_vals[arm] = statistics.fmean(seed_vals)
                if len(arm_vals) == 2:
                    per_cell.append(arm_vals['CECL'] - arm_vals['BASE'])

            if per_cell:
                pos = sum(1 for x in per_cell if x > 0)
                lines.append(f'| {label} | {statistics.fmean(per_cell):+.4f} | '
                             f'{pos}/{len(per_cell)} | {int(statistics.median(ns)) if ns else 0} |')
            else:
                lines.append(f'| {label} | no data | 0/0 | 0 |')
        lines.append('')

        # jaccard-binned view for the forecast tasks
        if suffix == 'moved':
            lines += ['Delta by how much the present answer overlaps the future one. '
                      'Lower jaccard means less can be carried forward.', '',
                      '| jaccard | mean delta | cells positive | median n |', '|---|---|---|---|']
            for lo, hi in JACCARD_BINS:
                per_cell, ns = [], []
                for (mp, enc) in task_keys:
                    sf = Path(args.subset_dir) / f'{mp}-{task}-moved.csv'
                    if not sf.exists():
                        continue
                    flags = pd.read_csv(sf)
                    sub = flags[(flags['jaccard'] > lo) & (flags['jaccard'] <= hi)]
                    keep = set(sub['idx'].tolist())
                    if len(keep) < 10:
                        continue
                    arm_vals = {}
                    for arm in ('BASE', 'CECL'):
                        seed_vals = []
                        for seed in (1, 2):
                            v = runs.get((arm, mp, enc, task, seed))
                            if v is None:
                                continue
                            sel = np.isin(v['idx'], list(keep))
                            if sel.sum() < 10:
                                continue
                            seed_vals.append(metric(v['ml_form'], v['logits'][sel], v['targets'][sel]))
                            ns.append(int(sel.sum()))
                        if seed_vals:
                            arm_vals[arm] = statistics.fmean(seed_vals)
                    if len(arm_vals) == 2:
                        per_cell.append(arm_vals['CECL'] - arm_vals['BASE'])
                if per_cell:
                    pos = sum(1 for x in per_cell if x > 0)
                    lines.append(f'| {lo:.2f}-{hi:.2f} | {statistics.fmean(per_cell):+.4f} | '
                                 f'{pos}/{len(per_cell)} | {int(statistics.median(ns))} |')
            lines.append('')

    report = '\n'.join(lines)
    print('\n' + report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report, encoding='utf-8')
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
