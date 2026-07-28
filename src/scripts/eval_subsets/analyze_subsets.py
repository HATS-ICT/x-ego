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
against reference_metrics.tsv, which was extracted BEFORE any rerun. Comparing
against the run's own test_results_last.json would be circular, since the rerun
overwrites that file in place. Run --verify first. If the two agree, the subset
slices provably come from the same predictions behind the published deltas; if
they do not, nothing downstream is trustworthy.

Runs are located under $OUTPUT_BASE_PATH, or --output-base, defaulting to the
same path as scripts/dump_test_predictions.sh.

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

# Same default as scripts/dump_test_predictions.sh. Keep the two in step.
DEFAULT_OUTPUT_BASE = '/project2/ustun_1726/x-ego/output'


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
        # MulticlassAccuracy is built without an `average` argument
        # (downstream.py:185,517) and torchmetrics defaults to average='macro',
        # so the published "acc" is per-class recall averaged over classes, not
        # plain accuracy. On enemy_aliveCount the two differ by 0.09.
        y = targets.reshape(len(targets), -1)
        true = y[:, 0].astype(int) if y.shape[1] == 1 else y.argmax(axis=1)
        pred = logits.argmax(axis=1)
        recalls = []
        for c in range(logits.shape[1]):
            m = true == c
            recalls.append(float(np.mean(pred[m] == c)) if m.any() else 0.0)
        return float(np.mean(recalls))
    if ml_form == 'regression':
        y = targets.reshape(len(targets), -1)
        p = logits.reshape(len(logits), -1)
        ss_res = float(np.sum((y - p) ** 2))
        ss_tot = float(np.sum((y - y.mean(axis=0)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    raise ValueError(ml_form)


def load_run(output_base: Path, run_dir: str, reference: dict | None = None):
    """Load dumped predictions. `--mode test` evaluates the 'last' checkpoint, so
    that file is preferred; '_best' is accepted for runs dumped during training."""
    d = output_base / run_dir
    frame = None
    for name in ('test_predictions_last.parquet', 'test_predictions_best.parquet',
                 'test_predictions_last.csv', 'test_predictions_best.csv'):
        f = d / name
        if f.exists():
            frame = pd.read_parquet(f) if f.suffix == '.parquet' else pd.read_csv(f)
            which = 'last' if 'last' in name else 'best'
            break
    if frame is None:
        return None
    hp = yaml.safe_load((d / 'hparam.yaml').read_text(encoding='utf-8'))
    ml_form = (hp.get('task') or {}).get('ml_form')
    metric_key = {'multi_label_cls': 'f1', 'regression': 'r2'}.get(ml_form, 'acc')

    # Written by torchmetrics in the same process that produced the parquet, so
    # comparing it against a recompute from the raw logits is an independent
    # check of the metric definition. This is check A below.
    self_reported = None
    jf = d / f'test_results_{which}.json'
    if jf.exists():
        v = (json.loads(jf.read_text(encoding='utf-8')).get('metrics') or {}).get(metric_key)
        self_reported = float(v) if v is not None else None
    lcols = sorted([c for c in frame.columns if c.startswith('logit_')],
                   key=lambda c: int(c.split('_')[1]))
    tcols = sorted([c for c in frame.columns if c.startswith('target_')],
                   key=lambda c: int(c.split('_')[1]))
    return {
        'idx': frame['original_csv_idx'].to_numpy(),
        'logits': frame[lcols].to_numpy(dtype=float),
        'targets': frame[tcols].to_numpy(dtype=float),
        'ml_form': ml_form,
        'which': which,
        # torchmetrics value from this rerun, for the definition check.
        'self_reported': self_reported,
        # Metrics recorded BEFORE this rerun, for the checkpoint-identity check.
        # NOTE the reference table's `which` column is not meaningful: all 1133
        # runs carry identical best and last rows, so these are best-checkpoint
        # values. A dump of `last` will differ slightly wherever best != last.
        'reported': (reference or {}).get((run_dir, which),
                                          (reference or {}).get((run_dir, 'best'))),
    }


def cell_delta(runs: dict, task: str, mp: str, enc: str, keep_idx, min_rows: int = 10):
    """CECL minus baseline for one (map, encoder) cell, restricted to `keep_idx`.

    `keep_idx` of None means the full split. Seeds are averaged within an arm
    before differencing. Returns (delta, median_rows) or None when either arm has
    no seed with at least `min_rows` selected rows, so an under-powered slice is
    dropped rather than reported as a delta of zero.
    """
    arm_vals, ns = {}, []
    for arm in ('BASE', 'CECL'):
        seed_vals = []
        for seed in (1, 2):
            v = runs.get((arm, mp, enc, task, seed))
            if v is None:
                continue
            sel = (np.ones(len(v['idx']), dtype=bool) if keep_idx is None
                   else np.isin(v['idx'], keep_idx))
            if sel.sum() < min_rows:
                continue
            seed_vals.append(metric(v['ml_form'], v['logits'][sel], v['targets'][sel]))
            ns.append(int(sel.sum()))
        if seed_vals:
            arm_vals[arm] = statistics.fmean(seed_vals)
    if len(arm_vals) != 2:
        return None
    return arm_vals['CECL'] - arm_vals['BASE'], (statistics.median(ns) if ns else 0)


def summary_row(label: str, results: list) -> str:
    """One markdown row from a list of (delta, n) per cell."""
    results = [r for r in results if r is not None]
    if not results:
        return f'| {label} | no data | 0/0 | 0 |'
    deltas = [d for d, _ in results]
    pos = sum(1 for d in deltas if d > 0)
    return (f'| {label} | {statistics.fmean(deltas):+.4f} | {pos}/{len(deltas)} | '
            f'{int(statistics.median([n for _, n in results]))} |')


def load_flags(subset_dir: str, mp: str, task: str, suffix: str):
    """Subset flag CSV for one (map, task), or None when it was never built."""
    p = Path(subset_dir) / f'{mp}-{task}-{suffix}.csv'
    return pd.read_csv(p) if p.exists() else None


def load_relay_flags(subset_dir: str, mp: str, task: str):
    """Relay condition CSV for one (map, task), whichever backend pair was used."""
    matches = sorted(Path(subset_dir).glob(f'{mp}-{task}-relay-*.csv'))
    if not matches:
        return None, None
    return pd.read_csv(matches[0]), matches[0].name


def load_reference(path: Path) -> dict:
    """Original per-run metrics, keyed by (run_dir, which). Written before any
    rerun, so it is an external check rather than a self-comparison."""
    if not path.exists():
        print(f'WARNING: no reference table at {path}; verification will be skipped')
        return {}
    ref = pd.read_csv(path, sep='	')
    return {(r['run_dir'], r['which']): float(r['value']) for r in ref.to_dict('records')}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', default='src/scripts/eval_subsets/dump_manifest.tsv')
    ap.add_argument('--output-base', default=None,
                    help='dir holding the probe run dirs (default: $OUTPUT_BASE_PATH or output)')
    ap.add_argument('--subset-dir', default='output/eval_subsets')
    ap.add_argument('--reference', default='src/scripts/eval_subsets/reference_metrics.tsv',
                    help='original metrics recorded before any rerun, for --verify')
    ap.add_argument('--verify', action='store_true',
                    help='only check dumped logits reproduce the reference metrics')
    ap.add_argument('--out', default=None, help='write a markdown report here')
    args = ap.parse_args()

    import os
    # Must agree with the default in scripts/dump_test_predictions.sh, otherwise
    # a shell that dumped fine reports every run missing here.
    base = Path(args.output_base or os.environ.get('OUTPUT_BASE_PATH', DEFAULT_OUTPUT_BASE))
    print(f'output base: {base}')
    man = pd.read_csv(args.manifest, sep='\t')
    reference = load_reference(Path(args.reference))

    runs, missing = {}, []
    for r in man.to_dict('records'):
        loaded = load_run(base, r['run_dir'], reference)
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

    if not runs:
        raise SystemExit(
            f'\nERROR: no dumped predictions found under {base}\n'
            '  Point --output-base at the dir holding the probe run dirs, or export\n'
            '  OUTPUT_BASE_PATH. Nothing is written when zero runs load.'
        )

    # ---- verification -----------------------------------------------------
    # Two independent questions, previously conflated into one number.
    #
    #   A  Does a recompute from the raw logits match what torchmetrics computed
    #      in the same process? Catches metric-definition errors in this file.
    #      Must be tight; anything above 1e-2 is a definition mismatch.
    #   B  Does this rerun agree with the metrics recorded before it? Catches a
    #      changed checkpoint. Loose, because the reference holds best-checkpoint
    #      values while --mode test dumps `last`.
    recomputed = {k: metric(v['ml_form'], v['logits'], v['targets']) for k, v in runs.items()}

    def report(label, field, tol, note=''):
        diffs = sorted(((abs(recomputed[k] - v[field]), k, recomputed[k], v[field])
                        for k, v in runs.items() if v.get(field) is not None), reverse=True)
        if not diffs:
            print(f'\n{label}\n  WARNING: 0 of {len(runs)} runs had a value to compare '
                  f'against, so this check did NOT run.')
            return False
        med = statistics.median(d[0] for d in diffs)
        print(f'\n{label}\n  {len(diffs)} runs, median |diff| {med:.5f}, max {diffs[0][0]:.5f}'
              f'{note}')
        by_form = collections.defaultdict(list)
        for d, k, _, _ in diffs:
            by_form[runs[k]['ml_form']].append(d)
        for form, ds in sorted(by_form.items()):
            worst = max(ds)
            print(f'    {"OK  " if worst < tol else "BAD "}{form:16s} n={len(ds):3d} '
                  f'median {statistics.median(ds):.5f}  max {worst:.5f}')
        for d, k, got, rep in diffs[:3]:
            if d >= tol:
                print(f'    worst  recomputed {got:.4f} vs {rep:.4f}  {k}')
        return diffs[0][0] < tol

    ok_a = report('check A, recompute vs this run\'s own torchmetrics value', 'self_reported',
                  1e-2, '  (tolerance 1e-2; small gaps are mixed-precision)')
    report('check B, this run vs metrics recorded before the rerun', 'reported', 5e-2,
           '  (reference holds best-checkpoint values, dumps are last)')
    if not ok_a:
        print('\n  WARNING: the recompute does not match torchmetrics. The metric in '
              'this file is\n  defined differently from the training code. Do not trust '
              'the subset numbers.')
    if args.verify:
        return

    # ---- subset deltas ----------------------------------------------------
    lines = ['# Subset analysis', '',
             'Delta is CECL minus baseline, seeds averaged within each arm before',
             'differencing. `n` is the number of test rows in the condition.', '']
    for task, (suffix, conds) in CONDITIONS.items():
        task_keys = sorted({(m, e) for (a, m, e, t, s) in runs if t == task})
        if not task_keys:
            continue
        lines += [f'## {task}', '',
                  '| condition | mean delta | cells positive | median n |', '|---|---|---|---|']

        for label, query in [('full test split', None)] + conds:
            results = []
            for (mp, enc) in task_keys:
                keep = None
                if query is not None:
                    flags = load_flags(args.subset_dir, mp, task, suffix)
                    if flags is None:
                        continue
                    keep = flags.query(query)['idx'].tolist()
                    if not keep:
                        continue
                results.append(cell_delta(runs, task, mp, enc, keep))
            lines.append(summary_row(label, results))
        lines.append('')

        # jaccard-binned view for the forecast tasks
        if suffix == 'moved':
            lines += ['Delta by how much the present answer overlaps the future one. '
                      'Lower jaccard means less can be carried forward.', '',
                      '| jaccard | mean delta | cells positive | median n |', '|---|---|---|---|']
            for lo, hi in JACCARD_BINS:
                results = []
                for (mp, enc) in task_keys:
                    flags = load_flags(args.subset_dir, mp, task, 'moved')
                    if flags is None:
                        continue
                    keep = flags[(flags['jaccard'] > lo)
                                 & (flags['jaccard'] <= hi)]['idx'].tolist()
                    if len(keep) < 10:
                        continue
                    results.append(cell_delta(runs, task, mp, enc, keep))
                lines.append(summary_row(f'{lo:.2f}-{hi:.2f}', results))
            lines.append('')

    # ---- relay conditions -------------------------------------------------
    # Only present once build_relay_conditions.py has been run, which needs the
    # angle-augmented parse. Absent is normal, not an error.
    relay_tasks = sorted({t for (_, _, _, t, _) in runs
                          if any(Path(args.subset_dir).glob(f'*-{t}-relay-*.csv'))})
    for task in relay_tasks:
        task_keys = sorted({(m, e) for (a, m, e, t, s) in runs if t == task})
        srcs = set()
        lines += [f'## {task}, relay conditions', '',
                  'C1 the POV agent sees an enemy. C2 it does not, but a teammate does '
                  'and the agent',
                  'can see that teammate. C3 the same without visual contact. C4 nobody '
                  'on the team',
                  'sees any enemy. C2 versus C3 is the discriminator.', '',
                  '| condition | mean delta | cells positive | median n |', '|---|---|---|---|']
        for cond in ('C1', 'C2', 'C3', 'C4'):
            results = []
            for (mp, enc) in task_keys:
                flags, src = load_relay_flags(args.subset_dir, mp, task)
                if flags is None:
                    continue
                srcs.add(src)
                keep = flags[flags['condition'] == cond]['idx'].tolist()
                if len(keep) < 10:
                    continue
                results.append(cell_delta(runs, task, mp, enc, keep))
            lines.append(summary_row(cond, results))
        lines += ['', f'Source: {", ".join(sorted(srcs)) or "none"}', '']

    report_text = '\n'.join(lines)
    print('\n' + report_text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report_text, encoding='utf-8')
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
