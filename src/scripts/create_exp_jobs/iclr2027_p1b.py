#!/usr/bin/env python3
"""
P1b job scripts: the P1 arms (cecl / single / shuffle x seeds) trained with the
ORIGINAL cohort's batch regime (configs/train/contrastive_iclr2027_orig.yaml).
All recipe settings come from that config; the job only sets arm and seed.

    uv run python -m src.scripts.create_exp_jobs.iclr2027_p1b
"""
import argparse
from pathlib import Path

ROOT = Path("/home1/yunzhewa/projects/x-ego")
JOBS, LOGS = ROOT / "jobs" / "iclr2027_p1b", ROOT / "logs" / "iclr2027_p1b"
CONFIG = "configs/train/contrastive_iclr2027_orig.yaml"
ARMS = {"cecl": None, "single": None, "shuffle": "contrastive_matchheldout_v1_shuffle_s{seed}.csv"}

TEMPLATE = """#!/bin/bash
#SBATCH --account=ustun_1726
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint={constraint}
#SBATCH --gres=gpu:1
#SBATCH --mem=90G
#SBATCH --time=20:00:00
#SBATCH --job-name={name}
#SBATCH --output {logs}/{name}-%j.out
#SBATCH --error  {logs}/{name}-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yunzhewa@usc.edu

module purge
module restore
cd {root}
export TMPDIR=/scratch1/$USER/tmp
mkdir -p "$TMPDIR"

uv run python main.py --mode train --task contrastive \\
  --config {config} \\
  meta.seed={seed} \\
  meta.exp_name=iclr2027_p1b \\
  meta.run_name={run} \\
  data.pair_mode={arm} \\
  data.shuffle_manifest={manifest} \\
  data.num_workers=12
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--constraint", default="a40|a100|l40s")
    args = ap.parse_args()
    JOBS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    for arm, tpl in ARMS.items():
        for seed in args.seeds:
            run = f"mirage_siglip2_mean_orig_{arm}_s{seed}"
            name = f"iclr2027_p1b-{run}"
            text = TEMPLATE.format(constraint=args.constraint, name=name, logs=LOGS, root=ROOT, config=CONFIG,
                                   seed=seed, run=run, arm=arm, manifest=tpl.format(seed=seed) if tpl else "null")
            (JOBS / f"{name}.job").write_text(text)
            (JOBS / f"{name}.job").chmod(0o750)
            print(JOBS / f"{name}.job")


if __name__ == "__main__":
    main()
