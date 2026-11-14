#!/usr/bin/env python3
"""
Generate P2 job scripts.

  extract_{tier}_{i}_{partition}.job   one feature shard; same shard exists for debug and gpu
  probe_{tier}_{encoder}.job           CPU probe fitting for one encoder

    uv run python -m src.scripts.create_exp_jobs.iclr2027_p2 --core-shards 12 --rest-shards 9 --require 10
"""
import argparse
from pathlib import Path

ROOT = Path("/home1/yunzhewa/projects/x-ego")
JOBS = ROOT / "jobs" / "iclr2027_p2"
LOGS = ROOT / "logs" / "iclr2027_p2"

HEADER = """#!/bin/bash
#SBATCH --account=ustun_1726
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
{gpu_lines}#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --job-name={name}
#SBATCH --output {logs}/{name}-%j.out
#SBATCH --error  {logs}/{name}-%j.err
set -euo pipefail
module purge
module restore
cd {root}
export TMPDIR=/scratch1/$USER/tmp; mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1
"""


def write(name, body, partition, cpus, mem, time, gpu):
    gpu_lines = "#SBATCH --constraint=a40|a100|l40s\n#SBATCH --gres=gpu:1\n" if gpu else ""
    text = HEADER.format(partition=partition, cpus=cpus, gpu_lines=gpu_lines, mem=mem, time=time,
                         name=name, logs=LOGS, root=ROOT) + body + "\n"
    path = JOBS / f"{name}.job"
    path.write_text(text)
    path.chmod(0o750)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-shards", type=int, required=True)
    ap.add_argument("--rest-shards", type=int, required=True)
    ap.add_argument("--encoders", nargs="+", required=True,
                    help="exact encoder names every shard of this tag must contain")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--feat-tag", default="p1")
    ap.add_argument("--probe-out-tag", default="p2")
    ap.add_argument("--strip-prefix", default="", help="probe folder = encoder name without this prefix (P1b: orig_)")
    args = ap.parse_args()
    JOBS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    seeds = " ".join(map(str, args.seeds))
    enc_arg = " ".join(args.encoders)

    n = 0
    for tier, shards in (("core", args.core_shards), ("rest", args.rest_shards)):
        for i in range(shards):
            body = (f"uv run python -m src.scripts.iclr2027.extract_features --tier {tier} --shard {i} "
                    f"--num-shards {shards} --seeds {seeds} --encoders {enc_arg} --num-workers 10 --out-tag {args.feat_tag}")
            # debug's lone a40 node usually has only ~8 CPUs free; shards are GPU-bound at ~7 win/s
            write(f"p2x_{args.feat_tag}_{tier}_{i:03d}_debug", body.replace("--num-workers 10", "--num-workers 7"),
                  "debug", 8, "48G", "00:58:00", True)
            write(f"p2x_{args.feat_tag}_{tier}_{i:03d}_gpu", body, "gpu", 16, "64G", "01:00:00", True)
            n += 2

    encoders = args.encoders
    for tier in ("core", "rest"):
        for enc in encoders:
            out_name = enc[len(args.strip_prefix):] if args.strip_prefix and enc.startswith(args.strip_prefix) else enc
            body = (f"uv run python -m src.scripts.iclr2027.fit_probes --encoder {enc} --out-name {out_name} --tier {tier} "
                    f"--feat-tag {args.feat_tag} --out-tag {args.probe_out_tag}")
            write(f"p2p_{args.feat_tag}_{tier}_{enc}", body, "main", 16, "48G", "04:00:00", False)
            write(f"p2p_{args.feat_tag}_{tier}_{enc}_debug", body, "debug", 8, "32G", "00:58:00", False)
            n += 1
    print(f"wrote {n} job files to {JOBS}")


if __name__ == "__main__":
    main()
