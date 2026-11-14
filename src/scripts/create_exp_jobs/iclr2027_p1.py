#!/usr/bin/env python3
"""
Generate the ICLR-2027 P1 job scripts: 3 arms x 3 seeds = 9 adaptation runs,
plus one debug-partition pilot job that is the P0 completion gate.

Arm F (frozen pretrained) needs no adaptation run and is therefore not a job
here; it enters the comparison at the probe stage as the unadapted encoder.

    uv run python -m src.scripts.create_exp_jobs.iclr2027_p1
    # extra seeds only, deprioritized, without regenerating seeds 1-3:
    uv run python -m src.scripts.create_exp_jobs.iclr2027_p1 --seeds 4 5 --nice 100 \
        --constraint "a40|a100|l40s" --no-pilot
"""
from pathlib import Path
import argparse
import os

from dotenv import load_dotenv

load_dotenv()

PROJECT_SRC = os.getenv("SRC_BASE_PATH")
ACCOUNT = "ustun_1726"
PARTITION = "gpu"
GPU_CONSTRAINT = "a40|a100"
GPU_COUNT = 1
CPUS = 26
MEM = "90G"
TIME = "48:00:00"
MAIL_USER = "yunzhewa@usc.edu"
MAIL_TYPE = "all"
LOGS_SUBDIR = "logs"

EXP_PREFIX = "iclr2027_p1"
CONFIG = "configs/train/contrastive_iclr2027.yaml"
COHORT = "matchheldout_v1"
MAP = "mirage"
MODEL = "siglip2"

# Arm id -> (pair_mode, shuffle manifest template or None)
ARMS = {
    "cecl": ("cecl", None),
    "single": ("single", None),
    "shuffle": ("shuffle", f"contrastive_{COHORT}_shuffle_s{{seed}}.csv"),
}
SEEDS = [1, 2, 3]

# Matched across arms: 64 x 16 = 1024-video effective contrastive batch.
BATCH_SIZE = 64
ACCUM = 16
MAX_EPOCHS = 40

SCRIPT_HEADER = """#!/bin/bash
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --constraint={gpu_constraint}
#SBATCH --gres=gpu:{gpu_count}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --job-name={job_name}
#SBATCH --output {log_dir}/{job_name}-%j.out
#SBATCH --error  {log_dir}/{job_name}-%j.err
{nice_line}{mail_block}
"""

JOB_BODY = """module purge
module restore

cd {project_src}

export TMPDIR=/scratch1/$USER/tmp
mkdir -p "$TMPDIR"

uv run python main.py --mode train --task contrastive \\
  --config {config} \\
  meta.seed={seed} \\
  meta.exp_name={exp_name} \\
  meta.run_name={run_name} \\
  data.map={map_name} \\
  data.pair_mode={pair_mode} \\
  data.shuffle_manifest={shuffle_manifest} \\
  data.batch_size={batch_size} \\
  training.contrastive_accumulate_batches={accum} \\
  training.max_epochs={max_epochs} \\
  training.accumulate_grad_batches=1 \\
  training.torch_compile=false
"""

PILOT_BODY = """module purge
module restore

cd {project_src}

export TMPDIR=/scratch1/$USER/tmp
mkdir -p "$TMPDIR"

# P0 completion gate: pair masks, trainable parameters, frame aggregation,
# arm exposure budgets, and a measured steps/second for the GPU-hour estimate.
uv run python -m src.scripts.iclr2027.pilot_check_arms \\
  --arms cecl single shuffle \\
  --seed 1 \\
  --batches 8 \\
  --timed-batches 40
"""

SBATCH_ALL = """#!/bin/bash
# Submit the 9 P1 adaptation runs.
# Run the pilot job first and read artifact/iclr2027_pilot_s1.json before this.
set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
shopt -s nullglob

jobs=("$dir"/iclr2027_p1-*.job)
if (( ${#jobs[@]} == 0 )); then
  echo "No .job files found in $dir"
  exit 1
fi

for f in "${jobs[@]}"; do
  echo "sbatch $f"
  sbatch "$f"
done
"""


def build_mail_block(mail_user: str, mail_type: str) -> str:
    if mail_type.upper() == "NONE":
        return ""
    return f"#SBATCH --mail-type={mail_type.upper()}\n#SBATCH --mail-user={mail_user}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ICLR-2027 P1 job scripts")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--nice", type=int, default=0,
                        help="SLURM --nice for these jobs (lowers their priority vs. our other jobs)")
    parser.add_argument("--constraint", default=GPU_CONSTRAINT)
    parser.add_argument("--no-pilot", action="store_true",
                        help="do not (re)write the pilot job or the sbatch_all script")
    args = parser.parse_args()
    nice_line = f"#SBATCH --nice={args.nice}\n" if args.nice else ""

    project_src = Path(PROJECT_SRC).resolve()
    jobs_root = project_src / "jobs" / EXP_PREFIX
    log_root = project_src / LOGS_SUBDIR / EXP_PREFIX
    jobs_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    mail_block = build_mail_block(MAIL_USER, MAIL_TYPE)
    written = []

    for arm, (pair_mode, manifest_tpl) in ARMS.items():
        for seed in args.seeds:
            run_name = f"{MAP}_{MODEL}_mean_{arm}_s{seed}"
            job_name = f"{EXP_PREFIX}-{run_name}"
            manifest = manifest_tpl.format(seed=seed) if manifest_tpl else "null"

            header = SCRIPT_HEADER.format(
                account=ACCOUNT, partition=PARTITION, cpus=CPUS,
                gpu_constraint=args.constraint, gpu_count=GPU_COUNT,
                mem=MEM, time=TIME, job_name=job_name,
                log_dir=str(log_root), nice_line=nice_line, mail_block=mail_block,
            ).rstrip()

            body = JOB_BODY.format(
                project_src=str(project_src), config=CONFIG, seed=seed,
                exp_name=EXP_PREFIX, run_name=run_name, map_name=MAP,
                pair_mode=pair_mode, shuffle_manifest=manifest,
                batch_size=BATCH_SIZE, accum=ACCUM, max_epochs=MAX_EPOCHS,
            ).rstrip()

            path = jobs_root / f"{job_name}.job"
            path.write_text(header + "\n\n" + body + "\n", encoding="utf-8")
            path.chmod(0o750)
            written.append(path)

    if args.no_pilot:
        print(f"Generated {len(written)} adaptation job(s) in {jobs_root}")
        for path in written:
            print(f"  {path.name}")
        print("Submit them individually with sbatch (sbatch_all would also resubmit seeds 1-3).")
        return

    # Pilot gate job on the debug partition (1 h limit, negligible queue wait).
    pilot_name = f"{EXP_PREFIX}_pilot"
    pilot_header = SCRIPT_HEADER.format(
        account=ACCOUNT, partition="debug", cpus=CPUS,
        gpu_constraint=GPU_CONSTRAINT, gpu_count=1,
        mem=MEM, time="00:50:00", job_name=pilot_name,
        log_dir=str(log_root), nice_line="", mail_block=mail_block,
    ).rstrip()
    pilot_path = jobs_root / f"{pilot_name}.job"
    pilot_path.write_text(
        pilot_header + "\n\n" + PILOT_BODY.format(project_src=str(project_src)).rstrip() + "\n",
        encoding="utf-8",
    )
    pilot_path.chmod(0o750)

    sbatch_all = jobs_root / f"sbatch_all_{EXP_PREFIX}.sh"
    sbatch_all.write_text(SBATCH_ALL, encoding="utf-8")
    sbatch_all.chmod(0o750)

    print(f"Generated {len(written)} adaptation job(s) + 1 pilot job in {jobs_root}")
    for path in [pilot_path] + written:
        print(f"  {path.name}")
    print(f"\nLogs: {log_root}")
    print(f"Pilot first:  sbatch {pilot_path}")
    print(f"Then the 9:   bash {sbatch_all}")


if __name__ == "__main__":
    main()
