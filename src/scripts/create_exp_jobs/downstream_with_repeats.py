#!/usr/bin/env python3
"""
Generate SLURM job scripts for downstream experiments with repeats.

Combines baseline and finetuned experiments with multiple seeds (1, 2, 3).

Job structure:
- siglip2 & dinov2: 1 job per model runs all 3 seeds sequentially
  - 2 models × 2 types (baseline + finetuned) = 4 jobs
- vjepa2: 1 job per seed
  - 3 seeds × 2 types (baseline + finetuned) = 6 jobs

Total: 10 jobs
"""
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# ===== Configuration =====
PROJECT_SRC = os.getenv("SRC_BASE_PATH")
ACCOUNT = "ustun_1726"
PARTITION = "gpu"
GPU_CONSTRAINT = "a40|a100"
GPU_COUNT = 1
CPUS = 26
MEM = "60G"
TIME = "36:00:00"
MAIL_USER = "yunzhewa@usc.edu"
MAIL_TYPE = "all"
LOGS_SUBDIR = "logs"

# Experiment configuration
EXP_PREFIX = "downstream_with_repeats"

# Models
MODELS_FAST = ["siglip2", "dinov2"]  # All seeds in 1 job
MODELS_SLOW = ["vjepa2"]  # 1 seed per job

# UI mask setting (fixed to "all")
UI_MASK = "all"

# Seeds for repeats
SEEDS = [1, 2, 3]

# Stage 1 checkpoint mapping: model -> checkpoint folder name (for ui_mask=all)
STAGE1_CHECKPOINTS = {
    "dinov2": "main_ui_cover-dinov2-ui-all-260122-035704-my8c",
    "siglip2": "main_ui_cover-siglip2-ui-all-260122-064933-md8t",
    "vjepa2": "main_ui_cover-vjepa2-ui-all-260122-072237-nrz4",
}

# ===== Templates =====
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
{mail_block}
"""

JOB_BODY_BASELINE = """module restore

cd {project_src}

{commands}"""

TRAIN_CMD_BASELINE = """uv run python train_all_downstream.py \\
  --model-type {model} \\
  --ui-mask {ui_mask} \\
  --extra-overrides meta.seed={seed} data.num_workers=8"""

TRAIN_CMD_FINETUNED = """uv run python train_all_downstream.py \\
  --model-type {model} \\
  --ui-mask {ui_mask} \\
  --stage1-checkpoint {stage1_checkpoint} \\
  --extra-overrides meta.seed={seed} data.num_workers=8"""

SBATCH_ALL_HEADER = """#!/bin/bash
set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
shopt -s nullglob

jobs=("$dir"/*.job)
if (( ${#jobs[@]} == 0 )); then
  echo "No .job files found in $dir"
  exit 1
fi

for f in "${jobs[@]}"; do
  echo "sbatch $f"
  sbatch "$f"
done
"""


# ===== Helpers =====
def build_mail_block(mail_user: str, mail_type: str) -> str:
    if mail_type.upper() == "NONE":
        return ""
    return f"#SBATCH --mail-type={mail_type.upper()}\n#SBATCH --mail-user={mail_user}"


def create_job_file(
    jobs_root: Path,
    log_root: Path,
    job_name: str,
    commands: list[str],
    mail_block: str,
    project_src: Path,
) -> Path:
    """Create a job file with the given commands."""
    header = SCRIPT_HEADER.format(
        account=ACCOUNT,
        partition=PARTITION,
        cpus=CPUS,
        gpu_constraint=GPU_CONSTRAINT,
        gpu_count=GPU_COUNT,
        mem=MEM,
        time=TIME,
        job_name=job_name,
        log_dir=str(log_root),
        mail_block=mail_block,
    ).rstrip()

    body = JOB_BODY_BASELINE.format(
        project_src=str(project_src),
        commands="\n\n".join(commands),
    ).rstrip()

    content = header + "\n\n" + body + "\n"
    job_path = jobs_root / f"{job_name}.job"
    job_path.write_text(content, encoding="utf-8")
    job_path.chmod(0o750)
    return job_path


# ===== Main =====
def main():
    project_src = Path(PROJECT_SRC).resolve()
    jobs_root = project_src / "jobs" / EXP_PREFIX
    log_root = project_src / LOGS_SUBDIR / EXP_PREFIX

    jobs_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    mail_block = build_mail_block(MAIL_USER, MAIL_TYPE)

    all_jobs = []

    # ===== Slow model (vjepa2): 1 job per seed (submitted first) =====
    for model in MODELS_SLOW:
        stage1_checkpoint = f"{STAGE1_CHECKPOINTS[model]}/checkpoint/last.ckpt"

        for seed in SEEDS:
            # Baseline job for this seed
            job_name = f"{EXP_PREFIX}-{model}-baseline-seed{seed}"
            commands = [
                TRAIN_CMD_BASELINE.format(
                    model=model,
                    ui_mask=UI_MASK,
                    seed=seed,
                )
            ]
            job_path = create_job_file(
                jobs_root, log_root, job_name, commands, mail_block, project_src
            )
            all_jobs.append(job_path)

            # Finetuned job for this seed
            job_name = f"{EXP_PREFIX}-{model}-finetuned-seed{seed}"
            commands = [
                TRAIN_CMD_FINETUNED.format(
                    model=model,
                    ui_mask=UI_MASK,
                    stage1_checkpoint=stage1_checkpoint,
                    seed=seed,
                )
            ]
            job_path = create_job_file(
                jobs_root, log_root, job_name, commands, mail_block, project_src
            )
            all_jobs.append(job_path)

    # ===== Fast models (siglip2, dinov2): 1 job per model, all seeds =====
    for model in MODELS_FAST:
        stage1_checkpoint = f"{STAGE1_CHECKPOINTS[model]}/checkpoint/last.ckpt"

        # Baseline job: all 3 seeds
        job_name = f"{EXP_PREFIX}-{model}-baseline"
        commands = [
            TRAIN_CMD_BASELINE.format(
                model=model,
                ui_mask=UI_MASK,
                seed=seed,
            )
            for seed in SEEDS
        ]
        job_path = create_job_file(
            jobs_root, log_root, job_name, commands, mail_block, project_src
        )
        all_jobs.append(job_path)

        # Finetuned job: all 3 seeds
        job_name = f"{EXP_PREFIX}-{model}-finetuned"
        commands = [
            TRAIN_CMD_FINETUNED.format(
                model=model,
                ui_mask=UI_MASK,
                stage1_checkpoint=stage1_checkpoint,
                seed=seed,
            )
            for seed in SEEDS
        ]
        job_path = create_job_file(
            jobs_root, log_root, job_name, commands, mail_block, project_src
        )
        all_jobs.append(job_path)

    # Generate sbatch_all script
    sbatch_all_path = jobs_root / f"sbatch_all_{EXP_PREFIX}.sh"
    sbatch_all_path.write_text(SBATCH_ALL_HEADER, encoding="utf-8")
    sbatch_all_path.chmod(0o750)

    print(f"Generated {len(all_jobs)} job(s) in {jobs_root}")
    print(f"Logs will be written to {log_root}")
    print("\nBreakdown:")
    print(f"  - Fast models ({', '.join(MODELS_FAST)}): 2 types × {len(MODELS_FAST)} models = {2 * len(MODELS_FAST)} jobs (all {len(SEEDS)} seeds per job)")
    print(f"  - Slow model ({', '.join(MODELS_SLOW)}): 2 types × {len(SEEDS)} seeds = {2 * len(SEEDS)} jobs (1 seed per job)")
    print(f"  - UI mask: {UI_MASK}")
    print(f"  - Seeds: {SEEDS}")
    print(f"  - Total: {len(all_jobs)} jobs")
    print("\nStage 1 checkpoints:")
    for model, folder in sorted(STAGE1_CHECKPOINTS.items()):
        print(f"  - {model}: {folder}")
    print("\nTo submit all jobs to SLURM, run:")
    print(f"  bash {sbatch_all_path}")


if __name__ == "__main__":
    main()
