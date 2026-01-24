#!/usr/bin/env python3
"""
Generate SLURM job scripts for downstream finetuned experiments.

Creates 9 jobs: 3 models × 3 UI mask settings (none, minimap_only, all)
Each job uses the corresponding stage 1 checkpoint from main_ui_cover.
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
TIME = "24:00:00"
MAIL_USER = "yunzhewa@usc.edu"
MAIL_TYPE = "all"
LOGS_SUBDIR = "logs"

# Experiment configuration
EXP_PREFIX = "downstream_finetuned"

# Models to sweep over (3 options)
MODELS = [
    "siglip2",
    "dinov2",
    "vjepa2",
]

# UI mask settings to sweep over (3 options)
UI_MASKS = [
    "none",
    "minimap_only",
    "all",
]

# Stage 1 checkpoint mapping: (model, ui_mask) -> checkpoint folder name
STAGE1_CHECKPOINTS = {
    ("dinov2", "all"): "main_ui_cover-dinov2-ui-all-260122-035704-my8c",
    ("dinov2", "minimap_only"): "main_ui_cover-dinov2-ui-minimap-260122-045334-demx",
    ("dinov2", "none"): "main_ui_cover-dinov2-ui-none-260122-051419-yq1p",
    ("siglip2", "all"): "main_ui_cover-siglip2-ui-all-260122-064933-md8t",
    ("siglip2", "minimap_only"): "main_ui_cover-siglip2-ui-minimap-260122-064933-1z0g",
    ("siglip2", "none"): "main_ui_cover-siglip2-ui-none-260122-071834-ct2l",
    ("vjepa2", "all"): "main_ui_cover-vjepa2-ui-all-260122-072237-nrz4",
    ("vjepa2", "minimap_only"): "main_ui_cover-vjepa2-ui-minimap-260122-072237-os7x",
    ("vjepa2", "none"): "main_ui_cover-vjepa2-ui-none-260122-101106-8h2z",
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

JOB_BODY = """module restore

cd {project_src}

uv run python train_all_downstream.py \\
  --model-type {model} \\
  --ui-mask {ui_mask} \\
  --stage1-checkpoint {stage1_checkpoint}
"""

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


def get_ui_mask_short_name(ui_mask: str) -> str:
    """Convert ui_mask to short form for naming"""
    mask_map = {
        "none": "ui-none",
        "minimap_only": "ui-minimap",
        "all": "ui-all",
    }
    return mask_map.get(ui_mask, ui_mask)


# ===== Main =====
def main():
    project_src = Path(PROJECT_SRC).resolve()
    jobs_root = project_src / "jobs" / EXP_PREFIX
    log_root = project_src / LOGS_SUBDIR / EXP_PREFIX

    jobs_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    mail_block = build_mail_block(MAIL_USER, MAIL_TYPE)

    all_jobs = []

    # Generate jobs for each combination (3 models × 3 ui_masks = 9 jobs)
    for model in MODELS:
        for ui_mask in UI_MASKS:
            ui_mask_short = get_ui_mask_short_name(ui_mask)
            run_name = f"{EXP_PREFIX}-{model}-{ui_mask_short}"

            # Get stage 1 checkpoint path
            checkpoint_folder = STAGE1_CHECKPOINTS[(model, ui_mask)]
            stage1_checkpoint = f"{checkpoint_folder}/checkpoint/last.ckpt"

            header = SCRIPT_HEADER.format(
                account=ACCOUNT,
                partition=PARTITION,
                cpus=CPUS,
                gpu_constraint=GPU_CONSTRAINT,
                gpu_count=GPU_COUNT,
                mem=MEM,
                time=TIME,
                job_name=run_name,
                log_dir=str(log_root),
                mail_block=mail_block,
            ).rstrip()

            body = JOB_BODY.format(
                project_src=str(project_src),
                model=model,
                ui_mask=ui_mask,
                stage1_checkpoint=stage1_checkpoint,
            ).rstrip()

            content = header + "\n\n" + body + "\n"
            job_path = jobs_root / f"{run_name}.job"
            job_path.write_text(content, encoding="utf-8")
            job_path.chmod(0o750)
            all_jobs.append(job_path)

    # Generate sbatch_all script
    sbatch_all_path = jobs_root / f"sbatch_all_{EXP_PREFIX}.sh"
    sbatch_all_path.write_text(SBATCH_ALL_HEADER, encoding="utf-8")
    sbatch_all_path.chmod(0o750)

    print(f"Generated {len(all_jobs)} job(s) in {jobs_root}")
    print(f"Logs will be written to {log_root}")
    print("\nBreakdown:")
    print(f"  - {len(MODELS)} models: {', '.join(MODELS)}")
    print(f"  - {len(UI_MASKS)} ui_mask settings: {', '.join(UI_MASKS)}")
    print(f"  - Total: {len(MODELS)} × {len(UI_MASKS)} = {len(all_jobs)} jobs")
    print("\nStage 1 checkpoints:")
    for (model, ui_mask), folder in sorted(STAGE1_CHECKPOINTS.items()):
        print(f"  - {model} + {ui_mask}: {folder}")
    print("\nTo submit all jobs to SLURM, run:")
    print(f"  bash {sbatch_all_path}")


if __name__ == "__main__":
    main()
