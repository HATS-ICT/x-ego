#!/usr/bin/env python3
"""
Generate SLURM job scripts for downstream baseline experiments.

Creates downstream baseline jobs with a map-specific model sweep and
per-seed repeats (SEEDS). Jobs are emitted and named so all (map, model)
runs for one seed complete before the next seed; sbatch_all submits in
that same order (sorted .job paths).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ===== Configuration =====
PROJECT_SRC = os.getenv("SRC_BASE_PATH")
ACCOUNT = "ustun_1726"
PARTITION = "gpu"
GPU_CONSTRAINT = "a40|a100"
GPU_COUNT = 1
CPUS = 26
MEM = "80G"
SLURM_TIME = {
    "vjepa": "48:00:00",
    "other": "24:00:00",
}
MAIL_USER = "yunzhewa@usc.edu"
MAIL_TYPE = "all"
LOGS_SUBDIR = "logs"

# Experiment configuration
EXP_PREFIX = "downstream_baseline_v2"

# Model sweep by map. Mirage gets the full model sweep; dust2 and inferno
# only run the main pretrained baseline and the from-scratch ResNet baseline.
MODELS_BY_MAP = {
    "dust2": [
        "siglip2",
        "resnet50",
    ],
    "inferno": [
        "siglip2",
        "resnet50",
    ],
    "mirage": [
        "siglip2",
        "dinov3",
        "clip",
        "vjepa2",
        "resnet50",
    ],
}

UI_MASK = "all"

SEEDS = [1, 2]

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
  --map {map_name} \\
  --model-type {model} \\
  --ui-mask {ui_mask} \\
  --extra-overrides {extra_overrides}
"""

SBATCH_ALL_HEADER = """#!/bin/bash
set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
shopt -s nullglob

mapfile -t jobs < <(printf '%s\n' "$dir"/*.job | LC_ALL=C sort)
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
    """Convert ui_mask to short form for naming."""
    mask_map = {
        "none": "ui-none",
        "minimap_only": "ui-minimap",
        "all": "ui-all",
    }
    return mask_map.get(ui_mask, ui_mask)


def get_map_short_name(map_name: str) -> str:
    """Convert map name to short form for naming."""
    return map_name.removeprefix("de_")


def get_model_overrides(model: str) -> list[str]:
    """Return config overrides for the requested model baseline."""
    if model == "resnet50":
        return ["model.encoder.trainable=true"]
    return ["model.encoder.trainable=false"]


def slurm_time_for_model(model: str) -> str:
    """V-JEPA runs get a longer wall time; all other baselines use the default."""
    if model.startswith("vjepa"):
        return SLURM_TIME["vjepa"]
    return SLURM_TIME["other"]


# ===== Main =====
def main():
    project_src = Path(PROJECT_SRC).resolve()
    jobs_root = project_src / "jobs" / EXP_PREFIX
    log_root = project_src / LOGS_SUBDIR / EXP_PREFIX

    jobs_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    for old_job in jobs_root.glob("*.job"):
        old_job.unlink()

    mail_block = build_mail_block(MAIL_USER, MAIL_TYPE)

    all_jobs = []

    # Generate jobs: all (map, model) for seed 1, then all for seed 2, ...
    for seed in SEEDS:
        for map_name, models in MODELS_BY_MAP.items():
            for model in models:
                map_short = get_map_short_name(map_name)
                ui_mask_short = get_ui_mask_short_name(UI_MASK)
                run_name = f"{EXP_PREFIX}-seed{seed}-{map_short}-{model}-{ui_mask_short}"
                model_overrides = get_model_overrides(model) + [f"meta.seed={seed}"]

                header = SCRIPT_HEADER.format(
                    account=ACCOUNT,
                    partition=PARTITION,
                    cpus=CPUS,
                    gpu_constraint=GPU_CONSTRAINT,
                    gpu_count=GPU_COUNT,
                    mem=MEM,
                    time=slurm_time_for_model(model),
                    job_name=run_name,
                    log_dir=str(log_root),
                    mail_block=mail_block,
                ).rstrip()

                body = JOB_BODY.format(
                    project_src=str(project_src),
                    map_name=map_name,
                    model=model,
                    ui_mask=UI_MASK,
                    extra_overrides=" ".join(model_overrides),
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
    for map_name, models in MODELS_BY_MAP.items():
        print(f"  - {map_name}: {', '.join(models)}")
    print(f"  - ui_mask: {UI_MASK}")
    print(f"  - seeds: {SEEDS} (all map/model jobs per seed before next seed)")
    print(
        f"  - SLURM time: {SLURM_TIME['vjepa']} (vjepa*), {SLURM_TIME['other']} (other models)"
    )
    print("  - resnet50: train encoder from scratch")
    print("  - other models: frozen encoder linear probe")
    print(f"  - Total: {len(all_jobs)} jobs")
    print("\nTo submit all jobs to SLURM, run:")
    print(f"  bash {sbatch_all_path}")


if __name__ == "__main__":
    main()
