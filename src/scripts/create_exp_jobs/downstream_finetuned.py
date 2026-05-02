#!/usr/bin/env python3
"""
Generate SLURM job scripts for downstream experiments initialized from
main_contra_with_accu checkpoints.

The generated jobs mirror the current downstream_baseline.py runner shape:
one train_all_downstream.py invocation per (seed, map, model), with the
map/model-specific Stage 1 checkpoint passed through --stage1-checkpoint.
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
EXP_PREFIX = "downstream_finetuned_v2"

MODELS = [
    "siglip2",
    "dinov3",
    "clip",
    "vjepa2",
    "resnet50",
]

MAPS = [
    "dust2",
    "inferno",
    "mirage",
]

UI_MASK = "all"

SEEDS = [1, 2]

# Stage 1 checkpoint mapping: (model, map) -> checkpoint folder name
STAGE1_CHECKPOINTS = {
    ("clip", "dust2"): "main_contra_with_accu-clip-dust2-ui-all-260426-061227-kdhc",
    ("clip", "inferno"): "main_contra_with_accu-clip-inferno-ui-all-260426-062557-v5dk",
    ("clip", "mirage"): "main_contra_with_accu-clip-mirage-ui-all-260426-063549-w4kf",
    ("dinov3", "dust2"): "main_contra_with_accu-dinov3-dust2-ui-all-260426-064234-29fw",
    ("dinov3", "inferno"): "main_contra_with_accu-dinov3-inferno-ui-all-260426-065505-e4pl",
    ("dinov3", "mirage"): "main_contra_with_accu-dinov3-mirage-ui-all-260427-052436-jkwm",
    ("resnet50", "dust2"): "main_contra_with_accu-resnet50-dust2-ui-all-260427-071310-4l22",
    ("resnet50", "inferno"): "main_contra_with_accu-resnet50-inferno-ui-all-260427-073350-f0ra",
    ("resnet50", "mirage"): "main_contra_with_accu-resnet50-mirage-ui-all-260427-074405-vq44",
    ("siglip2", "dust2"): "main_contra_with_accu-siglip2-dust2-ui-all-260427-075806-ylp0",
    ("siglip2", "inferno"): "main_contra_with_accu-siglip2-inferno-ui-all-260427-080246-9h16",
    ("siglip2", "mirage"): "main_contra_with_accu-siglip2-mirage-ui-all-260427-080317-eixo",
    ("vjepa2", "dust2"): "main_contra_with_accu-vjepa2-dust2-ui-all-260427-080726-ujzu",
    ("vjepa2", "inferno"): "main_contra_with_accu-vjepa2-inferno-ui-all-260427-081410-64sg",
    ("vjepa2", "mirage"): "main_contra_with_accu-vjepa2-mirage-ui-all-260427-083324-13tq",
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
  --map {map_name} \\
  --model-type {model} \\
  --ui-mask {ui_mask} \\
  --stage1-checkpoint {stage1_checkpoint} \\
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


def get_stage1_checkpoint(model: str, map_name: str) -> str:
    """Return the relative last-checkpoint path consumed by train_all_downstream.py."""
    try:
        checkpoint_folder = STAGE1_CHECKPOINTS[(model, map_name)]
    except KeyError as exc:
        raise ValueError(f"No Stage 1 checkpoint configured for {model}/{map_name}") from exc
    return f"{checkpoint_folder}/checkpoint/last.ckpt"


def slurm_time_for_model(model: str) -> str:
    """V-JEPA runs get a longer wall time; all other finetuned runs use default."""
    if model.startswith("vjepa"):
        return SLURM_TIME["vjepa"]
    return SLURM_TIME["other"]


def validate_checkpoint_mapping() -> None:
    """Fail early if any requested (model, map) pair is missing from the mapping."""
    missing = [
        (model, map_name)
        for model in MODELS
        for map_name in MAPS
        if (model, map_name) not in STAGE1_CHECKPOINTS
    ]
    if missing:
        missing_text = ", ".join(f"{model}/{map_name}" for model, map_name in missing)
        raise ValueError(f"Missing Stage 1 checkpoint mapping(s): {missing_text}")


# ===== Main =====
def main():
    validate_checkpoint_mapping()

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
        for map_name in MAPS:
            for model in MODELS:
                map_short = get_map_short_name(map_name)
                ui_mask_short = get_ui_mask_short_name(UI_MASK)
                run_name = f"{EXP_PREFIX}-seed{seed}-{map_short}-{model}-{ui_mask_short}"
                stage1_checkpoint = get_stage1_checkpoint(model, map_name)
                model_overrides = [
                    "model.encoder.trainable=false",
                    f"meta.seed={seed}",
                ]

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
                    stage1_checkpoint=stage1_checkpoint,
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
    print(f"  - models: {', '.join(MODELS)}")
    print(f"  - maps: {', '.join(MAPS)}")
    print(f"  - ui_mask: {UI_MASK}")
    print(f"  - seeds: {SEEDS} (all map/model jobs per seed before next seed)")
    print(
        f"  - SLURM time: {SLURM_TIME['vjepa']} (vjepa*), {SLURM_TIME['other']} (other models)"
    )
    print("  - encoder: frozen Stage 1 checkpoint linear probe")
    print(f"  - Total: {len(all_jobs)} jobs")
    print("\nStage 1 checkpoints:")
    for model in MODELS:
        for map_name in MAPS:
            folder = STAGE1_CHECKPOINTS[(model, map_name)]
            print(f"  - {model}/{map_name}: {folder}")
    print("\nTo submit all jobs to SLURM, run:")
    print(f"  bash {sbatch_all_path}")


if __name__ == "__main__":
    main()
