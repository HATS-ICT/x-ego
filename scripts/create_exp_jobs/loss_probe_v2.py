#!/usr/bin/env python3
from pathlib import Path

# ===== Configuration =====
PROJECT_SRC = "/home1/yunzhewa/projects/x-ego"
ACCOUNT = "ustun_1726"
PARTITION = "gpu"
GPU_CONSTRAINT = "a40|a100"  # Use constraint to get either A40 or A100
GPU_COUNT = 1
CPUS = 15
MEM = "44G"
TIME = "03:00:00"
MAIL_USER = "yunzhewa@usc.edu"
MAIL_TYPE = "all"
LOGS_SUBDIR = "logs"

# Experiment configuration
EXP_PREFIX = "loss_probe_v2"
MODEL = "dinov2"
MAX_EPOCHS = 1

# Tasks to sweep over
TASKS = [
    "teammate_location_forecast",
    "enemy_location_forecast",
    "enemy_location_nowcast",
]

# grid-cls configurations (same as multi-label-cls from v1)
GRID_CLS_CONFIGS = [
    {
        "name": "bce_null",
        "loss": "bce",
        "class_weights": "null",
        "focal_alpha": None,
        "focal_gamma": None,
    },
    {
        "name": "bce_posweight",
        "loss": "bce",
        "class_weights": "pos_weight",
        "focal_alpha": None,
        "focal_gamma": None,
    },
    {
        "name": "focal_a025_g20",
        "loss": "focal",
        "class_weights": "null",
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
    },
    {
        "name": "focal_a050_g20",
        "loss": "focal",
        "class_weights": "null",
        "focal_alpha": 0.5,
        "focal_gamma": 2.0,
    },
]

# coord-gen configurations
COORD_GEN_CONFIGS = [
    {"name": "mse", "loss": "mse", "sinkhorn_blur": None, "sinkhorn_p": None},
    {"name": "sinkhorn_b005_p1", "loss": "sinkhorn", "sinkhorn_blur": 0.05, "sinkhorn_p": 1},
    {"name": "sinkhorn_b005_p2", "loss": "sinkhorn", "sinkhorn_blur": 0.05, "sinkhorn_p": 2},
    {"name": "sinkhorn_b010_p1", "loss": "sinkhorn", "sinkhorn_blur": 0.1, "sinkhorn_p": 1},
    {"name": "sinkhorn_b010_p2", "loss": "sinkhorn", "sinkhorn_blur": 0.1, "sinkhorn_p": 2},
]

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

JOB_BODY_GRID_CLS = """module restore

cd {project_src}

uv run python main.py --mode train --task {task} \\
  data.task_form=grid-cls \\
  model.encoder.video.model_type={model} \\
  model.loss_fn.grid-cls={loss} \\
  model.class_weights={class_weights}{focal_params} \\
  training.max_epochs={max_epochs} \\
  meta.exp_name={exp_name} \\
  meta.run_name={run_name}
"""

JOB_BODY_COORD_GEN = """module restore

cd {project_src}

uv run python main.py --mode train --task {task} \\
  data.task_form=coord-gen \\
  model.encoder.video.model_type={model} \\
  model.loss_fn.coord-gen={loss}{sinkhorn_params} \\
  training.max_epochs={max_epochs} \\
  meta.exp_name={exp_name} \\
  meta.run_name={run_name}
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

def build_focal_params(alpha, gamma):
    """Build focal loss parameter string"""
    if alpha is not None and gamma is not None:
        return f" \\\n  model.focal.alpha={alpha} \\\n  model.focal.gamma={gamma}"
    return ""

def build_sinkhorn_params(blur, p):
    """Build sinkhorn loss parameter string"""
    if blur is not None and p is not None:
        return f" \\\n  model.sinkhorn.blur={blur} \\\n  model.sinkhorn.p={p}"
    return ""

# ===== Main =====
def main():
    project_src = Path(PROJECT_SRC).resolve()
    jobs_root = project_src / "jobs" / EXP_PREFIX
    log_root = project_src / LOGS_SUBDIR / EXP_PREFIX
    
    jobs_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    mail_block = build_mail_block(MAIL_USER, MAIL_TYPE)

    all_jobs = []

    # Generate jobs for each task
    for task in TASKS:
        # grid-cls configurations
        for config in GRID_CLS_CONFIGS:
            run_name = f"{EXP_PREFIX}-{task}-gcls-{config['name']}"
            
            header = SCRIPT_HEADER.format(
                account=ACCOUNT, partition=PARTITION, cpus=CPUS,
                gpu_constraint=GPU_CONSTRAINT, gpu_count=GPU_COUNT,
                mem=MEM, time=TIME,
                job_name=run_name, log_dir=str(log_root),
                mail_block=mail_block,
            ).rstrip()

            focal_params = build_focal_params(config['focal_alpha'], config['focal_gamma'])
            
            body = JOB_BODY_GRID_CLS.format(
                project_src=str(project_src),
                task=task,
                model=MODEL,
                loss=config['loss'],
                class_weights=config['class_weights'],
                focal_params=focal_params,
                max_epochs=MAX_EPOCHS,
                exp_name=EXP_PREFIX,
                run_name=run_name,
            ).rstrip()

            content = header + "\n\n" + body + "\n"
            job_path = jobs_root / f"{run_name}.job"
            job_path.write_text(content, encoding="utf-8")
            job_path.chmod(0o750)
            all_jobs.append(job_path)

        # coord-gen configurations
        for config in COORD_GEN_CONFIGS:
            run_name = f"{EXP_PREFIX}-{task}-cgen-{config['name']}"
            
            header = SCRIPT_HEADER.format(
                account=ACCOUNT, partition=PARTITION, cpus=CPUS,
                gpu_constraint=GPU_CONSTRAINT, gpu_count=GPU_COUNT,
                mem=MEM, time=TIME,
                job_name=run_name, log_dir=str(log_root),
                mail_block=mail_block,
            ).rstrip()

            sinkhorn_params = build_sinkhorn_params(config['sinkhorn_blur'], config['sinkhorn_p'])
            
            body = JOB_BODY_COORD_GEN.format(
                project_src=str(project_src),
                task=task,
                model=MODEL,
                loss=config['loss'],
                sinkhorn_params=sinkhorn_params,
                max_epochs=MAX_EPOCHS,
                exp_name=EXP_PREFIX,
                run_name=run_name,
            ).rstrip()

            content = header + "\n\n" + body + "\n"
            job_path = jobs_root / f"{run_name}.job"
            job_path.write_text(content, encoding="utf-8")
            job_path.chmod(0o750)
            all_jobs.append(job_path)

    # Generate sbatch_all script
    sbatch_all_path = jobs_root / "sbatch_all_loss_probe_v2.sh"
    sbatch_all_path.write_text(SBATCH_ALL_HEADER, encoding="utf-8")
    sbatch_all_path.chmod(0o750)

    print(f"Generated {len(all_jobs)} job(s) in {jobs_root}")
    print(f"Logs will be written to {log_root}")
    print(f"\nBreakdown:")
    print(f"  - {len(TASKS)} tasks")
    print(f"  - {len(GRID_CLS_CONFIGS)} grid-cls configs per task")
    print(f"  - {len(COORD_GEN_CONFIGS)} coord-gen configs per task")
    print(f"  - Total: {len(TASKS)} × ({len(GRID_CLS_CONFIGS)} + {len(COORD_GEN_CONFIGS)}) = {len(all_jobs)} jobs")
    print(f"\nTo submit all jobs, run:")
    print(f"  bash {sbatch_all_path}")

if __name__ == "__main__":
    main()


