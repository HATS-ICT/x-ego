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
EXP_PREFIX = "loss_probe"
MODEL = "dinov2"
MAX_EPOCHS = 1

# Tasks to sweep over
TASKS = [
    "teammate_location_forecast",
    "enemy_location_forecast",
    "enemy_location_nowcast",
]

# multi-label-cls configurations
MULTI_LABEL_CLS_CONFIGS = [
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

# density-cls configurations
DENSITY_CLS_CONFIGS = [
    {"name": "mse_sigma10", "loss": "mse", "gaussian_sigma": 1.0},
    {"name": "mse_sigma20", "loss": "mse", "gaussian_sigma": 2.0},
    {"name": "mse_sigma30", "loss": "mse", "gaussian_sigma": 3.0},
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

JOB_BODY_MULTI_LABEL = """module restore

cd {project_src}

uv run python main.py --mode train --task {task} \\
  data.task_form=multi-label-cls \\
  model.encoder.video.model_type={model} \\
  model.loss_fn.multi-label-cls={loss} \\
  model.class_weights={class_weights}{focal_params} \\
  training.max_epochs={max_epochs} \\
  meta.exp_name={exp_name} \\
  meta.run_name={run_name}
"""

JOB_BODY_DENSITY = """module restore

cd {project_src}

uv run python main.py --mode train --task {task} \\
  data.task_form=density-cls \\
  data.gaussian_sigma={gaussian_sigma} \\
  model.encoder.video.model_type={model} \\
  model.loss_fn.density-cls={loss} \\
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
        # multi-label-cls configurations
        for config in MULTI_LABEL_CLS_CONFIGS:
            run_name = f"{EXP_PREFIX}-{task}-mlcls-{config['name']}"
            
            header = SCRIPT_HEADER.format(
                account=ACCOUNT, partition=PARTITION, cpus=CPUS,
                gpu_constraint=GPU_CONSTRAINT, gpu_count=GPU_COUNT,
                mem=MEM, time=TIME,
                job_name=run_name, log_dir=str(log_root),
                mail_block=mail_block,
            ).rstrip()

            focal_params = build_focal_params(config['focal_alpha'], config['focal_gamma'])
            
            body = JOB_BODY_MULTI_LABEL.format(
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

        # density-cls configurations
        for config in DENSITY_CLS_CONFIGS:
            run_name = f"{EXP_PREFIX}-{task}-dcls-{config['name']}"
            
            header = SCRIPT_HEADER.format(
                account=ACCOUNT, partition=PARTITION, cpus=CPUS,
                gpu_constraint=GPU_CONSTRAINT, gpu_count=GPU_COUNT,
                mem=MEM, time=TIME,
                job_name=run_name, log_dir=str(log_root),
                mail_block=mail_block,
            ).rstrip()

            body = JOB_BODY_DENSITY.format(
                project_src=str(project_src),
                task=task,
                model=MODEL,
                loss=config['loss'],
                gaussian_sigma=config['gaussian_sigma'],
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
    sbatch_all_path = jobs_root / "sbatch_all_loss_probe.sh"
    sbatch_all_path.write_text(SBATCH_ALL_HEADER, encoding="utf-8")
    sbatch_all_path.chmod(0o750)

    print(f"Generated {len(all_jobs)} job(s) in {jobs_root}")
    print(f"Logs will be written to {log_root}")
    print(f"\nBreakdown:")
    print(f"  - {len(TASKS)} tasks")
    print(f"  - {len(MULTI_LABEL_CLS_CONFIGS)} multi-label-cls configs per task")
    print(f"  - {len(DENSITY_CLS_CONFIGS)} density-cls configs per task")
    print(f"  - Total: {len(TASKS)} × ({len(MULTI_LABEL_CLS_CONFIGS)} + {len(DENSITY_CLS_CONFIGS)}) = {len(all_jobs)} jobs")
    print(f"\nTo submit all jobs, run:")
    print(f"  bash {sbatch_all_path}")

if __name__ == "__main__":
    main()

