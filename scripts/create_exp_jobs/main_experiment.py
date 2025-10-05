#!/usr/bin/env python3
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# ===== Configuration =====
PROJECT_SRC = os.getenv("SRC_BASE_PATH")
ACCOUNT = "ustun_1726"
PARTITION = "gpu"
GPU_CONSTRAINT = "a40|a100"  # Use constraint to get either A40 or A100
GPU_COUNT = 1
CPUS = 20
MEM = "64G"
TIME = "06:00:00"
MAIL_USER = "yunzhewa@usc.edu"
MAIL_TYPE = "all"
LOGS_SUBDIR = "logs"

# Experiment configuration
EXP_PREFIX = "main_exp_v2"
TASK_FORM = "multi-label-cls"

# Tasks to sweep over
TASKS = [
    "enemy_location_nowcast",
    "teammate_location_nowcast",
]

# Models to sweep over
MODELS = [
    "dinov2",
    "siglip",
    "vivit",
    "videomae",
]

# Number of POV agents to sweep over
NUM_POV_AGENTS = [1, 2, 3, 4, 5]

# Contrastive settings to sweep over
CONTRASTIVE_SETTINGS = [
    {"enable": False, "suffix": "no-contra"},
    {"enable": True, "suffix": "yes-contra"},
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

JOB_BODY_NO_CONTRA = """module restore

cd {project_src}

uv run python main.py --mode train --task {task} \\
  data.num_pov_agents={num_pov_agents} \\
  data.task_form={task_form} \\
  model.encoder.video.model_type={model} \\
  model.contrastive.enable=false \\
  meta.exp_name={exp_name} \\
  meta.run_name={run_name}
"""

JOB_BODY_YES_CONTRA = """module restore

cd {project_src}

uv run python main.py --mode train --task {task} \\
  data.num_pov_agents={num_pov_agents} \\
  data.task_form={task_form} \\
  model.encoder.video.model_type={model} \\
  model.contrastive.enable=true \\
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

SEQUENTIAL_RUN_HEADER = """#!/bin/bash
# Sequential execution of all experiments on a single machine
# Run this script to execute all experiments one after another
set -euo pipefail

cd {project_src}

"""

# ===== Helpers =====
def build_mail_block(mail_user: str, mail_type: str) -> str:
    if mail_type.upper() == "NONE":
        return ""
    return f"#SBATCH --mail-type={mail_type.upper()}\n#SBATCH --mail-user={mail_user}"

def get_task_short_name(task: str) -> str:
    """Convert task name to short form for naming"""
    task_map = {
        "teammate_location_nowcast": "tm-nowcast",
        "enemy_location_nowcast": "en-nowcast",
    }
    return task_map.get(task, task)

# ===== Main =====
def main():
    project_src = Path(PROJECT_SRC).resolve()
    base_jobs_root = project_src / "jobs" / EXP_PREFIX
    base_log_root = project_src / LOGS_SUBDIR / EXP_PREFIX
    
    base_jobs_root.mkdir(parents=True, exist_ok=True)
    base_log_root.mkdir(parents=True, exist_ok=True)

    mail_block = build_mail_block(MAIL_USER, MAIL_TYPE)

    total_jobs = 0
    
    # Generate jobs organized by model and task
    for model in MODELS:
        for task in TASKS:
            task_short = get_task_short_name(task)
            folder_name = f"{model}_{task_short}"
            
            # Create folder-specific directories
            jobs_root = base_jobs_root / folder_name
            log_root = base_log_root / folder_name
            
            jobs_root.mkdir(parents=True, exist_ok=True)
            log_root.mkdir(parents=True, exist_ok=True)
            
            all_jobs = []
            all_commands = []
            
            # Generate jobs for each POV agent and contrastive setting
            for num_pov in NUM_POV_AGENTS:
                for contra_setting in CONTRASTIVE_SETTINGS:
                    contra_enable = contra_setting["enable"]
                    contra_suffix = contra_setting["suffix"]
                    
                    run_name = f"{model}-{task_short}-{contra_suffix}-pov{num_pov}"
                    exp_name = f"{EXP_PREFIX}/{folder_name}"
                    
                    header = SCRIPT_HEADER.format(
                        account=ACCOUNT, partition=PARTITION, cpus=CPUS,
                        gpu_constraint=GPU_CONSTRAINT, gpu_count=GPU_COUNT,
                        mem=MEM, time=TIME,
                        job_name=run_name, log_dir=str(log_root),
                        mail_block=mail_block,
                    ).rstrip()
                    
                    # Choose appropriate job body template
                    if contra_enable:
                        body_template = JOB_BODY_YES_CONTRA
                    else:
                        body_template = JOB_BODY_NO_CONTRA
                    
                    body = body_template.format(
                        project_src=str(project_src),
                        task=task,
                        num_pov_agents=num_pov,
                        task_form=TASK_FORM,
                        model=model,
                        exp_name=exp_name,
                        run_name=run_name,
                    ).rstrip()

                    content = header + "\n\n" + body + "\n"
                    job_path = jobs_root / f"{run_name}.job"
                    job_path.write_text(content, encoding="utf-8")
                    job_path.chmod(0o750)
                    all_jobs.append(job_path)
                    
                    # Extract the command for sequential run script
                    if contra_enable:
                        command = f"""uv run python main.py --mode train --task {task} \\
  data.num_pov_agents={num_pov} \\
  data.task_form={TASK_FORM} \\
  model.encoder.video.model_type={model} \\
  model.contrastive.enable=true \\
  meta.exp_name={exp_name} \\
  meta.run_name={run_name}"""
                    else:
                        command = f"""uv run python main.py --mode train --task {task} \\
  data.num_pov_agents={num_pov} \\
  data.task_form={TASK_FORM} \\
  model.encoder.video.model_type={model} \\
  model.contrastive.enable=false \\
  meta.exp_name={exp_name} \\
  meta.run_name={run_name}"""
                    all_commands.append((run_name, command))
            
            # Generate sbatch_all script for this folder
            sbatch_all_path = jobs_root / f"sbatch_all_{folder_name}.sh"
            sbatch_all_path.write_text(SBATCH_ALL_HEADER, encoding="utf-8")
            sbatch_all_path.chmod(0o750)
            
            # Generate sequential run script for this folder
            sequential_run_path = jobs_root / f"run_all_{folder_name}_sequential.sh"
            sequential_content = SEQUENTIAL_RUN_HEADER.format(project_src=str(project_src))
            for i, (run_name, command) in enumerate(all_commands, 1):
                sequential_content += f"\n# Job {i}/{len(all_commands)}: {run_name}\n"
                sequential_content += f"echo '=== Running {run_name} ({i}/{len(all_commands)}) ==='\n"
                sequential_content += f"noti -m '=== Running {run_name} ({i}/{len(all_commands)}) ==='\n"
                sequential_content += command + "\n"
            sequential_run_path.write_text(sequential_content, encoding="utf-8")
            sequential_run_path.chmod(0o750)
            
            total_jobs += len(all_jobs)
            print(f"Generated {len(all_jobs)} job(s) in {jobs_root}")

    print(f"\n{'='*70}")
    print(f"Total jobs generated: {total_jobs}")
    print(f"Jobs organized in: {base_jobs_root}")
    print(f"Logs will be written to: {base_log_root}")
    print("\nBreakdown:")
    print(f"  - {len(MODELS)} models")
    print(f"  - {len(TASKS)} tasks")
    print(f"  - {len(MODELS)} × {len(TASKS)} = {len(MODELS) * len(TASKS)} folders")
    print(f"  - Per folder: {len(NUM_POV_AGENTS)} POV agents × {len(CONTRASTIVE_SETTINGS)} contrastive settings = {len(NUM_POV_AGENTS) * len(CONTRASTIVE_SETTINGS)} jobs")
    print(f"  - Total: {len(MODELS)} × {len(TASKS)} × {len(NUM_POV_AGENTS)} × {len(CONTRASTIVE_SETTINGS)} = {total_jobs} jobs")
    print("\nConfiguration:")
    print(f"  - task_form: {TASK_FORM}")
    print(f"  - num_pov_agents: {NUM_POV_AGENTS}")
    print(f"  - contrastive settings: {[s['suffix'] for s in CONTRASTIVE_SETTINGS]}")
    print("\nTo submit all jobs in a specific folder to SLURM, run:")
    print(f"  bash {base_jobs_root}/<folder_name>/sbatch_all_<folder_name>.sh")
    print("\nTo run all jobs in a specific folder sequentially, run:")
    print(f"  bash {base_jobs_root}/<folder_name>/run_all_<folder_name>_sequential.sh")

if __name__ == "__main__":
    main()
