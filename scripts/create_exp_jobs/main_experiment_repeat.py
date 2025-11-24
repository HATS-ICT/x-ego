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
TIME = "48:00:00"  # Longer time since we're running multiple seeds
MAIL_USER = "yunzhewa@usc.edu"
MAIL_TYPE = "all"
LOGS_SUBDIR = "logs"

# Experiment configuration
EXP_PREFIX = "main_exp_repeat_v1"
TASK_FORM = "multi-label-cls"
NUM_EPOCHS = 40
NUM_SEEDS = 5

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
    "vjepa2",
    "clip",
]

# Number of POV agents to sweep over
NUM_POV_AGENTS = [1, 2, 3, 4, 5]

# Random seeds to use
SEEDS = list(range(1, NUM_SEEDS + 1))

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

JOB_BODY_BASH = """module restore

cd {project_src}

bash {script_path}
"""

SEQUENTIAL_RUN_HEADER_BASH = """#!/bin/bash
# Sequential execution of all experiments for {model} model
# Run this script to execute all experiments one after another
set -euo pipefail

cd {project_src}

"""

SEQUENTIAL_RUN_HEADER_BAT = """@echo off
REM Sequential execution of all experiments for {model} model
REM Run this script to execute all experiments one after another

cd /d {project_src}

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
    return task_map[task]

def generate_command(task: str, num_pov: int, model: str, with_contra: bool, seed: int) -> tuple[str, str]:
    """Generate command and run name for a specific configuration"""
    task_short = get_task_short_name(task)
    contra_str = "contra" if with_contra else "no-contra"
    run_name = f"{EXP_PREFIX}-{task_short}-{model}-pov{num_pov}-{contra_str}-seed{seed}"
    
    if with_contra:
        command = f"""uv run python main.py --mode train --task {task} \\
  data.num_pov_agents={num_pov} \\
  data.task_form={TASK_FORM} \\
  model.encoder.video.model_type={model} \\
  model.contrastive.enable=true \\
  training.num_epochs={NUM_EPOCHS} \\
  meta.exp_name={EXP_PREFIX} \\
  meta.run_name={run_name} \\
  meta.seed={seed}"""
    else:
        command = f"""uv run python main.py --mode train --task {task} \\
  data.num_pov_agents={num_pov} \\
  data.task_form={TASK_FORM} \\
  model.encoder.video.model_type={model} \\
  model.contrastive.enable=false \\
  training.num_epochs={NUM_EPOCHS} \\
  meta.exp_name={EXP_PREFIX} \\
  meta.run_name={run_name} \\
  meta.seed={seed}"""
    
    return run_name, command

# ===== Main =====
def main():
    project_src = Path(PROJECT_SRC).resolve()
    jobs_root = project_src / "jobs" / EXP_PREFIX
    log_root = project_src / LOGS_SUBDIR / EXP_PREFIX
    
    jobs_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    mail_block = build_mail_block(MAIL_USER, MAIL_TYPE)

    total_runs = 0
    all_job_files = []

    # Generate one script per (model, task) combination
    for model in MODELS:
        for task in TASKS:
            all_commands = []
            task_short = get_task_short_name(task)
            
            # Order: for each seed, do with_contra and without_contra
            for seed in SEEDS:
                # For each pov_num
                for num_pov in NUM_POV_AGENTS:
                    # For each contra setting (yes then no)
                    for with_contra in [True, False]:
                        run_name, command = generate_command(task, num_pov, model, with_contra, seed)
                        all_commands.append((run_name, command))
                        total_runs += 1
            
            # Generate bash script for this model-task combination
            bash_script_path = jobs_root / f"run_{model}_{task_short}_all_seeds.sh"
            bash_content = SEQUENTIAL_RUN_HEADER_BASH.format(
                model=f"{model} on {task_short}",
                project_src=str(project_src)
            )
            
            for i, (run_name, command) in enumerate(all_commands, 1):
                bash_content += f"\n# Run {i}/{len(all_commands)}: {run_name}\n"
                bash_content += f"echo '=== Running {run_name} ({i}/{len(all_commands)}) ==='\n"
                bash_content += command + "\n"
            
            bash_script_path.write_text(bash_content, encoding="utf-8")
            bash_script_path.chmod(0o750)
            
            # Generate Windows batch script for this model-task combination
            bat_script_path = jobs_root / f"run_{model}_{task_short}_all_seeds.bat"
            bat_content = SEQUENTIAL_RUN_HEADER_BAT.format(
                model=f"{model} on {task_short}",
                project_src=str(project_src)
            )
            
            for i, (run_name, command) in enumerate(all_commands, 1):
                # Convert bash line continuation to Windows batch
                command_bat = command.replace(" \\\n", " ^\n")
                bat_content += f"\nREM Run {i}/{len(all_commands)}: {run_name}\n"
                bat_content += f"echo === Running {run_name} ({i}/{len(all_commands)}) ===\n"
                bat_content += command_bat + "\n"
            
            bat_script_path.write_text(bat_content, encoding="utf-8")
            
            # Generate SLURM job file that runs the bash script
            job_name = f"{EXP_PREFIX}-{model}-{task_short}"
            header = SCRIPT_HEADER.format(
                account=ACCOUNT, partition=PARTITION, cpus=CPUS,
                gpu_constraint=GPU_CONSTRAINT, gpu_count=GPU_COUNT,
                mem=MEM, time=TIME,
                job_name=job_name, log_dir=str(log_root),
                mail_block=mail_block,
            ).rstrip()
            
            body = JOB_BODY_BASH.format(
                project_src=str(project_src),
                script_path=str(bash_script_path),
            ).rstrip()
            
            content = header + "\n\n" + body + "\n"
            job_path = jobs_root / f"{job_name}.job"
            job_path.write_text(content, encoding="utf-8")
            job_path.chmod(0o750)
            
            all_job_files.append((model, task_short, bash_script_path, bat_script_path, job_path))

    print(f"Generated job scripts in {jobs_root}")
    print(f"Logs will be written to {log_root}")
    print("\nBreakdown:")
    print(f"  - {len(MODELS)} models")
    print(f"  - {len(TASKS)} tasks")
    print(f"  - {len(NUM_POV_AGENTS)} num_pov_agents values")
    print(f"  - 2 contrastive settings (with/without)")
    print(f"  - {NUM_SEEDS} seeds per configuration")
    print(f"  - Total runs: {total_runs}")
    print(f"  - Script files: {len(all_job_files)} (one per model-task combination)")
    print(f"  - Runs per script: {total_runs // len(all_job_files)}")
    print("\nConfiguration:")
    print(f"  - task_form: {TASK_FORM}")
    print(f"  - num_epochs: {NUM_EPOCHS}")
    print(f"  - seeds: {SEEDS}")
    print(f"  - num_pov_agents: {NUM_POV_AGENTS}")
    print("\nGenerated files (per model-task combination):")
    for model, task_short, bash_path, bat_path, job_path in all_job_files:
        print(f"\n  {model} on {task_short}:")
        print(f"    - {bash_path.name}")
        print(f"    - {bat_path.name}")
        print(f"    - {job_path.name}")
    print("\n\nTo submit all jobs to SLURM, run:")
    for model, task_short, _, _, job_path in all_job_files:
        print(f"  sbatch {job_path}")
    print("\nTo run locally (e.g., for dinov2 on enemy nowcast), run:")
    print(f"  bash {jobs_root / 'run_dinov2_en-nowcast_all_seeds.sh'}  # Linux/Mac")
    print(f"  {jobs_root / 'run_dinov2_en-nowcast_all_seeds.bat'}  # Windows")

if __name__ == "__main__":
    main()

