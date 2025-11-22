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
TIME = "12:00:00"  # Longer time for embedding computation
MAIL_USER = "yunzhewa@usc.edu"
MAIL_TYPE = "all"
LOGS_SUBDIR = "logs"

# Experiment configuration
EXP_PREFIX = "pre_compute_video_embed"

# Tasks to sweep over
TASKS = [
    "teammate_location_nowcast",
    "enemy_location_nowcast",
]

# Encoders to sweep over
ENCODERS = [
    "dinov2",
    "siglip",
    "vivit",
    "videomae",
    "vjepa2",
]

# Batch size for embedding
BATCH_SIZE = 16

# Number of workers for data loading
NUM_WORKERS = 8

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

uv run python scripts/data_processing/embed_video.py \\
  --task {task} \\
  --encoders {encoder} \\
  --batch-size {batch_size} \\
  --num-workers {num_workers}
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
# Sequential execution of all embedding jobs on a single machine
# Run this script to execute all embedding tasks one after another
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
    jobs_root = project_src / "jobs" / EXP_PREFIX
    log_root = project_src / LOGS_SUBDIR / EXP_PREFIX
    
    jobs_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    mail_block = build_mail_block(MAIL_USER, MAIL_TYPE)

    all_jobs = []
    all_commands = []  # Store commands for sequential run script

    # Generate jobs for each combination
    for task in TASKS:
        task_short = get_task_short_name(task)
        
        for encoder in ENCODERS:
            run_name = f"{EXP_PREFIX}-{task_short}-{encoder}"
            
            header = SCRIPT_HEADER.format(
                account=ACCOUNT, partition=PARTITION, cpus=CPUS,
                gpu_constraint=GPU_CONSTRAINT, gpu_count=GPU_COUNT,
                mem=MEM, time=TIME,
                job_name=run_name, log_dir=str(log_root),
                mail_block=mail_block,
            ).rstrip()
            
            body = JOB_BODY.format(
                project_src=str(project_src),
                task=task,
                encoder=encoder,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
            ).rstrip()

            content = header + "\n\n" + body + "\n"
            job_path = jobs_root / f"{run_name}.job"
            job_path.write_text(content, encoding="utf-8")
            job_path.chmod(0o750)
            all_jobs.append(job_path)
            
            # Extract the command for sequential run script
            command = f"""uv run python scripts/data_processing/embed_video.py \\
  --task {task} \\
  --encoders {encoder} \\
  --batch-size {BATCH_SIZE} \\
  --num-workers {NUM_WORKERS}"""
            all_commands.append((run_name, command))

    # Generate sbatch_all script
    sbatch_all_path = jobs_root / f"sbatch_all_{EXP_PREFIX}.sh"
    sbatch_all_path.write_text(SBATCH_ALL_HEADER, encoding="utf-8")
    sbatch_all_path.chmod(0o750)
    
    # Generate sequential run script
    sequential_run_path = jobs_root / f"run_all_{EXP_PREFIX}_sequential.sh"
    sequential_content = SEQUENTIAL_RUN_HEADER.format(project_src=str(project_src))
    for i, (run_name, command) in enumerate(all_commands, 1):
        sequential_content += f"\n# Job {i}/{len(all_commands)}: {run_name}\n"
        sequential_content += f"echo '=== Running {run_name} ({i}/{len(all_commands)}) ==='\n"
        sequential_content += command + "\n"
    sequential_run_path.write_text(sequential_content, encoding="utf-8")
    sequential_run_path.chmod(0o750)

    print(f"Generated {len(all_jobs)} job(s) in {jobs_root}")
    print(f"Logs will be written to {log_root}")
    print("\nBreakdown:")
    print(f"  - {len(TASKS)} tasks")
    print(f"  - {len(ENCODERS)} encoders")
    print(f"  - Total: {len(TASKS)} × {len(ENCODERS)} = {len(all_jobs)} jobs")
    print("\nConfiguration:")
    print(f"  - batch_size: {BATCH_SIZE}")
    print(f"  - num_workers: {NUM_WORKERS}")
    print(f"  - encoders: {ENCODERS}")
    print(f"  - tasks: {TASKS}")
    print("\nTo submit all jobs to SLURM, run:")
    print(f"  bash {sbatch_all_path}")
    print("\nTo run all jobs sequentially on a single machine, run:")
    print(f"  bash {sequential_run_path}")

if __name__ == "__main__":
    main()

