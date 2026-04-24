"""
Script to copy dust2 and inferno map data to the remote server.

Remote: yunzhewa@discovery2.carc.usc.edu:/project2/ustun_1726/x-ego/data

Copies the following per map:
  Folders : demo, event, metadata, trajectory, video_306x306_4fps
  Files   : match_round_partitioned.csv, time_offset.json,
            trajectory_minmax_scaler.pkl, video_list.txt
"""

import subprocess
import sys

# ── Configuration ─────────────────────────────────────────────────────────────

LOCAL_DATA_ROOT = r"C:\Users\wangy\projects\x-ego\data"
REMOTE_USER_HOST = "yunzhewa@discovery2.carc.usc.edu"
REMOTE_DATA_ROOT = "/project2/ustun_1726/x-ego/data"

MAPS = ["dust2", "inferno"]

FOLDERS_TO_COPY = [
    "demo",
    "event",
    "metadata",
    "trajectory",
    "video_306x306_4fps",
]

FILES_TO_COPY = [
    "match_round_partitioned.csv",
    "time_offset.json",
    "trajectory_minmax_scaler.pkl",
    "video_list.txt",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd: list[str]) -> None:
    """Run a command, streaming output, and exit on failure."""
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with return code {result.returncode}.")
        sys.exit(result.returncode)


def scp_folder(local_path: str, remote_dest: str) -> None:
    """Copy a folder recursively via scp."""
    run(["scp", "-r", local_path, remote_dest])


def scp_file(local_path: str, remote_dest: str) -> None:
    """Copy a single file via scp."""
    run(["scp", local_path, remote_dest])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    for map_name in MAPS:
        local_map_root = f"{LOCAL_DATA_ROOT}\\{map_name}"
        remote_map_root = f"{REMOTE_USER_HOST}:{REMOTE_DATA_ROOT}/{map_name}"

        print(f"\n{'='*60}")
        print(f"  Copying map: {map_name}")
        print(f"  Local  : {local_map_root}")
        print(f"  Remote : {REMOTE_DATA_ROOT}/{map_name}")
        print(f"{'='*60}")

        # ── Folders ──────────────────────────────────────────────────────────
        for folder in FOLDERS_TO_COPY:
            local_folder = f"{local_map_root}\\{folder}"
            scp_folder(local_folder, remote_map_root)

        # ── Files ─────────────────────────────────────────────────────────────
        for filename in FILES_TO_COPY:
            local_file = f"{local_map_root}\\{filename}"
            scp_file(local_file, remote_map_root)

    print("\n[DONE] All maps copied successfully.")


if __name__ == "__main__":
    main()
