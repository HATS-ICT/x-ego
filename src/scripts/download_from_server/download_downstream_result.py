import subprocess
from pathlib import Path

REMOTE_USER = "yunzhewa"
REMOTE_HOST = "discovery1"
REMOTE_BASE = "/project2/ustun_1726/x-ego/output"
LOCAL_BASE = Path(r"C:\Users\wangy\projects\x-ego\output")

SCRIPT_DIR = Path(__file__).parent
FOLDERS_FILE = SCRIPT_DIR / "output_folders.txt"

FILES_TO_DOWNLOAD = [
    "hparam.yaml",
    "test_results_best.json",
    "test_results_last.json",
]


def load_folders() -> list[str]:
    """Load folder names from output_folders.txt."""
    folders = []
    with FOLDERS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                folders.append(line)
    return folders


def main():
    LOCAL_BASE.mkdir(parents=True, exist_ok=True)
    folders = load_folders()

    for folder in folders:
        print(f"Downloading {folder}...")

        local_folder = LOCAL_BASE / folder
        local_folder.mkdir(parents=True, exist_ok=True)

        for filename in FILES_TO_DOWNLOAD:
            remote_file = f"{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_BASE}/{folder}/{filename}"
            result = subprocess.run(
                ["scp", remote_file, str(local_folder)], capture_output=True
            )
            if result.returncode == 0:
                print(f"  Downloaded {filename}")
            else:
                print(f"  Failed to download {filename}: {result.stderr.decode().strip()}")

        print(f"Completed {folder}")


if __name__ == "__main__":
    main()
