"""
Environment utilities for X-EGO project.
Provides consistent environment variable handling across all scripts.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_src_base_path() -> Path:
    """Get the base project path from environment or default."""
    return os.getenv("SRC_BASE_PATH")

def get_data_base_path() -> str:
    """Get the data directory name from environment or default."""
    return os.getenv("DATA_BASE_PATH")

def get_output_base_path() -> str:
    """Get the data directory name from environment or default."""
    return os.getenv("OUTPUT_BASE_PATH")

def resolve_data_dir(explicit: str | None = None) -> str:
    """Data directory to use, preferring an explicit CLI value over .env.

    Exists because scripts that default --data-dir to the literal "data" fail
    confusingly on the cluster, where the data lives outside the repo and the
    path is already recorded in .env as DATA_BASE_PATH. Raises with both
    remedies rather than letting a missing directory surface later as an empty
    result set.
    """
    chosen = explicit or get_data_base_path() or "data"
    if not Path(chosen).is_dir():
        raise SystemExit(
            f"data dir not found: {chosen!r}\n"
            "Pass --data-dir, or set DATA_BASE_PATH in .env."
        )
    return chosen


def print_env_info():
    """Print current environment configuration for debugging."""
    print("=" * 50)
    print("X-EGO Environment Configuration:")
    print("=" * 50)
    print(f"Src Base Path:     {get_src_base_path()}")
    print(f"Data Base Path:      {get_data_base_path()}")
    print(f"Output Base Path:    {get_output_base_path()}")
    print("=" * 50)

if __name__ == "__main__":
    print_env_info()