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