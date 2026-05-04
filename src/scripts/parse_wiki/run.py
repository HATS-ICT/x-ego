"""End-to-end driver: fetch CS2 wiki pages then parse them into a corpus."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).parent
    repo_root = here.parents[2]
    raw_dir = repo_root / "data" / "wiki" / "raw_html"
    out_dir = repo_root / "data" / "wiki"

    print("=== Step 1: fetch ===")
    subprocess.check_call(
        [sys.executable, str(here / "fetch.py"), "--out", str(raw_dir)],
        cwd=repo_root,
    )

    print("\n=== Step 2: parse ===")
    subprocess.check_call(
        [
            sys.executable,
            str(here / "parse.py"),
            "--in",
            str(raw_dir),
            "--out",
            str(out_dir),
        ],
        cwd=repo_root,
    )


if __name__ == "__main__":
    main()
