"""
Language-side entry point for cross-ego contrastive space analysis.

This wrapper keeps language visualization commands discoverable while the
implementation lives in src/scripts/contra_cluster_v2.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.scripts.contra_cluster_v2.analyze_contrastive_space import main


if __name__ == "__main__":
    main()
