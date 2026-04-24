"""
Print unique trajectory `place` labels for selected maps.

Usage:
    python -m src.scripts.data_processing.check_unique_locations_from_traj
"""

from pathlib import Path
import os
import pandas as pd
from dotenv import load_dotenv


TARGET_MAPS = ["dust2", "inferno"]


def collect_unique_places(traj_root: Path) -> list[str]:
    unique_places = set()

    if not traj_root.exists():
        return []

    for match_dir in sorted(traj_root.iterdir()):
        if not match_dir.is_dir():
            continue
        for player_dir in sorted(match_dir.iterdir()):
            if not player_dir.is_dir():
                continue
            for csv_file in sorted(player_dir.glob("round_*.csv")):
                try:
                    df = pd.read_csv(csv_file, usecols=["place"])
                except Exception:
                    continue
                values = df["place"].dropna().astype(str)
                unique_places.update(place for place in values.unique().tolist() if place)

    return sorted(unique_places)


def main():
    load_dotenv()

    data_base_path = Path(os.getenv("DATA_BASE_PATH", "data"))
    if not data_base_path.is_absolute():
        data_base_path = Path(__file__).resolve().parents[3] / data_base_path

    for map_name in TARGET_MAPS:
        traj_root = data_base_path / map_name / "trajectory"
        unique_places = collect_unique_places(traj_root)
        print(f"{map_name} ({len(unique_places)} places)")
        print(unique_places)
        print()


if __name__ == "__main__":
    main()
