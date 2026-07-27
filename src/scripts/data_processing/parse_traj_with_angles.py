"""
Re-parse per-player trajectories with view angles, and spotted state if available.

The existing parse (parse_traj_per_player.py) requests `player_props=[]` and keeps
only tick/steamid/name/side/X/Y/Z/place/health. That is enough for region labels
but not for any visibility test, which needs where each player is looking. This
script writes the same per-player per-round CSV layout into a separate folder so
nothing downstream changes until you point at it.

Output layout, mirroring the original:
    {data_dir}/{map}/{out_folder}/{match_id}/{steamid}/round_{N}.csv

Prop names vary by demoparser2 version, so start with discovery:

    python -m src.scripts.data_processing.parse_traj_with_angles \
        --map inferno --list-props

That prints the props the installed parser accepts. Then run for real, adjusting
--props if a name differs:

    python -m src.scripts.data_processing.parse_traj_with_angles \
        --map inferno --props pitch,yaw

Angles alone give a field-of-view cone, which over-counts visibility because it
ignores walls. Combine with awpy's triangle-mesh line-of-sight, or with an engine
spotted flag if your parser exposes one, before reporting visibility conditions.
See src/scripts/eval_subsets/build_relay_conditions.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

BASE_COLUMNS = [
    "tick_norm", "tick", "game_sec", "round_num", "map_name",
    "steamid", "name", "side", "X", "Y", "Z", "place", "health",
]

# Candidate angle and visibility props, in preference order. Not all builds
# expose all of these; unavailable ones are dropped with a warning rather than
# failing the run.
DEFAULT_PROPS = ["pitch", "yaw"]
OPTIONAL_PROPS = ["is_spotted", "spotted", "spotted_by_mask", "is_alive", "flash_duration"]


def list_props(demo_path: str) -> None:
    """Print the props the installed demoparser2 accepts for this demo."""
    from awpy import Demo

    dem = Demo(demo_path, tickrate=64)
    for attr in ("list_props", "available_props", "props"):
        fn = getattr(dem, attr, None)
        if callable(fn):
            print(f"via {attr}():")
            print(fn())
            return
        if fn is not None:
            print(f"via {attr}:")
            print(fn)
            return
    # No listing API. Force an error, which typically enumerates valid names.
    try:
        dem.parse(player_props=["__definitely_not_a_prop__"])
    except Exception as exc:
        print("Parser rejected a bogus prop. Its message usually lists valid names:\n")
        print(exc)
        return
    print(
        "No listing API found and a bogus prop did not raise. Consult the "
        "demoparser2 documentation for the prop list."
    )


def parse_one(dem_file: str, out_root: Path, map_name: str, props: list[str]) -> dict:
    import polars as pl
    from awpy import Demo

    demo_id = Path(dem_file).stem
    meta_path = Path(dem_file).parent.parent / "metadata" / f"{demo_id}.json"
    if not meta_path.exists():
        return {"success": False, "file": dem_file, "error": f"no metadata at {meta_path}"}
    with open(meta_path, "r", encoding="utf-8") as f:
        player_alive_times = json.load(f)["player_alive_times"]

    dem = Demo(dem_file, tickrate=64)

    # Ask for everything requested, then fall back to whatever parsed.
    try:
        dem.parse(player_props=props)
    except Exception as exc:
        return {
            "success": False,
            "file": dem_file,
            "error": f"parse failed for props={props}: {exc}",
        }

    dem.ticks = dem.ticks.with_columns(pl.lit(map_name).alias("map_name"))
    present = set(dem.ticks.columns)

    missing = [p for p in props if p not in present]
    if missing:
        print(f"  WARNING: props absent after parse, dropping: {missing}")
    keep = BASE_COLUMNS + [p for p in props if p in present]
    absent_base = [c for c in BASE_COLUMNS if c not in present and c not in ("tick_norm", "game_sec")]
    if absent_base:
        return {
            "success": False,
            "file": dem_file,
            "error": f"required base columns missing: {absent_base}",
        }

    out_dir = out_root / demo_id
    os.makedirs(out_dir, exist_ok=True)

    for round_num, round_data in player_alive_times.items():
        for entry in round_data:
            steamid = entry["steamid"]
            start_tick = int(entry["alive_start_tick"])
            end_tick = int(entry["alive_end_tick"]) - 1

            sub = dem.ticks.filter(
                (pl.col("steamid").cast(pl.Int64) == int(steamid))
                & (pl.col("tick").cast(pl.Int64) >= start_tick)
                & (pl.col("tick").cast(pl.Int64) <= end_tick)
            )
            if sub.is_empty():
                continue
            sub = sub.with_columns((pl.col("tick") - start_tick).alias("tick_norm"))
            sub = sub.with_columns((pl.col("tick_norm") / 64.0).round(3).alias("game_sec"))
            sub = sub.select([c for c in keep if c in sub.columns])

            player_dir = out_dir / str(steamid)
            os.makedirs(player_dir, exist_ok=True)
            sub.to_pandas().to_csv(player_dir / f"round_{round_num}.csv", index=False)

    return {"success": True, "file": dem_file, "kept_props": [p for p in props if p in present]}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--map", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-folder", default="trajectory_angles")
    ap.add_argument(
        "--props",
        default=",".join(DEFAULT_PROPS),
        help=f"comma-separated player props. Optional extras worth trying: {OPTIONAL_PROPS}",
    )
    ap.add_argument("--list-props", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="parse only the first N demos")
    args = ap.parse_args()

    demo_dir = Path(args.data_dir) / args.map / "demo"
    demos = sorted(str(p) for p in demo_dir.glob("*.dem"))
    if not demos:
        print(f"No .dem files under {demo_dir}")
        return

    if args.list_props:
        list_props(demos[0])
        return

    props = [p.strip() for p in args.props.split(",") if p.strip()]
    out_root = Path(args.data_dir) / args.map / args.out_folder
    out_root.mkdir(parents=True, exist_ok=True)

    if args.limit:
        demos = demos[: args.limit]

    map_full = args.map if args.map.startswith("de_") else f"de_{args.map}"
    ok = 0
    for i, dem_file in enumerate(demos, 1):
        print(f"[{i}/{len(demos)}] {Path(dem_file).name}")
        res = parse_one(dem_file, out_root, map_full, props)
        if res["success"]:
            ok += 1
        else:
            print(f"  FAILED: {res['error']}")

    print(f"\n{ok}/{len(demos)} demos parsed into {out_root}")
    if ok:
        print(
            "Next: python -m src.scripts.eval_subsets.build_relay_conditions "
            f"--map {args.map} --trajectory-folder {args.out_folder} --backend geometric"
        )


if __name__ == "__main__":
    main()
