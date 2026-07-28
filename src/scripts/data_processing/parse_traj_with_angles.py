"""
Re-parse per-player trajectories with view angles and engine spotted state.

The existing parse (parse_traj_per_player.py) requests `player_props=[]` and keeps
only tick/steamid/name/side/X/Y/Z/place/health. That is enough for region labels
but not for any visibility test. This script adds the fields needed for one, and
writes the same per-player per-round CSV layout into a separate folder so nothing
downstream changes until you point at it.

Output layout, mirroring the original:
    {data_dir}/{map}/{out_folder}/{match_id}/{steamid}/round_{N}.csv

Props added, all listed as supported by demoparser2:

    pitch, yaw              m_angEyeAngles, the eye direction. Needed for any
                            field-of-view test.
    spotted                 m_bSpotted. True when this player is spotted by the
                            opposing team. Team-level, so it cannot say WHO sees
                            them.
    approximate_spotted_by  m_bSpottedByMask. A bitmask over player slots naming
                            which players have spotted this one. This is the field
                            the relay experiment needs, because it distinguishes
                            "teammate A sees the enemy" from "somebody on my team
                            sees the enemy".
    entity_id               Needed to decode the bitmask above, whose bits are
                            indexed by entity slot rather than by steamid.
    is_alive, flash_duration, is_scoped
                            A blinded observer is not an observer. Kept so
                            conditions can exclude them.

IMPORTANT on the spotted mask. The engine maintains spotted state for ENEMIES. A
player's mask names the opponents who can see them, not their teammates, since
teammates appear on the radar regardless of line of sight. So the mask answers
"does teammate A see enemy E" exactly, but says nothing about "does B see teammate
A". That second predicate needs geometry. See
src/scripts/eval_subsets/visibility.py.

Prop names vary by parser version, so verify before a long run:

    python -m src.scripts.data_processing.parse_traj_with_angles --map inferno --list-props

Then run for real, on one demo first:

    python -m src.scripts.data_processing.parse_traj_with_angles --map inferno --limit 1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.utils.env_utils import resolve_data_dir

BASE_COLUMNS = [
    "tick_norm", "tick", "game_sec", "round_num", "map_name",
    "steamid", "name", "side", "X", "Y", "Z", "place", "health",
]

# Required for the relay conditions. Losing either is a hard failure, since
# without eye angles there is no visibility test at all.
REQUIRED_PROPS = ["pitch", "yaw"]

# Wanted, but the pipeline degrades without them: the geometric backend can stand
# in for the mask at the cost of over-counting visibility.
OPTIONAL_PROPS = [
    "approximate_spotted_by",  # m_bSpottedByMask, the exact per-observer answer
    "spotted",                 # m_bSpotted, team-level fallback
    "entity_id",               # needed to decode the mask bits
    "is_alive",
    "flash_duration",
    "is_scoped",
]
DEFAULT_PROPS = REQUIRED_PROPS + OPTIONAL_PROPS


def list_props(demo_path: str) -> None:
    """Probe which of the props we want this parser build actually delivers.

    Asking is not enough: some builds accept a prop name and then omit the column,
    so this reports acceptance and presence separately, with a value sample.
    """
    from demoparser2 import DemoParser

    parser = DemoParser(demo_path)
    print(f"demo: {demo_path}\n")
    print("probing each prop this script wants:")
    for p in DEFAULT_PROPS:
        try:
            df = parser.parse_ticks([p])
        except Exception as exc:
            print(f"  {p:24s} REJECTED  {type(exc).__name__}: {str(exc)[:110]}")
            continue
        if p not in df.columns:
            print(f"  {p:24s} ACCEPTED BUT NO COLUMN IN OUTPUT")
            continue
        nn = int(df[p].notna().sum())
        sample = df[p].dropna().unique()[:3].tolist()
        print(f"  {p:24s} OK  non-null {nn}/{len(df)}  dtype {df[p].dtype}  e.g. {sample}")
    print(
        "\nInterpretation. pitch and yaw must be OK or nothing downstream works.\n"
        "approximate_spotted_by should be an integer bitmask; if it is a list or a\n"
        "string, visibility.py handles both but confirm the sample above looks like\n"
        "slot bits and not steamids."
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

    # Ask for everything, then drop whatever this build did not deliver. A single
    # unsupported name can fail the whole parse, so retry with the required subset
    # rather than losing the demo.
    try:
        dem.parse(player_props=props)
    except Exception:
        try:
            dem.parse(player_props=REQUIRED_PROPS)
            props = list(REQUIRED_PROPS)
        except Exception as exc:
            return {
                "success": False,
                "file": dem_file,
                "error": f"parse failed even for {REQUIRED_PROPS}: {exc}",
            }

    dem.ticks = dem.ticks.with_columns(pl.lit(map_name).alias("map_name"))
    present = set(dem.ticks.columns)

    missing_required = [p for p in REQUIRED_PROPS if p not in present]
    if missing_required:
        return {
            "success": False,
            "file": dem_file,
            "error": f"required props absent after parse: {missing_required}. "
                     "Run --list-props to find the right names for this build.",
        }
    dropped = [p for p in props if p not in present]
    keep = BASE_COLUMNS + [p for p in props if p in present]
    absent_base = [
        c for c in BASE_COLUMNS
        if c not in present and c not in ("tick_norm", "game_sec")
    ]
    if absent_base:
        return {
            "success": False,
            "file": dem_file,
            "error": f"required base columns missing: {absent_base}",
        }

    out_dir = out_root / demo_id
    os.makedirs(out_dir, exist_ok=True)

    n_written = 0
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
            n_written += 1

    return {
        "success": True,
        "file": dem_file,
        "kept_props": [p for p in props if p in present],
        "dropped_props": dropped,
        "files_written": n_written,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--map", required=True)
    ap.add_argument("--data-dir", default=None,
                    help='defaults to $DATA_BASE_PATH from .env, else "data"')
    ap.add_argument("--out-folder", default="trajectory_angles")
    ap.add_argument(
        "--props",
        default=",".join(DEFAULT_PROPS),
        help=f"comma-separated player props. Default: {DEFAULT_PROPS}",
    )
    ap.add_argument("--list-props", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="parse only the first N demos")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip demos whose output folder already exists")
    args = ap.parse_args()
    args.data_dir = resolve_data_dir(args.data_dir)

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
    ok, skipped, all_dropped = 0, 0, set()
    for i, dem_file in enumerate(demos, 1):
        demo_id = Path(dem_file).stem
        if args.skip_existing and (out_root / demo_id).exists():
            skipped += 1
            continue
        print(f"[{i}/{len(demos)}] {Path(dem_file).name}")
        res = parse_one(dem_file, out_root, map_full, props)
        if res["success"]:
            ok += 1
            all_dropped.update(res.get("dropped_props") or [])
            if i == 1:
                print(f"  kept props: {res['kept_props']}")
        else:
            print(f"  FAILED: {res['error']}")

    print(f"\n{ok}/{len(demos)} demos parsed into {out_root}"
          + (f", {skipped} skipped" if skipped else ""))
    if all_dropped:
        print(f"props unavailable in this parser build: {sorted(all_dropped)}")
        if "approximate_spotted_by" in all_dropped:
            print(
                "  The spotted bitmask is the exact per-observer signal. Without it\n"
                "  the relay conditions fall back to geometry, which over-counts\n"
                "  visibility because it ignores walls. Check --list-props."
            )
    if ok:
        print(
            "\nNext, validate the visibility signal before building conditions:\n"
            f"  python -m src.scripts.eval_subsets.validate_visibility --map {args.map}"
        )


if __name__ == "__main__":
    main()
