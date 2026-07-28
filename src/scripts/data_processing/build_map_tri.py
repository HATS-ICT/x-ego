"""
Build the per-map collision mesh (.tri) that awpy's visibility checker needs.

awpy.visibility.VisibilityChecker tests line of sight against a triangle soup
loaded from a .tri file. That file is produced from the map's .vphys collision
data, which ships inside the game's map VPK and must be extracted once per map
with Source 2 Viewer. awpy cannot download it for you.

One-time setup per map:

  1. Open Source 2 Viewer (https://valveresourceformat.github.io/).
  2. Open  <steam>/steamapps/common/Counter-Strike Global Offensive/game/csgo/
           maps/de_inferno.vpk
  3. Export  maps/de_inferno/world_physics.vphys_c  as de_inferno.vphys
  4. Put it under  {data-dir}/{map}/mesh/de_inferno.vphys
  5. Run this script, which writes  {data-dir}/{map}/mesh/de_inferno.tri

Then point the relay builder at the .tri:

    python -m src.scripts.eval_subsets.build_relay_conditions \
        --map inferno --tri-path data/inferno/mesh/de_inferno.tri

Parsing a .vphys takes a few minutes and a couple of GB of RAM. The .tri is
reusable forever, so do it once and keep it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.env_utils import resolve_data_dir


def build(vphys: Path, out: Path) -> None:
    from awpy.visibility import VphysParser

    print(f"parsing {vphys}  ({vphys.stat().st_size / 1e6:.1f} MB)")
    parser = VphysParser(vphys)
    parser.parse()
    n = len(getattr(parser, "triangles", []) or [])
    print(f"  extracted {n} triangles")
    if n == 0:
        raise SystemExit(
            "0 triangles extracted. The .vphys is probably the wrong export; it "
            "must be world_physics, not the model or navigation data."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    parser.to_tri(out)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--map", required=True, help="e.g. inferno, dust2, mirage")
    ap.add_argument("--data-dir", default=None,
                    help='defaults to $DATA_BASE_PATH from .env, else "data"')
    ap.add_argument("--vphys", default=None, help="override the .vphys path")
    ap.add_argument("--out", default=None, help="override the .tri output path")
    args = ap.parse_args()
    args.data_dir = resolve_data_dir(args.data_dir)

    full = args.map if args.map.startswith("de_") else f"de_{args.map}"
    mesh_dir = Path(args.data_dir) / args.map / "mesh"
    vphys = Path(args.vphys) if args.vphys else mesh_dir / f"{full}.vphys"
    out = Path(args.out) if args.out else mesh_dir / f"{full}.tri"

    if out.exists():
        print(f"{out} already exists; delete it to rebuild")
        return
    if not vphys.exists():
        raise SystemExit(
            f"no .vphys at {vphys}\n\n"
            "Extract it once with Source 2 Viewer, see this file's docstring for the\n"
            "exact steps, then rerun. Without the mesh the relay builder can only use\n"
            "the --teammate-backend fov fallback, which ignores walls."
        )
    build(vphys, out)


if __name__ == "__main__":
    main()
