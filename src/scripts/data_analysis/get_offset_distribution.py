"""
Analyse the distribution of video-to-trajectory time offsets across all maps.

offset_sec = video_time_at_event - game_time_at_event
  > 0  → video started BEFORE the round (normal, recording caught the freeze time)
  < 0  → video started AFTER round start (late recording, first N seconds missing)

Usage:
    python -m src.scripts.data_analysis.get_offset_distribution
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_ROOT = Path(r"c:\Users\wangy\projects\x-ego\data")
MAPS = ["dust2", "inferno", "mirage"]
OUT_PNG = DATA_ROOT / "offset_distribution.png"


# ---------------------------------------------------------------------------
# Load all offsets
# ---------------------------------------------------------------------------
def load_offsets(
    data_root: Path, maps: list[str]
) -> tuple[dict[str, list[float]], list[dict]]:
    """
    Returns
    -------
    per_map  : {map_name: [offset_sec, ...]}
    records  : list of dicts with full context for every round
    """
    per_map: dict[str, list[float]] = {}
    records: list[dict] = []

    for map_name in maps:
        offset_file = data_root / map_name / "time_offset.json"
        if not offset_file.exists():
            print(f"[SKIP] {map_name}: no time_offset.json found")
            continue

        with open(offset_file, "r") as f:
            data = json.load(f)

        offsets: list[float] = []
        for match_id, players in data.items():
            for player_id, rounds in players.items():
                for round_key, info in rounds.items():
                    offset = float(info["offset_sec"])
                    offsets.append(offset)
                    records.append({
                        "map":        map_name,
                        "match_id":   match_id,
                        "player_id":  player_id,
                        "round":      round_key,
                        "offset_sec": offset,
                        "video_fps":  info.get("video_fps"),
                        "start_timer": info.get("start_timer"),
                        "transition_timer": info.get("transition_timer"),
                        "transition_frame": info.get("transition_frame"),
                        "transition_video_sec": info.get("transition_video_sec"),
                        "transition_game_sec":  info.get("transition_game_sec"),
                    })

        per_map[map_name] = offsets
        print(f"[OK] {map_name}: loaded {len(offsets)} rounds")

    return per_map, records


# ---------------------------------------------------------------------------
# Print statistics
# ---------------------------------------------------------------------------
def print_stats(map_name: str, offsets: list[float]) -> None:
    arr = np.array(offsets)
    n_neg = int((arr < 0).sum())
    n_pos = int((arr >= 0).sum())

    print(f"\n{'='*55}")
    print(f"  Map: {map_name}  ({len(arr)} rounds total)")
    print(f"{'='*55}")
    print(f"  positive offset (video started EARLY):  {n_pos:4d}  ({100*n_pos/len(arr):.1f}%)")
    print(f"  negative offset (video started LATE) :  {n_neg:4d}  ({100*n_neg/len(arr):.1f}%)")
    print(f"  min    : {arr.min():+.4f} s")
    print(f"  max    : {arr.max():+.4f} s")
    print(f"  mean   : {arr.mean():+.4f} s")
    print(f"  median : {np.median(arr):+.4f} s")
    print(f"  std    : {arr.std():.4f} s")

    # Late-recording severity: how many seconds of content are lost?
    if n_neg > 0:
        lost = arr[arr < 0]
        print(f"\n  --- Late recordings (offset < 0) ---")
        print(f"  avg seconds lost : {(-lost).mean():.2f} s")
        print(f"  max seconds lost : {(-lost).max():.2f} s")
        # Bucket by seconds lost
        for threshold in [1, 2, 3, 5, 10]:
            count = int(((-lost) >= threshold).sum())
            print(f"  >= {threshold:2d}s lost       : {count}")


# ---------------------------------------------------------------------------
# Print anomalous (large positive offset) and positive-offset samples
# ---------------------------------------------------------------------------
ROW_FMT = (
    "  {map:<8}  {match_id:<45}  {player_id:<20}  "
    "{round:<10}  offset={offset_sec:+10.4f}s  "
    "start_timer={start_timer}  trans_timer={transition_timer}  "
    "trans_frame={transition_frame}"
)


def _fmt(r: dict) -> str:
    return ROW_FMT.format(
        map=r["map"],
        match_id=r["match_id"],
        player_id=r["player_id"],
        round=r["round"],
        offset_sec=r["offset_sec"],
        start_timer=r.get("start_timer", "?"),
        transition_timer=r.get("transition_timer", "?"),
        transition_frame=r.get("transition_frame", "?"),
    )


def print_anomalous(
    records: list[dict],
    pos_threshold: float = 10.0,
    neg_threshold: float = -5.0,
    top_n: int = 30,
) -> None:
    """Print the worst large-positive and large-negative (beyond normal) offsets."""
    large_pos = sorted(
        [r for r in records if r["offset_sec"] >= pos_threshold],
        key=lambda r: r["offset_sec"], reverse=True
    )
    large_neg = sorted(
        [r for r in records if r["offset_sec"] <= neg_threshold],
        key=lambda r: r["offset_sec"]
    )

    print(f"\n{'#'*70}")
    print(f"  ANOMALOUS: large POSITIVE offsets (>= +{pos_threshold}s)  [{len(large_pos)} rounds]")
    print(f"  (These likely indicate a bad OCR detection — score screen, etc.)")
    print(f"{'#'*70}")
    if large_pos:
        for r in large_pos[:top_n]:
            print(_fmt(r))
        if len(large_pos) > top_n:
            print(f"  ... and {len(large_pos) - top_n} more")
    else:
        print("  None found.")

    print(f"\n{'#'*70}")
    print(f"  ANOMALOUS: large NEGATIVE offsets (<= {neg_threshold}s)  [{len(large_neg)} rounds]")
    print(f"  (These rounds are missing >= {abs(neg_threshold)}s of footage from the start)")
    print(f"{'#'*70}")
    if large_neg:
        for r in large_neg[:top_n]:
            print(_fmt(r))
        if len(large_neg) > top_n:
            print(f"  ... and {len(large_neg) - top_n} more")
    else:
        print("  None found.")


def print_positive_samples(records: list[dict], n: int = 15) -> None:
    """Print a few rounds with positive offset (video started before round)."""
    pos = sorted(
        [r for r in records if 0 < r["offset_sec"] < 10.0],
        key=lambda r: r["offset_sec"], reverse=True
    )
    print(f"\n{'#'*70}")
    print(f"  POSITIVE (normal early-start) offsets: 0 < offset < 10s  [{len(pos)} rounds]")
    print(f"  (Showing up to {n} — sorted largest first)")
    print(f"{'#'*70}")
    if pos:
        for r in pos[:n]:
            print(_fmt(r))
    else:
        print("  None found.")



# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_distributions(per_map: dict[str, list[float]], out_png: Path) -> None:
    maps = list(per_map.keys())
    n_maps = len(maps)
    if n_maps == 0:
        print("[WARN] No data to plot.")
        return

    # Color palette
    colors = ["#4e9af1", "#f17c4e", "#4ef19a"]

    fig, axes = plt.subplots(
        n_maps, 2,
        figsize=(14, 4 * n_maps),
        gridspec_kw={"width_ratios": [3, 1]},
        squeeze=False,
    )
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Video-to-Trajectory Offset Distribution", fontsize=16,
                 color="white", fontweight="bold", y=1.01)

    all_offsets = [v for vals in per_map.values() for v in vals]
    global_min = min(all_offsets) - 0.5
    global_max = max(all_offsets) + 0.5

    for row, (map_name, offsets) in enumerate(per_map.items()):
        arr = np.array(offsets)
        color = colors[row % len(colors)]
        ax_hist = axes[row, 0]
        ax_box  = axes[row, 1]

        for ax in (ax_hist, ax_box):
            ax.set_facecolor("#16213e")
            for spine in ax.spines.values():
                spine.set_color("#444")

        # --- Histogram ---
        bins = np.linspace(global_min, global_max, 60)
        counts_neg, _, _ = ax_hist.hist(
            arr[arr < 0], bins=bins,
            color="#e05c5c", alpha=0.85, label="offset < 0  (late start)"
        )
        counts_pos, _, _ = ax_hist.hist(
            arr[arr >= 0], bins=bins,
            color=color, alpha=0.85, label="offset ≥ 0  (early/on-time)"
        )

        ax_hist.axvline(0, color="white", linewidth=1.2, linestyle="--", alpha=0.6)
        ax_hist.axvline(float(np.median(arr)), color="yellow", linewidth=1.2,
                        linestyle=":", alpha=0.8, label=f"median = {np.median(arr):+.2f}s")

        ax_hist.set_title(f"{map_name}  (n={len(arr)})", color="white",
                          fontsize=13, fontweight="bold")
        ax_hist.set_xlabel("offset_sec  (video_time − game_time)", color="#aaa")
        ax_hist.set_ylabel("# rounds", color="#aaa")
        ax_hist.tick_params(colors="#aaa")
        ax_hist.legend(fontsize=9, framealpha=0.3, labelcolor="white")
        ax_hist.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        # Annotate late %
        n_neg = int((arr < 0).sum())
        pct_neg = 100 * n_neg / len(arr)
        ax_hist.text(0.02, 0.96, f"Late: {n_neg}/{len(arr)}  ({pct_neg:.1f}%)",
                     transform=ax_hist.transAxes, color="#e05c5c",
                     fontsize=10, va="top",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#111", alpha=0.6))

        # --- Boxplot ---
        bp = ax_box.boxplot(arr, vert=True, patch_artist=True,
                            medianprops=dict(color="yellow", linewidth=2),
                            whiskerprops=dict(color="#aaa"),
                            capprops=dict(color="#aaa"),
                            flierprops=dict(marker="o", color="#e05c5c",
                                            markerfacecolor="#e05c5c", markersize=3))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax_box.axhline(0, color="white", linewidth=1.0, linestyle="--", alpha=0.6)
        ax_box.set_ylabel("offset_sec", color="#aaa")
        ax_box.tick_params(colors="#aaa")
        ax_box.set_xticks([])
        ax_box.set_title("Box", color="#aaa", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[Plot saved] {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading offsets …")
    per_map, records = load_offsets(DATA_ROOT, MAPS)

    if not per_map:
        print("No offset data found. Exiting.")
        return

    # Print stats per map
    for map_name, offsets in per_map.items():
        print_stats(map_name, offsets)

    # Cross-map summary
    all_offsets = [v for vals in per_map.values() for v in vals]
    arr_all = np.array(all_offsets)
    n_neg_all = int((arr_all < 0).sum())
    print(f"\n{'='*55}")
    print(f"  TOTAL across all maps: {len(arr_all)} rounds")
    print(f"  Late recordings (< 0): {n_neg_all}  ({100*n_neg_all/len(arr_all):.1f}%)")
    print(f"  Overall mean offset  : {arr_all.mean():+.4f} s")
    print(f"  Overall median offset: {np.median(arr_all):+.4f} s")
    print(f"{'='*55}")

    # Anomalous entries
    print_anomalous(records, pos_threshold=10.0, neg_threshold=-5.0, top_n=30)

    # Sample of normal positive offsets
    print_positive_samples(records, n=15)

    plot_distributions(per_map, OUT_PNG)


if __name__ == "__main__":
    main()
