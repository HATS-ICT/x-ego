
## Explain Offset

### What is `offset_sec`?

`offset_sec` is the **time (in seconds) that the video starts before game-time zero**.

```
offset_sec = video_time_at_event − game_time_at_event
```

The script detects the first in-game timer tick (e.g. `1:55 → 1:54`) and records:
- `transition_video_sec` — when that tick happened on the video clock
- `transition_game_sec` — what game-time that tick corresponds to (seconds elapsed since round start, where `1:55` = 0 s, `1:54` = 1 s, …)

```
Video timeline:  |----[offset_sec]----[game t=0 (timer=1:55)]----[game t=5s]---- …
                 0s                  offset_sec                offset_sec+5s
```

A **positive** offset means the video started recording before the round clock hit 1:55.  
A **negative** offset would mean the video clip started after round start (rare / clipped recording).

> `offset_sec` is stored per round in `data/<map>/time_offset.json`:
> ```
> time_offset.json[match_id][player_id][round_name]["offset_sec"]
> ```

---

### Aligning video + trajectory for a single video

Trajectory ticks are in **game-time** (tick index from round start, at 64 TPS).  
Video frames are in **video-time** (seconds from the start of the `.mp4` file).

To convert a game tick → the corresponding video frame:

```python
game_sec   = tick / 64.0                        # game-time in seconds
video_sec  = game_sec + offset_sec              # position in the video file
video_frame = int(video_sec * video_fps)        # frame index to seek to
```

To convert a video frame → game tick (inverse):

```python
video_sec  = frame_index / video_fps
game_sec   = video_sec - offset_sec
tick       = int(game_sec * 64)
```

**Quick sanity check:** at `tick=0` (round start / timer=1:55) the video should be at second `offset_sec`. Seek there and verify the on-screen timer reads `1:55`.

---

### Loading multiple videos (different players / rounds)

Each video has its **own** `offset_sec` because recording start times differ.  
Always look up the per-round, per-player entry — never share offsets across rounds or players.

```python
import json

offsets = json.load(open("data/dust2/time_offset.json"))

# --- for each segment you want to render ---
entry   = offsets[match_id][player_id][round_name]   # e.g. "round_01"
offset  = entry["offset_sec"]
fps     = entry["video_fps"]

# Seek video to the game tick of interest
video_sec   = (tick / 64.0) + offset
video_frame = int(video_sec * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
```

```
Match A / Player X / Round 3  → offset =  2.14 s  (video started 2.14 s early)
Match A / Player Y / Round 3  → offset =  0.87 s  (different recorder, different offset)
Match A / Player X / Round 4  → offset =  3.02 s  (new recording, new offset)
```

> **Never** reuse one round's offset for another round or another player's video.

