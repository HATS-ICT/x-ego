# Attention Visualization Scripts

## Single Video (3-panel)

Renders: **Original | Model + CECL | Model**

```bash
python -m src.scripts.attention_visualization.render_attention_video \
    --video_path "data/video_306x306_4fps/MATCH_ID/PLAYER_ID/round_X.mp4" \
    --model_type dinov2
```

Options: `--model_type {dinov2, siglip2, clip}`, `--output_path`, `--no_titles`

## Team Synchronized (3×5 grid)

Renders all 5 team members in sync with attention overlays.

```bash
python -m src.scripts.attention_visualization.render_team_attention_video \
    --match_id "1-2b61eddf-3ab3-47ff-ac3e-3d730458667b-1-1" \
    --round_num 5 \
    --team_number 0 \
    --model_type dinov2
```

Options: `--team_number {0, 1}`, `--model_type {dinov2, siglip2, clip}`, `--output_path`

## Batch Processing

Processes 20 random videos for each model type.

```bash
python -m src.scripts.attention_visualization.batch_render_attention_videos
```

Options: `--num_videos 20`, `--models dinov2 siglip2 clip`
