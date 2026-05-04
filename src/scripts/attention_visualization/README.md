# Attention Visualization Scripts

## Static Attention Grids

Defaults to the current SigLIP2 Mirage checkpoint:
`main_contra_with_accu-siglip2-mirage-ui-all-260427-080317-eixo`.

```bash
python -m src.scripts.attention_visualization.visualize_attention
```

Outputs are written to `artifacts/attention_visualization_v3/`.

Useful options:
- `--heads mean,all` renders the mean attention map plus one output per head.
- `--heads 0,1,2` renders only selected heads.
- `--attention-source spatial|temporal|both` chooses vision self-attention, the new temporal transformer attention, or both.
- `--epochs 1 5 20 last` renders multiple checkpoints into separate subfolders.
- `--layout compact` writes one image per sample with heads as rows and teammates x frames as columns.
- `--layout per-teammate` restores the older one-file-per-teammate/head layout.
- `--frame-indices 0,5,10,15,19` limits compact columns; by default compact uses all frames.
- `--selected` uses the fixed paper sample/teammate list.

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
