# X-EGO Cheatsheet

## Training

```bash
# Case 1: Contrastive learning (Stage 1)
python main.py --mode dev --task contrastive

# Case 2: Baseline downstream (off-the-shelf encoder)
python main.py --mode dev --task downstream task.task_id=self_location_0s

# Case 3: Downstream with pretrained contrastive encoder
python main.py --mode dev --task downstream task.task_id=self_location_0s model.stage1_checkpoint=/path/to/contrastive/checkpoint.ckpt
```

### Config Overrides

```bash
# Change model type
python main.py --mode dev --task contrastive model.encoder.video.model_type=dinov2

# Change batch size
python main.py --mode dev --task downstream task.task_id=self_location_0s data.batch_size=16

# Multiple overrides
python main.py --mode train --task downstream task.task_id=enemy_location_5s model.encoder.video.model_type=clip data.batch_size=8
```

### Modes
- `dev`: Quick test with small batches (batch_size=1, 10 steps, 2 epochs)
- `train`: Full training
- `test`: Test-only (requires `meta.resume_exp=<exp_name>`)

### Model Types
`siglip`, `dinov2`, `clip`, `vivit`, `videomae`, `vjepa2`

---

## Generate Labels

```bash
# contrastive pairs
python -m src.scripts.task_creator.create_contrastive_data

# task labels
python -m src.scripts.task_creator.create_all_labels 

# analyze label distribution
python -m src.scripts.task_creator.analyze_label_stats
```