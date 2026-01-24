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
python main.py --mode dev --task contrastive model.encoder.model_type=dinov2

# Change batch size
python main.py --mode dev --task downstream task.task_id=self_location_0s data.batch_size=16

# Multiple overrides
python main.py --mode train --task downstream task.task_id=enemy_location_5s model.encoder.model_type=clip data.batch_size=8
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


## Test all model and training setup and all tasks

```
# Test all model setups (3 settings × 6 models)
python test_all_model_setup.py

# Test all tasks (35 tasks with siglip baseline)
python test_all_tasks.py
```


Run standalong test model on existing experiment
```
python main.py --mode test --task downstream meta.resume_exp=probe-siglip2-teammate_aliveCount-none-260122-214354-ky6j
```