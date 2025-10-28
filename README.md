# X-Ego: Cross-Egocentric Multi-Agent Video Understanding in Counter-Strike 2

[![Paper](https://img.shields.io/badge/arXiv-2510.19150-b31b1b.svg)](https://arxiv.org/abs/2510.19150)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-blue.svg)](https://huggingface.co/datasets/wangyz1999/X-EGO-CS)

**X-Ego-CS** is a large-scale multi-agent gameplay dataset for **cross-egocentric video understanding** in *Counter-Strike 2*.  
It provides synchronized **first-person (egocentric)** video streams from **10 players (5v5)** across **45 professional-level matches**, enabling research on **tactical reasoning**, **situational awareness**, and **multi-agent coordination**.

<video controls>
  <source src="https://huggingface.co/datasets/wangyz1999/X-EGO-CS/resolve/main/multi-ego-sync-demo-pistol.mp4" type="video/mp4">
</video>

---

## Quick Start

We use [`uv`](https://docs.astral.sh/uv) for package management.

After installing `uv`, run
```bash
uv sync
```
to set up venv and install required dependencies

### 🚀 Training

```
# Train enemy location nowcast task
python main.py --mode train --task enemy_location_nowcast

# Train teammate location forecast task
python main.py --mode train --task teammate_location_forecast
```

All configuration files are located in the `configs/` directory:
- `configs/global.yaml` — global default configuration
- `configs/train/<task>.yaml` — task-specific training configuration
- `configs/dev/<task>.yaml` — lightweight debug configuration

Configuration priority (higher overrides lower):
```
Command line > dev  >  train  >  global
```

All configs in `.yaml` files can be overwritten in the command line, for example

```
python main.py --mode train --task enemy_location_nowcast training.max_epochs=20 data.batch_size=8 meta.seed=123
```

## Data File Structure
```
data/
├── demos/                       # Raw .dem files (by match)
│   └── <match_id>.dem
├── labels/                      # Global label datasets
│   ├── enemy_location_nowcast_s1s_l5s.csv
│   └── teammate_location_nowcast_s1s_l5s.csv
├── metadata/                    # Match / round metadata
│   ├── matches/
│   │   └── <match_id>.json
│   └── rounds/
│       └── <match_id>/
│           └── round_<nn>.json
├── trajectories/                # Player movement trajectories
│   └── <match_id>/
│       └── <player_id>/
│           ├── round_<nn>.csv
│           └── ...
└── videos/                      # Player POV recordings
    └── <match_id>/
        └── <player_id>/
            ├── round_<nn>.mp4
            └── ...
```


## 🧩 Citation
If you use this repo, please cite our paper:
```bibtex
@article{wang2025x,
  title={X-Ego: Acquiring Team-Level Tactical Situational Awareness via Cross-Egocentric Contrastive Video Representation Learning},
  author={Wang, Yunzhe and Hans, Soham and Ustun, Volkan},
  journal={arXiv preprint arXiv:2510.19150},
  year={2025}
}
```
