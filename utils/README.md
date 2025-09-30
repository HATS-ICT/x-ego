# Utils Module Organization

This directory contains utility functions organized by their purpose. This refactoring improves code organization and makes it easier to find and maintain utility functions.

## Module Overview

### Core Utilities

#### `env_utils.py`
**Purpose**: Environment variable management and path configuration
- `get_src_base_path()` - Get source base path from environment
- `get_data_base_path()` - Get data directory path
- `get_output_base_path()` - Get output directory path
- `print_env_info()` - Print environment configuration for debugging

#### `config_utils.py`
**Purpose**: Configuration file handling
- `load_config()` - Load YAML configuration using OmegaConf
- `validate_config()` - Validate configuration structure
- `apply_config_overrides()` - Apply command-line overrides to config

#### `experiment_utils.py`
**Purpose**: Experiment lifecycle management
- `create_experiment_dir()` - Create experiment directory structure
- `save_hyperparameters()` - Save config to experiment directory
- `find_experiment_directory()` - Locate experiment by name or path
- `load_experiment_config()` - Load config from experiment
- `find_checkpoint()` - Find best/last checkpoint in experiment
- `find_resume_checkpoint()` - Find checkpoint for resuming training
- `setup_resume_config()` - Configure for training resumption
- `save_evaluation_results()` - Save evaluation results as JSON

### Model & Training Utilities

#### `model_utils.py`
**Purpose**: Model components and loading utilities
- `SelectIndex` - Neural network module for sequence indexing
- `load_model_from_checkpoint()` - Load trained model from checkpoint

#### `training_utils.py`
**Purpose**: PyTorch Lightning training setup
- `setup_callbacks()` - Configure Lightning callbacks (checkpointing, early stopping)
- `setup_logger()` - Configure WandB logger
- `debug_batch_plot()` - Visualize training batches for debugging
- Helper functions: `_to_numpy()`, `_stack_20_frames()`

### Metrics & Evaluation

#### `metric_utils.py`
**Purpose**: Metric calculations and loss functions
- **NLP Metrics** (lazy-loaded):
  - `compute_commentary_metrics()` - BLEU, METEOR, ROUGE-L
  - `_initialize_nlp_metrics()` - Initialize NLP metric libraries
- **Location Prediction Metrics**:
  - `kl_divergence_histogram()` - KL divergence for distributions
  - `multinomial_loss()` - Multinomial loss for histograms
  - `exact_match_accuracy()` - Exact match accuracy
  - `l1_count_error()` - L1 error for count predictions
  - `chamfer_distance_batch()` - Chamfer distance between point clouds

### Visualization & Serialization

#### `plot_utils.py`
**Purpose**: Plotting and visualization
- `create_prediction_plots()` - Create regression or classification plots
- `create_regression_plots()` - Scatter plots for coordinate predictions
- `create_classification_plots()` - Histogram analysis plots
- `heatmap()` - Create KDE heatmap on CS:GO maps
- `create_prediction_heatmaps()` - Single sample heatmaps
- `create_prediction_heatmaps_grid()` - Grid of heatmaps for comparison

#### `serialization_utils.py`
**Purpose**: Data type conversion for logging and storage
- `json_serializable()` - Convert numpy/OmegaConf to JSON-compatible types
- `convert_numpy_to_python()` - Recursively convert numpy arrays to Python lists

## Import Examples

```python
# Configuration
from utils.config_utils import load_config, validate_config

# Experiment management
from utils.experiment_utils import create_experiment_dir, find_checkpoint

# Training
from utils.training_utils import setup_callbacks, setup_logger

# Metrics
from utils.metric_utils import chamfer_distance_batch, multinomial_loss

# Model utilities
from utils.model_utils import load_model_from_checkpoint

# Plotting
from utils.plot_utils import create_prediction_heatmaps_grid

# Serialization
from utils.serialization_utils import json_serializable
```

## Migration Guide

If you're updating old code, here's how the imports have changed:

| Old Import | New Import |
|------------|-----------|
| `from models.utils import load_config` | `from utils.config_utils import load_config` |
| `from models.utils import create_experiment_dir` | `from utils.experiment_utils import create_experiment_dir` |
| `from models.utils import setup_callbacks` | `from utils.training_utils import setup_callbacks` |
| `from models.utils import json_serializable` | `from utils.serialization_utils import json_serializable` |
| `from models.metric_utils import *` | `from utils.metric_utils import *` |
| `from models.plot_utils import *` | `from utils.plot_utils import *` |
| `from env_utils import *` | `from utils.env_utils import *` |

## Design Principles

1. **Single Responsibility**: Each module focuses on one domain
2. **Clear Naming**: Module names clearly indicate their purpose
3. **Minimal Dependencies**: Modules import only what they need
4. **Ease of Discovery**: Related functions are grouped together
5. **Import Clarity**: Explicit imports make dependencies clear
