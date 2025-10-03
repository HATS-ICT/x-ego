# Baseline Results Table Generator

This script generates formatted tables summarizing baseline experiment results from the `exp_baseline` output directory.

## Features

- Automatically scans all baseline experiment folders
- Parses folder names to extract task, task_form, and model information
- Reads test results from `test_analysis/*/test_results.json`
- Generates separate tables for:
  - **Coordinate Generation (cgen)**: Shows MAE, MSE, Chamfer Distance, Wasserstein Distance
  - **Multi-Label Classification (mlcls)**: Shows Hamming Loss/Accuracy, F1 scores
- Displays team-specific metrics (CT vs T)
- Saves results to a text file

## Usage

### Basic Usage (Recommended)

The script automatically reads from the `.env` file in the project root:

```bash
# Navigate to project root
cd /path/to/x-ego

# Run the script (reads OUTPUT_BASE_PATH from .env)
python scripts/print_table/baseline.py

# Or use the shell script
bash scripts/print_table/run_baseline_table.sh
```

The `.env` file should contain:
```bash
OUTPUT_BASE_PATH=C:\Users\wangy\projects\x-ego\output
# or on Linux/Discovery:
OUTPUT_BASE_PATH=/project2/ustun_1726/x-ego/output
```

### On Discovery Server

```bash
# Navigate to project root
cd /project2/ustun_1726/x-ego

# Run the script (reads from .env file)
python scripts/print_table/baseline.py
```

### With Custom Path

```bash
# Specify a custom output directory (overrides .env)
python scripts/print_table/baseline.py /path/to/output/exp_baseline
```

### Path Priority

The script determines paths in this order:
1. **Command line argument** (highest priority)
2. **`.env` file** (`OUTPUT_BASE_PATH` variable)
3. **Auto-detection** from current working directory (fallback)

## Output

The script generates a file named `baseline_results_table.txt` in the output directory containing:

1. Summary statistics for all coordinate generation experiments
2. Summary statistics for all multi-label classification experiments
3. Results grouped by task (en-forecast, en-nowcast, tm-forecast)
4. Metrics for each model (clip, dinov2, siglip, videomae, vivit, vjepa2)

### Example Output Format

```
COORDINATE GENERATION (CGEN) RESULTS
========================================
Task: en-forecast
Model        MAE      MSE    Chamfer    Wasser   CT-MAE    T-MAE  Samples
--------------------------------------------------------------------------------
clip       0.2022   0.0694    0.0937    0.0806   0.2034   0.2010      719
dinov2     0.2022   0.0694    0.0937    0.0806   0.2034   0.2010      719
...
```

## Folder Name Format

The script expects folder names in the format:
```
baseline-{task}-{task_form}-{model}-{timestamp}-{hash}
```

Example: `baseline-en-forecast-cgen-dinov2-251003-053126-16yy`

Where:
- **task**: `en-forecast`, `en-nowcast`, `tm-forecast`
- **task_form**: `cgen`, `mlcls`
- **model**: `clip`, `dinov2`, `siglip`, `videomae`, `vivit`, `vjepa2`

## Requirements

- Python 3.6+
- Standard library only (no additional packages required)

