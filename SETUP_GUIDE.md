# EHR_FM Complete Setup and Training Guide (Non-Docker)

This guide covers the full workflow for running EHR_FM without Docker: environment setup, MIMIC-IV data acquisition, data processing, tokenization, and training with remote MLflow integration.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Data Acquisition](#data-acquisition)
4. [MEDS ETL Pipeline](#meds-etl-pipeline)
5. [Tokenization](#tokenization)
6. [Remote MLflow Configuration](#remote-mlflow-configuration)
7. [Training](#training)
8. [Checkpoint Management](#checkpoint-management)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Linux or macOS (tested on Linux)
- NVIDIA GPU with CUDA support (for training)
- Conda or Miniconda installed
- PhysioNet credentials (for MIMIC-IV access)
- Access to a remote MLflow server

---

## Environment Setup

### Step 1: Create Conda Environment

```bash
# Deactivate any existing environment
conda deactivate

# Create new environment with Python 3.12
conda create --name MEDS python=3.12 -y
conda activate MEDS
```

### Step 2: Install EHR_FM Package

```bash
cd EHR_FM

# Install the package in editable mode with jupyter extras
pip install -e .[jupyter]
```

### Step 3: Install MEDS_transforms with Parallelism Support

```bash
# Install MEDS_transforms with local parallelism support
pip install "MEDS_transforms[local_parallelism]"

# Optional: For SLURM cluster support
# pip install "MEDS_transforms[slurm_parallelism]"

# Optional: For profiling ETL performance
# pip install hydra-profiler
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import mlflow; print(f'MLflow: {mlflow.__version__}')"
python -c "from MEDS_transforms.extract.utils import get_supported_fp; print('MEDS_transforms: OK')"
```

---

## Data Acquisition

### Step 1: Set Up Directory Structure

```bash
# Define your data directories (adjust paths as needed)
export MIMICIV_RAW_DIR="$HOME/data/mimic-iv-raw"
export MIMICIV_PRE_MEDS_DIR="$HOME/data/mimic-iv-pre-meds"
export MIMICIV_MEDS_DIR="$HOME/data/mimic-iv-meds"

# Create directories
mkdir -p "$MIMICIV_RAW_DIR"
mkdir -p "$MIMICIV_PRE_MEDS_DIR"
mkdir -p "$MIMICIV_MEDS_DIR"
```

### Step 2: Download MIMIC-IV from PhysioNet

You need credentialed access to PhysioNet. Visit https://physionet.org/content/mimiciv/2.2/ to:
1. Create a PhysioNet account
2. Complete required training (CITI)
3. Sign the data use agreement
4. Download the dataset

```bash
# Option A: Using wget with PhysioNet credentials
cd "$MIMICIV_RAW_DIR"
wget -r -N -c -np --user YOUR_USERNAME --ask-password \
    https://physionet.org/files/mimiciv/2.2/

# Option B: Using the PhysioNet CLI (if installed)
# physionet-download mimiciv 2.2 --destination "$MIMICIV_RAW_DIR"
```

After download, your directory should have:
```
$MIMICIV_RAW_DIR/
├── hosp/
│   ├── admissions.csv.gz
│   ├── diagnoses_icd.csv.gz
│   ├── drgcodes.csv.gz
│   ├── ...
└── icu/
    ├── icustays.csv.gz
    ├── chartevents.csv.gz
    ├── ...
```

### Step 3: Download MIMIC-IV Metadata Files

```bash
cd "$MIMICIV_RAW_DIR"

export MIMIC_URL="https://raw.githubusercontent.com/MIT-LCP/mimic-code/v2.4.0/mimic-iv/concepts/concept_map"

wget "$MIMIC_URL/d_labitems_to_loinc.csv"
wget "$MIMIC_URL/inputevents_to_rxnorm.csv"
wget "$MIMIC_URL/lab_itemid_to_loinc.csv"
wget "$MIMIC_URL/meas_chartevents_main.csv"
wget "$MIMIC_URL/meas_chartevents_value.csv"
wget "$MIMIC_URL/numerics-summary.csv"
wget "$MIMIC_URL/outputevents_to_loinc.csv"
wget "$MIMIC_URL/proc_datetimeevents.csv"
wget "$MIMIC_URL/proc_itemid.csv"
wget "$MIMIC_URL/waveforms-summary.csv"
```

---

## MEDS ETL Pipeline

The MEDS (Medical Event Data Standard) ETL pipeline converts raw MIMIC-IV data into a standardized format.

### Step 1: Navigate to MEDS Scripts Directory

```bash
cd EHR_FM/scripts/MEDS_transforms/mimic
```

### Step 2: Run the ETL Pipeline

```bash
# Set number of parallel workers (adjust based on your CPU cores)
export N_WORKERS=4

# Run the pipeline with automatic unzipping
./run.sh "$MIMICIV_RAW_DIR" "$MIMICIV_PRE_MEDS_DIR" "$MIMICIV_MEDS_DIR" do_unzip=true
```

**Note:** If your files are already unzipped, use `do_unzip=false`.

### Step 3: Verify ETL Output

After completion, verify the output structure:
```bash
ls -la "$MIMICIV_MEDS_DIR/data/"
# Should show: train/, test/ directories with .parquet files
```

The pipeline produces:
- `$MIMICIV_MEDS_DIR/data/train/` - Training split (~90%)
- `$MIMICIV_MEDS_DIR/data/test/` - Test split (~10%)

---

## Tokenization

Tokenization converts the MEDS parquet files into a format suitable for model training.

### Step 1: Configure Tokenization Paths

Edit `src/tokenizer/configs/tokenization.yaml`:

```yaml
# Update these paths to match your setup
input_dir: /path/to/mimic-iv-meds/data/train  # MEDS train directory
output_dir: /path/to/tokenized_datasets       # Output directory for tokenized data
```

Example with the paths from above:
```yaml
input_dir: ${oc.env:HOME}/data/mimic-iv-meds/data/train
output_dir: ${oc.env:HOME}/data/tokenized_datasets
```

### Step 2: Run Tokenization

```bash
cd EHR_FM

# Run tokenization
python3 -m src.tokenizer.run_tokenization
```

This process may take significant time depending on dataset size.

### Step 3: Verify Tokenization Output

```bash
ls -la /path/to/tokenized_datasets/mimic_train/
# Should contain:
# - vocab_t*.csv (vocabulary file)
# - *.safetensors (tokenized data shards)
# - static_data.pickle
# - interval_estimates.json
# - quantiles.json
# - code_counts.csv
```

### Step 4: Update Training Data Configuration

Edit `src/conf/data/mimic_data.yaml`:

```yaml
tokenized_dir: "/path/to/tokenized_datasets/mimic_train"
val_split_fraction: 0.05
```

---

## Remote MLflow Configuration

### Step 1: Set MLflow Environment Variables

Add these to your shell profile (`~/.bashrc` or `~/.zshrc`) or export before training:

```bash
# Required: Your remote MLflow tracking server URL
export MLFLOW_TRACKING_URI="https://your-mlflow-server.com"

# Optional: Experiment name (default: EHR_FM)
export MLFLOW_EXPERIMENT_NAME="EHR_FM"

# If your MLflow server requires authentication:
export MLFLOW_TRACKING_USERNAME="your_username"
export MLFLOW_TRACKING_PASSWORD="your_password"

# Alternative: Use token-based authentication
# export MLFLOW_TRACKING_TOKEN="your_token"
```

### Step 2: Verify MLflow Connection

```bash
python -c "
import mlflow
import os
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
print(f'Connected to: {mlflow.get_tracking_uri()}')
# List experiments to verify connection
for exp in mlflow.search_experiments():
    print(f'  Experiment: {exp.name}')
"
```

### MLflow Server Requirements

Your remote MLflow server should support:
- **Backend store**: For experiment/run metadata (PostgreSQL, MySQL, or SQLite)
- **Artifact store**: For model checkpoints (S3, GCS, Azure Blob, or local filesystem)

---

## Training

### Step 1: Verify GPU Availability

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### Step 2: Run Training

#### Basic Training (default configuration)

```bash
cd EHR_FM
python3 -m src.train
```

#### Training with Specific Configuration

```bash
# 4-expert MoE model
python3 -m src.train model=gpt2_small_4exp train=4_exp world_size=auto

# 8-expert MoE model (requires more GPU memory)
python3 -m src.train model=gpt2_small_8exp train=8_exp world_size=auto

# Single GPU training
python3 -m src.train model=gpt2_small_4exp train=4_exp world_size=1

# Multi-GPU training (specify exact count)
python3 -m src.train model=gpt2_small_4exp train=4_exp world_size=4
```

### Available Model Configurations

| Config | Description |
|--------|-------------|
| `gpt2_small_4exp` | 4-expert MoE, 768 embed dim, 6 layers |
| `gpt2_small_8exp` | 8-expert MoE, 768 embed dim, 6 layers |
| `gpt2_small_monolith` | Standard GPT (no MoE) |
| `gpt2_tiny` | Smaller model for testing |

### Available Training Configurations

| Config | Description |
|--------|-------------|
| `default` | Standard training, 600k iterations |
| `4_exp` | Optimized for 4-expert models, 200k iterations |
| `8_exp` | Memory-optimized for 8-expert models |
| `tiny` | Quick test runs |

### Step 3: Monitor Training

**TensorBoard (local):**
```bash
tensorboard --logdir outputs/
# Access at http://localhost:6006
```

**MLflow UI (remote):**
Access your MLflow server URL in a browser to view:
- Training/validation metrics
- Hyperparameters
- Model artifacts (checkpoints)

---

## Checkpoint Management

### Checkpoint Locations

Training saves checkpoints to:
```
outputs/YYYY-MM-DD/HH-MM-SS_model=MODEL,train=TRAIN,world_size=N/
├── train.log                 # Training logs
├── tensorboard_logs/         # TensorBoard metrics
├── profiler_traces/          # PyTorch profiler traces
├── ckpt_ITER.pt             # Periodic checkpoints
├── best_model.pt            # Best validation loss model
└── final_model.pt           # Final model after training
```

### Checkpoint Contents

Each checkpoint contains:

```python
{
    'iter_num': int,                    # Current iteration
    'model': dict,                      # Model state dict
    'optimizer': dict,                  # Optimizer state dict
    'scaler': dict,                     # Mixed precision scaler state
    'best_val_loss': float,             # Best validation loss so far
    'model_configs': dict,              # Model configuration
    'vocab_stoi': dict,                 # Vocabulary (string to index)
    'hydra_config_full': dict,          # Complete Hydra configuration
}
```

### MLflow Artifacts

Best model checkpoints are automatically logged to MLflow:
- Artifact path: `checkpoints/best_model.pt`
- Access via MLflow UI or API

### Loading a Checkpoint

```python
import torch
from pathlib import Path

# Load checkpoint
ckpt_path = Path("outputs/2024-01-01/12-00-00_model=gpt2_small_4exp,train=4_exp,world_size=1/best_model.pt")
checkpoint = torch.load(ckpt_path, map_location="cuda:0")

# Access components
model_state = checkpoint['model']
vocab = checkpoint['vocab_stoi']
config = checkpoint['hydra_config_full']

print(f"Loaded from iteration: {checkpoint['iter_num']}")
print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
```

### Resume Training from Checkpoint

```bash
python3 -m src.train \
    model=gpt2_small_4exp \
    train=4_exp \
    train.resume_from_checkpoint="/path/to/checkpoint.pt"
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

Reduce batch size or gradient accumulation steps:
```bash
python3 -m src.train model=gpt2_small_4exp train=4_exp train.batch_size=8 train.gradient_accumulation_steps=2
```

#### 2. MLflow Connection Failed

Verify connectivity:
```bash
curl -v "$MLFLOW_TRACKING_URI/api/2.0/mlflow/experiments/list"
```

Check firewall/network settings and authentication credentials.

#### 3. Tokenization Fails

- Ensure MEDS ETL completed successfully
- Check input_dir path points to directory with `.parquet` files
- Verify sufficient disk space for output

#### 4. MEDS ETL Hangs

- Reduce N_WORKERS if running out of memory
- Check for corrupted source files
- Ensure all required CSV files are present

#### 5. Multi-GPU Training Fails

Check NCCL backend:
```bash
python -c "import torch.distributed as dist; print(dist.is_nccl_available())"
```

Try gloo backend as fallback:
```bash
export NCCL_DEBUG=INFO
python3 -m src.train world_size=2
```

### Getting Help

- Check `train.log` in the output directory for detailed error messages
- Review TensorBoard logs for training anomalies
- Consult the original README.md for additional context

---

## Quick Reference

### Environment Variables Summary

```bash
# Data directories
export MIMICIV_RAW_DIR="/path/to/mimic-iv-raw"
export MIMICIV_PRE_MEDS_DIR="/path/to/mimic-iv-pre-meds"
export MIMICIV_MEDS_DIR="/path/to/mimic-iv-meds"

# MLflow (required for remote tracking)
export MLFLOW_TRACKING_URI="https://your-mlflow-server.com"
export MLFLOW_EXPERIMENT_NAME="EHR_FM"
export MLFLOW_TRACKING_USERNAME="your_username"  # if needed
export MLFLOW_TRACKING_PASSWORD="your_password"  # if needed

# ETL parallelism
export N_WORKERS=4
```

### Complete Training Command

```bash
# Activate environment
conda activate MEDS

# Set MLflow
export MLFLOW_TRACKING_URI="https://your-mlflow-server.com"
export MLFLOW_EXPERIMENT_NAME="EHR_FM"

# Run training
cd EHR_FM
python3 -m src.train model=gpt2_small_4exp train=4_exp world_size=auto
```

