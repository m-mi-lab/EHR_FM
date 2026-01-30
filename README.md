# EHR_FM

Electronic Health Record Foundation Model with Mixture of Experts (MoE) architecture.

## Evaluation

### Final Comprehensive Evaluation (All Tasks)

Evaluate the trained model on multiple clinical prediction tasks using ETHOS-style frequency-based trajectory method:

```bash
# Full evaluation on all 4 tasks (100 patients each, 20 trajectories per patient)
conda activate ehr_fm
cd /home/sud/temp_/ehr_stuff/EHR_FM
python scripts/eval_final.py \
  --config scripts/eval_final_config.yaml \
  --model /home/sud/temp_/ehr_stuff/EHR_FM/outputs/2026-01-16/08-49-53_model=gpt2_small_monolith,train=monolith_million,world_size=auto/best_model.pt \
  --output-dir eval_final_results \
  --patient-cache eval_final_results/patient_cache.json

# Quick test run (10 patients, 5 trajectories)
python scripts/eval_final.py \
  --config scripts/eval_final_config.yaml \
  --model $BEST_MODEL_PATH \
  --num-patients 10 \
  --num-trajectories 5 \
  --output-dir eval_test_run \
  --patient-cache eval_test_run/patient_cache.json

# Evaluate specific tasks only
python scripts/eval_final.py \
  --config scripts/eval_final_config.yaml \
  --model $BEST_MODEL_PATH \
  --tasks hosp_mortality icu_mortality \
  --output-dir eval_mortality_only
```

**Tasks evaluated:**
- `hosp_mortality` - Hospital Mortality Prediction (binary)
- `icu_mortality` - ICU Mortality Prediction (binary)
- `hosp_readmission` - Hospital 30-day Readmission Prediction (binary)
- `icu_readmission` - ICU Readmission Prediction (binary)
- `icu_los` - ICU Length of Stay Prediction in days (regression) *
- `drg_prediction` - DRG Classification, 771 classes (multiclass) *
- `sofa_prediction` - First-day SOFA Score Prediction (regression) *

\* Available but not in default config - uncomment in config to enable

**Output:**
- Per-task results JSON and trajectory data
- ROC curves comparing all tasks
- Task comparison bar plots (AUROC, AUPRC, Accuracy, F1)
- Evaluation summary JSON

## Setup

### Local Development Setup
```bash
conda deactivate 
conda create --name MEDS python=3.12 
conda activate MEDS 
pip install -e .[jupyter] 
```

### Docker Compose Setup

The project includes Docker Compose configuration for running training with MLflow tracking.

#### Prerequisites
- Docker and Docker Compose installed
- NVIDIA Docker runtime (`nvidia-docker2` or `nvidia-container-toolkit`) for GPU support
- NVIDIA drivers installed

#### Quick Start

1. **Configure environment variables** (optional):
   ```bash
   cp .env.example .env
   # Edit .env with your paths and settings
   ```

2. **Build and start services**:
   ```bash
   docker-compose up -d --build
   ```
   This starts the MLflow server and the training container (which stays running but doesn't start training automatically).

3. **Enter the training container**:
   ```bash
   docker exec -it ehr_fm_training bash
   ```

4. **Run training inside the container**:
   ```bash
   # Inside the container, you can run training with your desired arguments:
   python -m src.train
   # Or with custom configurations:
   python -m src.train model=gpt2_small_8exp train=8_exp world_size=1
   ```

5. **Access MLflow UI**:
   - Open browser to `http://localhost:5000` (or the port specified in `.env`)

6. **Access TensorBoard**:
   - Open browser to `http://localhost:6006` (or the port specified in `.env`)
   - TensorBoard automatically scans all subdirectories in `outputs/` for tensorboard logs

#### Alternative: Run container interactively from start

If you prefer to start the container in interactive mode immediately:

```bash
docker-compose run --rm training bash
```

This creates a new container, starts it with bash, and removes it when you exit.

#### Docker Compose Services

- **training**: Main training container with CUDA support
  - Uses `nvidia/cuda:11.6.2-base-ubuntu20.04` base image
  - GPU access via `nvidia` runtime
  - IPC host mode enabled for PyTorch DDP
  - Mounts data, outputs, and config directories
  - Container stays running for interactive use (does not auto-start training)
  - Enter with: `docker exec -it ehr_fm_training bash`

- **mlflow**: MLflow tracking server
  - Tracks experiments, metrics, and artifacts
  - Persists data to `mlflow_data/` directory
  - Accessible at `http://localhost:5000`

- **tensorboard**: TensorBoard visualization server
  - Visualizes training metrics and profiler traces
  - Reads from `outputs/` directory (scans all subdirectories)
  - Accessible at `http://localhost:6006`

#### Volume Mounts

- `DATA_DIR` (default: `./data`): Read-only data directory
- `OUTPUTS_DIR` (default: `./outputs`): Training outputs, checkpoints, logs
- `MLFLOW_DATA_DIR` (default: `./mlflow_data`): MLflow backend store and artifacts
- `EHR_STUFF_DIR` (default: `./ehr_stuff`): Additional data directory (based on previous setup)

#### Environment Variables

See `.env.example` for available configuration options:
- `CUDA_VISIBLE_DEVICES`: GPU selection (e.g., "0", "0,1", "all")
- `MLFLOW_PORT`: MLflow server port (default: 5000)
- `TENSORBOARD_PORT`: TensorBoard server port (default: 6006)
- `MLFLOW_EXPERIMENT_NAME`: MLflow experiment name (default: EHR_FM)
- `DATA_DIR`, `OUTPUTS_DIR`, `MLFLOW_DATA_DIR`, `EHR_STUFF_DIR`: Directory paths for volumes

## Data Preparation

```bash
python3 -m src.tokenizer.run_tokenization --config-name=tokenization_train
```

## Training

### Basic Training
```bash
python3 -m src.train
```

### Experiment Configuration

The training system uses Hydra configuration management. You can override default configurations:

```bash
# Run with different model and training configurations
python3 -m src.train model=gpt2_small_8exp train=8_exp world_size=1
python3 -m src.train model=gpt2_small_4exp train=4_exp world_size=auto
```

### Available Configurations

**Model Configurations:**
- `gpt2_small_4exp`: 4-expert MoE model
- `gpt2_small_8exp`: 8-expert MoE model

**Training Configurations:**
- `4_exp`: Training config optimized for 4-expert models
- `8_exp`: Training config optimized for 8-expert models (reduced batch size for memory efficiency)
- `default`: Standard training configuration

### Experiment Output

Experiments are automatically saved to timestamped directories:
```
outputs/YYYY-MM-DD/HH-MM-SS_model=MODEL,train=TRAIN,world_size=N/
├── train.log              # Training logs
├── tensorboard_logs/       # TensorBoard metrics
├── profiler_traces/        # PyTorch profiler traces
├── ckpt_*.pt              # Regular checkpoints
├── best_model.pt          # Best validation loss model
└── final_model.pt         # Final model
```

### MLflow Integration

When running with Docker Compose (or with `MLFLOW_TRACKING_URI` environment variable set), training automatically logs to MLflow:
- **Parameters**: All Hydra configuration parameters
- **Metrics**: Training/validation losses, learning rates, throughput, latency, GPU memory
- **Artifacts**: Model checkpoints (best_model.pt)
- **Tags**: Model type, world_size, dtype

Access the MLflow UI at `http://localhost:5000` to view experiments and compare runs.

### TensorBoard Integration

Training also logs to TensorBoard (runs in parallel with MLflow):
- **Metrics**: All training/validation metrics, learning rates, throughput, latency
- **Profiler Traces**: PyTorch profiler traces for performance analysis
- **Model Info**: Model type, expert configuration, loss weights

Access TensorBoard at `http://localhost:6006` to visualize training progress. TensorBoard automatically discovers all runs in the `outputs/` directory.

### Key Features

- **Mixture of Experts (MoE)**: Scalable architecture with expert routing
- **Memory Optimized**: Configurations tuned for different GPU memory constraints
- **Comprehensive Logging**: TensorBoard metrics, MLflow tracking, profiler traces, and detailed logs
- **Automatic Checkpointing**: Regular saves and best model preservation
- **MIMIC Dataset**: Trained on large-scale EHR data (~298M samples)
- **Docker Support**: Containerized training with GPU support and MLflow integration

## Mortality Prediction

The project includes a mortality prediction simulation tool that generates patient trajectories and evaluates model predictions.

### Configuration File

Edit `scripts/mortality_pred_config.yaml` to configure your simulation:

```yaml
data_dir: "/workspace/data/tokenized_datasets/mimic_train"  # Patient data directory
output_dir: "/workspace/outputs/mortality_prediction_results_JAN"  # Results output
num_patients: 100  # Total patients to simulate
target_death_patients: 50  # Target number who experienced death
num_trajectories_per_patient: 10  # Multiple runs per patient for statistical robustness
max_tokens: 500  # Safety limit (stops at death/discharge anyway)
temperature: 1.0  # Sampling temperature for generation
base_seed: 42  # Random seed for reproducibility

models:
  "Monolith":
    path: "/path/to/your/monolith/best_model.pt"
  "2-Expert":
    path: "/path/to/your/2expert/best_model.pt"
```

### Running Mortality Prediction

**Basic usage:**
```bash
cd scripts
python mortality_prediction.py --config mortality_pred_config.yaml
```

**Optional flags:**

```bash
# Cache patient data for faster re-runs
python mortality_prediction.py --config mortality_pred_config.yaml --patient-cache /path/to/cache

# Only keep complete trajectories
python mortality_prediction.py --config mortality_pred_config.yaml --discard-unfinished-trajectories
```

### What It Does

1. **Loads patient data** from the specified data directory
2. **Selects patients** based on `num_patients` and `target_death_patients` criteria
3. **Generates trajectories** by running each patient through the model(s) multiple times
4. **Evaluates predictions** by comparing generated outcomes to actual patient outcomes
5. **Saves results** including metrics, trajectories, and analysis to the output directory

### Output

Results are saved to the specified `output_dir` containing:
- Prediction metrics (accuracy, precision, recall, etc.)
- Generated trajectory samples
- Model comparison statistics
- Detailed logs