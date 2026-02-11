# EHR_FM

Electronic Health Record Foundation Model with Mixture of Experts (MoE) architecture.

## 🚀 Quick Start 

```bash
# 1. Install dependencies
conda create --name ehr_fm python=3.12
conda activate ehr_fm
pip install -e .[jupyter]

# 2. Configure environment (.env file)
cp env.example .env
# Edit .env and fill in:
#   - PHYSIONET_USERNAME and PHYSIONET_PASSWORD (required for --download)
#   - MLFLOW_TRACKING_URI (e.g., http://localhost:5000 or remote server)
#   - MODEL_CONFIG and TRAIN_CONFIG (defaults work fine)
#   - Optional: S3/Tigris credentials for artifact storage

# 3. Run complete pipeline (download, ETL, tokenize, train)
./run_pipeline.sh --download
```

**That's it!** The script will:
- ✅ Download MIMIC-IV v2.2 from PhysioNet (if `--download` flag used)
- ✅ Run MEDS ETL pipeline to convert raw data to MEDS format
- ✅ Tokenize patient timelines and build vocabulary
- ✅ Train the EHR foundation model with specified configuration
- ✅ Log all metrics to MLflow and TensorBoard

**If you already have MIMIC-IV data:**
```bash
# Skip download, point to existing data
# Set MIMICIV_RAW_DIR=/path/to/mimic-iv in .env, then:
./run_pipeline.sh
```

**Training-only (skip data prep):**
```bash
# If data is already tokenized at data/tokenized/
./run_pipeline.sh --skip-etl --skip-tokenization
```

**Output locations:**
- Training checkpoints: `outputs/YYYY-MM-DD/HH-MM-SS.../`
- MLflow experiments: Visit `MLFLOW_TRACKING_URI` in browser
- TensorBoard logs: `outputs/.../tensorboard_logs/`

### Detailed Pipeline Options

The `run_pipeline.sh` script supports fine-grained control over each step:

```bash
# Available flags:
./run_pipeline.sh --download              # Download MIMIC-IV from PhysioNet
./run_pipeline.sh --skip-etl              # Skip MEDS ETL (use existing MEDS data)
./run_pipeline.sh --skip-tokenization     # Skip tokenization (use existing tokenized data)
./run_pipeline.sh --skip-training         # Skip training (only prep data)
./run_pipeline.sh --dry-run               # Print commands without executing
./run_pipeline.sh --help                  # Show help message
```

**Environment variable overrides (via .env or CLI):**
```bash
# Override vars via CLI (takes precedence over .env)
MODEL_CONFIG=gpt2_small_8exp WORLD_SIZE=2 ./run_pipeline.sh

# Or edit .env file:
MODEL_CONFIG=gpt2_small_4exp      # Model architecture
TRAIN_CONFIG=4_exp                # Training configuration
WORLD_SIZE=auto                   # GPUs (auto-detect or specify number)
N_WORKERS=4                       # ETL parallelism
MLFLOW_TRACKING_URI=http://...    # MLflow server
```

**All configurable variables** (see `env.example` for full list):
- `PHYSIONET_USERNAME`, `PHYSIONET_PASSWORD` - PhysioNet credentials
- `MIMICIV_RAW_DIR`, `MIMICIV_MEDS_DIR`, `TOKENIZED_DIR` - Data paths
- `MODEL_CONFIG` - Model architecture (`gpt2_small_4exp`, `gpt2_small_8exp`, `gpt2_small_monolith`)
- `TRAIN_CONFIG` - Training preset (`4_exp`, `8_exp`, `monolith_million`, `tiny`)
- `WORLD_SIZE` - Number of GPUs (`auto`, `1`, `2`)
- `MLFLOW_TRACKING_URI` - MLflow server URL
- `MLFLOW_EXPERIMENT_NAME` - Experiment name (default: `EHR_FM`)
- `AWS_ENDPOINT_URL_S3`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` - S3/Tigris artifact storage
- `N_WORKERS` - Parallel workers for ETL
- `VAL_SPLIT_FRACTION` - Validation split (default: 0.05)
- `RESUME_CHECKPOINT` - Path to resume training from checkpoint

### Monitoring Training Progress

**MLflow UI** (comprehensive experiment tracking):
```bash
# If using local MLflow server
mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri ./mlflow_data/mlruns

# Then visit http://localhost:5000 in browser
```

**TensorBoard** (real-time metrics):
```bash
tensorboard --logdir outputs/ --port 6006 --bind_all

# Then visit http://localhost:6006 in browser
```

**Training Logs** (live progress):
```bash
# Find latest training run
ls -ltr outputs/

# Tail logs
tail -f outputs/YYYY-MM-DD/HH-MM-SS_.../train.log
```

### Expected Timeline

On 2x RTX 4090 (49GB VRAM total):
- **Download MIMIC-IV**: ~2-4 hours (depends on network)
- **MEDS ETL**: ~30-60 minutes (with N_WORKERS=4)
- **Tokenization**: ~20-30 minutes
- **Training (4-expert, ~298M samples)**: 
  - ~1.5-2 hours per epoch
  - Total: ~10-15 hours for convergence (5-8 epochs)

### Expected Timeline

On 2x RTX 4090 (49GB VRAM total):
- **Download MIMIC-IV**: ~2-4 hours (depends on network)
- **MEDS ETL**: ~30-60 minutes (with N_WORKERS=4)
- **Tokenization**: ~20-30 minutes
- **Training (4-expert, ~298M samples)**: 
  - ~1.5-2 hours per epoch
  - Total: ~10-15 hours for convergence (5-8 epochs)

### Troubleshooting

**Problem: PhysioNet download fails with 401 Unauthorized**
- Verify your PhysioNet account has credentialed access to MIMIC-IV v2.2
- Complete required training: https://physionet.org/about/citi-course/
- Check username/password in `.env` (no quotes needed)

**Problem: MEDS ETL fails with "file not found"**
- Verify `MIMICIV_RAW_DIR` contains `hosp/` and `icu/` subdirectories
- Check file structure: `$MIMICIV_RAW_DIR/hosp/admissions.csv` should exist
- If files are `.gz` compressed, set `DO_UNZIP=true` in `.env`

**Problem: Training hangs or doesn't start**
- Check GPU availability: `nvidia-smi` (should show 2 GPUs)
- Verify CUDA setup: `python -c "import torch; print(torch.cuda.is_available())"`
- Try single-GPU mode: `WORLD_SIZE=1 ./run_pipeline.sh --skip-etl --skip-tokenization`
- Check MASTER_PORT is available (default: 23556): `netstat -tuln | grep 23556`

**Problem: MLflow connection errors**
- Verify `MLFLOW_TRACKING_URI` is reachable: `curl $MLFLOW_TRACKING_URI`
- Start local MLflow server: `mlflow ui --host 0.0.0.0 --port 5000`
- For remote servers, check firewall/VPN settings

**Problem: Out of memory (OOM) during training**
- Reduce batch size in training config (e.g., use `TRAIN_CONFIG=8_exp` for smaller batches)
- Use single-GPU mode: `WORLD_SIZE=1`
- Try smaller model: `MODEL_CONFIG=gpt2_small_monolith`
- Check GPU memory usage: `nvidia-smi -l 1` (should not exceed 24GB per GPU)

**Problem: Tokenization fails or produces empty files**
- Verify MEDS data structure: `ls $MIMICIV_MEDS_DIR/data/train/` should have `.parquet` files
- Check tokenization logs for errors
- Ensure sufficient disk space (tokenization writes large files)

**Problem: Training loss not decreasing**
- ✅ **CRITICAL**: As of Feb 6, 2026, batch_size calculation bug has been fixed (line 195 in `src/train.py`)
- Verify learning rate: should be ~6e-4 (NOT 1e-2)
- Check validation loss: should decrease within first few steps
- Inspect MLflow/TensorBoard: look for gradient flow, check aux losses

**Problem: Checkpoint not loading (vocab mismatch)**
- Ensure checkpoint was saved from same vocabulary
- Check `vocab_stoi` key exists in checkpoint dict
- Verify vocabulary size is padded to multiple of 64

**Problem: DDP (multi-GPU) hangs at initialization**
- Check `MASTER_ADDR` and `MASTER_PORT` environment variables
- Ensure both GPUs are visible: `CUDA_VISIBLE_DEVICES=0,1 nvidia-smi`
- Try setting explicitly: `MASTER_ADDR=localhost MASTER_PORT=23556 ./run_pipeline.sh`
- Verify network connectivity between processes (rarely an issue on single machine)

**Getting help:**
- Check existing logs: `outputs/.../train.log`
- Review MLflow experiment page for detailed metrics
- See `.github/copilot-instructions.md` for architecture details

---

## Evaluation

### Direct Probability Inference (Fast, Deterministic)

Evaluate the trained model on multiple clinical prediction tasks using direct probability extraction:

```bash
# Run inference on all 5 tasks with 20 repetitions each
# Uses default config: scripts/eval_final_config.yaml
conda activate ehr_fm
cd /home/sud/temp_/ehr_stuff/EHR_FM
sbatch scripts/run_infer.sh

# Or run locally (no SLURM)
bash scripts/run_infer.sh

# Use custom config
sbatch scripts/run_infer.sh path/to/custom_config.yaml
bash scripts/run_infer.sh scripts/eval_final_config.yaml
```

**Tasks evaluated:**
- `hosp_mortality` - Hospital Mortality Prediction (binary classification)
- `icu_mortality` - ICU Mortality Prediction (binary classification)
- `hosp_readmission` - Hospital 30-day Readmission Prediction (binary classification)
- `icu_readmission` - ICU Readmission Prediction (binary classification)
- `icu_los` - ICU Length of Stay Prediction in days (regression)

**Method:**
- Single forward pass per patient (no trajectory sampling)
- Direct softmax probability extraction
- 10x faster than trajectory-based methods
- Deterministic results (no sampling variance)

**Output:**
```
infer_results/
├── hosp_mortality/
│   ├── hosp_mortality_rep1.json ... rep20.json  (20 repetitions)
│   ├── hosp_mortality_rep1_plot.png ... rep20_plot.png  (bar charts)
│   └── hosp_mortality_aggregated.json  (mean ± std across reps)
├── icu_mortality/
│   └── ...
├── hosp_readmission/
│   └── ...
├── icu_readmission/
│   └── ...
└── icu_los/
    └── ...
```

**Metrics:**
- Binary tasks: AUROC, AUPRC, Accuracy
- Regression (ICU LOS): MAE, RMSE, Pearson correlation


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

---

### For Inference/Evaluation

```bash
# Activate environment
conda activate ehr_fm

# Run evaluation on all tasks
bash scripts/run_infer.sh

# Or single task inference
python scripts/infer.py --test hosp_mortality --model outputs/.../best_model.pt --data-dir data/tokenized/mimic_train --output infer_results
```

---
## Support & Documentation

- **Quick help**: `./run_pipeline.sh --help` or `./verify_setup.sh`
- **Architecture details**: [.github/copilot-instructions.md](.github/copilot-instructions.md)
- **Configuration reference**: `src/conf/` directory
- **Issues & bugs**: Check training logs and MLflow experiment pages

## License

See LICENSE file for details.
