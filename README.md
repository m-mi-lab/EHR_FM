# EHR_FM

Electronic Health Record Foundation Model with Mixture of Experts (MoE) architecture.

## Setup

```bash
conda deactivate 
conda create --name MEDS python=3.12 
conda activate MEDS 
pip install -e .[jupyter] 
```

## Data Preparation

```bash
python3 -m src.tokenizer.run_tokenization
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
python3 -m src.train model=gpt2_small_4exp train=4_exp world_size=1
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

### Key Features

- **Mixture of Experts (MoE)**: Scalable architecture with expert routing
- **Memory Optimized**: Configurations tuned for different GPU memory constraints
- **Comprehensive Logging**: TensorBoard metrics, profiler traces, and detailed logs
- **Automatic Checkpointing**: Regular saves and best model preservation
- **MIMIC Dataset**: Trained on large-scale EHR data (~298M samples)