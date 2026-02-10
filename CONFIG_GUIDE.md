# EHR_FM Configuration Guide

**Last Updated**: February 6, 2026  
**Status**: All configs aligned with ETHOS ARES hyperparameters

---

## 📋 Summary of Changes

### Fixes Applied (2026-02-06)
1. ✅ **Critical batch_size_per_gpu bug fixed** in `src/train.py` line 195
2. ✅ **All training configs aligned** with ETHOS hyperparameters
3. ✅ **Removed redundant/experimental configs**
4. ✅ **Standardized dtype to bfloat16** for better mixed precision
5. ✅ **Increased warmup_iters to 5000** for training stability
6. ✅ **Fixed gradient accumulation steps** for proper effective batch sizes

### Deleted Configs (Unnecessary/Redundant)
- ❌ `monolith_million_fixed.yaml` (redundant after fixing monolith_million.yaml)
- ❌ `gpt2_small_monolith_tuned.yaml` (experimental, non-standard)
- ❌ `8_exp.yaml` (8 experts with top_k=4 too large, per debugging report)
- ❌ `gpt2_small_8exp.yaml` (corresponding model config)
- ❌ `gpt2_small.yaml` (redundant with gpt2_small_monolith.yaml)

---

## 🎯 Current Configuration Files

### Training Configs (`src/conf/train/`)

| Config | Purpose | Max Iters | Batch Size | Grad Accum | Effective Batch (2 GPUs) |
|--------|---------|-----------|------------|------------|--------------------------|
| `monolith_million.yaml` | **Production monolith** (1M iters) | 1,000,000 | 32 | 16 | 1024 |
| `monolith.yaml` | Shorter monolith run | 60,000 | 32 | 16 | 1024 |
| `4_exp.yaml` | 4-expert MoE | 200,000 | 16 | 16 | 512 |
| `2_exp.yaml` | 2-expert MoE | 200,000 | 16 | 16 | 512 |
| `default.yaml` | Default (long MoE) | 600,000 | 16 | 16 | 512 |
| `tiny.yaml` | Quick testing | 10,000 | 16 | 4 | 128 |

### Model Configs (`src/conf/model/`)

| Config | Type | Layers | Embed Dim | Heads | Dropout | Experts | Top-K |
|--------|------|--------|-----------|-------|---------|---------|-------|
| `gpt2_small_monolith.yaml` | **Monolith** | 6 | 768 | 12 | 0.3 | - | - |
| `gpt2_small_4exp.yaml` | MoE | 6 | 768 | 12 | 0.3 | 4 | 2 |
| `gpt2_small_2exp.yaml` | MoE | 6 | 768 | 12 | 0.3 | 2 | 2 |
| `gpt2_tiny.yaml` | Tiny (testing) | 1 | 64 | 4 | 0.0 | - | - |

---

## 🚀 Usage Guide

### 1. **Start with Monolith (RECOMMENDED)**
```bash
# Production run (1M iterations, ~1 week on 2x RTX 4090)
python -m src.train model=gpt2_small_monolith train=monolith_million world_size=2

# Shorter run (60k iterations, ~1 day)
python -m src.train model=gpt2_small_monolith train=monolith world_size=2

# Single GPU
python -m src.train model=gpt2_small_monolith train=monolith_million world_size=1
```

### 2. **Quick Testing**
```bash
# Fast iteration for debugging (10k iters)
python -m src.train model=gpt2_tiny train=tiny world_size=1
```

### 3. **MoE Experiments (after monolith baseline)**
```bash
# 2-expert MoE
python -m src.train model=gpt2_small_2exp train=2_exp world_size=2

# 4-expert MoE (recommended)
python -m src.train model=gpt2_small_4exp train=4_exp world_size=2
```

---

## 📊 Key Hyperparameters (All Configs Aligned)

### Learning Rate Schedule
- **Initial LR**: `6e-4` (matches ETHOS)
- **Min LR**: `6e-5` (10% of max)
- **Warmup**: 5000 iterations (increased for stability)
- **LR Decay**: Cosine decay over full training

### Optimizer (AdamW)
- **Weight Decay**: 0.1
- **Beta1**: 0.9
- **Beta2**: 0.95
- **Grad Clip**: 1.0

### Data Type & Precision
- **dtype**: `bfloat16` (better for mixed precision than float16)
- Automatic mixed precision (AMP) enabled

### Batch Sizes (Per-GPU)

**Monolith configs:**
- Per-GPU batch: 32
- Gradient accumulation: 16
- Effective per GPU: 32 × 16 = 512
- Total (2 GPUs): 1024

**MoE configs:**
- Per-GPU batch: 16 (smaller due to MoE memory overhead)
- Gradient accumulation: 16
- Effective per GPU: 16 × 16 = 256
- Total (2 GPUs): 512

---

## ⚠️ Important Notes

### MoE Training Considerations
1. **Start with monolith** to verify data pipeline and base model
2. **Use 4 experts, not 8**: 8 experts with top_k=4 is too large per debugging report
3. **Monitor expert utilization**: Check for load balancing via aux_loss logs
4. **Watch for capacity overflow**: Reduce batch size if seeing overflow warnings

### Hardware Context (Your System)
- **GPUs**: 2× NVIDIA RTX 4090 (24GB VRAM each)
- **Optimal world_size**: 2 (uses both GPUs)
- **Memory headroom**: ~10-15GB free per GPU with current configs
- **Expected GPU utilization**: 80-95% during training

### Checkpoint Management
- **Checkpoints saved**: Every 1000 iterations + best validation loss
- **Location**: `outputs/YYYY-MM-DD/HH-MM-SS_<override_dirname>/`
- **Contains**: model state, optimizer state, vocab, hydra config

---

## 🔍 Quick Reference

### Override Examples
```bash
# Change learning rate
python -m src.train train.lr=3e-4

# Change batch size
python -m src.train train.batch_size=64

# Change number of experts
python -m src.train model=gpt2_small_4exp model.n_experts=4

# Resume from checkpoint
python -m src.train train.resume_from_checkpoint=/path/to/checkpoint.pt

# Different seed
python -m src.train seed=123
```

### Expected Training Metrics

**Monolith (monolith_million.yaml on 2x RTX 4090):**
- Initial loss: ~10.0
- After 1000 iters: ~6-7
- After 20k iters: ~4-5
- Final (1M iters): ~2.5-3.5
- Training time: ~5-7 days
- AUROC (mortality): 0.70-0.75 (ETHOS baseline)

**MoE (4_exp.yaml on 2x RTX 4090):**
- Initial loss: ~10.0 (similar to monolith)
- Convergence: Slightly slower than monolith
- Final loss: Potentially better than monolith
- Training time: ~2-3 days (200k iters)
- AUROC: Target 0.73-0.78 (improvement over monolith)

---

## 📝 Config File Locations

```
src/conf/
├── config.yaml                    # Main config (defaults to monolith_million)
├── train/
│   ├── monolith_million.yaml     # ⭐ Production monolith (1M iters)
│   ├── monolith.yaml              # Shorter monolith (60k iters)
│   ├── 4_exp.yaml                 # 4-expert MoE
│   ├── 2_exp.yaml                 # 2-expert MoE
│   ├── default.yaml               # Default (long MoE run)
│   └── tiny.yaml                  # Quick testing
└── model/
    ├── gpt2_small_monolith.yaml  # ⭐ Standard monolith model
    ├── gpt2_small_4exp.yaml       # 4-expert MoE model
    ├── gpt2_small_2exp.yaml       # 2-expert MoE model
    └── gpt2_tiny.yaml             # Tiny model (testing)
```

---

## 🐛 Troubleshooting

### "Loss not decreasing"
- ✅ **FIXED**: batch_size_per_gpu bug resolved
- ✅ **FIXED**: Learning rate aligned to 6e-4
- Check logs for proper effective batch size (should be 1024 for monolith, 512 for MoE)

### "Out of memory"
- Reduce `batch_size` (try 16 for monolith, 8 for MoE)
- Reduce `gradient_accumulation_steps` proportionally
- Use `world_size=1` (single GPU)

### "Expert capacity overflow" (MoE only)
- Increase `train_capacity` to 2.5 or 3.0
- Reduce batch size
- Check expert utilization balance

### "Training hangs at initialization" (DDP)
- Check both GPUs visible: `nvidia-smi`
- Verify MASTER_PORT not in use: `netstat -tuln | grep 23556`
- Try `world_size=1` to isolate issue

---

## 📚 Related Documentation

- **Debugging Report**: `DEBUGGING_REPORT.md` (detailed analysis of fixes)
- **Main README**: `README.md` (setup and general usage)
- **Copilot Instructions**: `.github/copilot-instructions.md` (architecture overview)
- **Direct Evaluation**: `DIRECT_EVALUATION_METHODOLOGY.md` (inference methods)

---

**For questions or issues, refer to the debugging report or copilot instructions.**
