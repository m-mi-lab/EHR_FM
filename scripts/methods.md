# Final Evaluation Scripts - Three Methods

Similar to the mortality prediction scripts, we have three evaluation methods:

## 1. `eval_final.py` - Frequency-Based (ETHOS Style)

**Method:** Count outcomes across N trajectories
```
Probability = positive_count / (positive_count + negative_count)
```

**Pros:**
- Matches ETHOS methodology
- Captures trajectory diversity
- Interpretable (e.g., "15 out of 20 trajectories predicted death")

**Cons:**
- Discrete probabilities (0.0, 0.05, 0.10, ..., 1.0 for N=20)
- Slower (requires N forward passes per patient)
- Ignores model confidence

**Usage:**
```bash
python scripts/eval_final.py --config eval_final_config.yaml
```

---

## 2. `eval_final_1.py` - Direct Probability (Fast)

**Method:** Single forward pass, use softmax directly
```
Probability = softmax[positive] / (softmax[positive] + softmax[negative])
```

**Pros:**
- ⚡ **10x faster** (1 forward pass vs N)
- Continuous probabilities (0.0 to 1.0)
- Deterministic (no sampling variance)
- Better AUROC discrimination

**Cons:**
- Doesn't capture trajectory diversity
- Single point estimate (no uncertainty)
- May not match ETHOS methodology

**Usage:**
```bash
python scripts/eval_final_1.py --config eval_final_config.yaml
```

---

## 3. `eval_final_2.py` - Confidence-Weighted (Best of Both)

**Method:** Run N trajectories, weight each outcome by model confidence
```
Probability = sum(positive_confidences) / sum(all_confidences)
```

**Pros:**
- ✨ **Best of both worlds**
- Captures trajectory diversity (like frequency)
- Uses model confidence (like direct)
- Continuous probabilities
- Better calibration than frequency-based

**Cons:**
- Slower than direct (requires N forward passes)
- More complex to interpret

**Usage:**
```bash
python scripts/eval_final_2.py --config eval_final_config.yaml
```

---

## Comparison Summary

| Method | Speed | Probability Type | Trajectory Diversity | Model Confidence | AUROC Quality |
|--------|-------|------------------|---------------------|------------------|---------------|
| **Frequency** | Slow (N passes) | Discrete | ✅ Yes | ❌ No | Good |
| **Direct** | ⚡ Fast (1 pass) | Continuous | ❌ No | ✅ Yes | Better |
| **Confidence** | Slow (N passes) | Continuous | ✅ Yes | ✅ Yes | **Best** |

---

## Which Method to Use?

- **For ETHOS comparison:** Use `eval_final.py` (frequency-based)
- **For fast iteration:** Use `eval_final_1.py` (direct probability)
- **For best performance:** Use `eval_final_2.py` (confidence-weighted) ⭐

---

## Configuration

All three scripts use the same config file (`eval_final_config.yaml`):

```yaml
model: ${oc.env:BEST_MODEL_PATH}
data_dir: ${oc.env:TOKENIZED_TRAIN_DIR}
output_dir: eval_final_results

num_patients: 100
num_trajectories_per_patient: 20  # Ignored by eval_final_1.py
max_tokens: 10000
temperature: 1.0
base_seed: 42
```

---

## Example Output

All three scripts produce the same output format:

```json
{
  "hosp_mortality": {
    "auroc": 0.8234,
    "auprc": 0.7891,
    "accuracy": 0.7600,
    "method": "confidence"
  },
  "icu_mortality": {
    "auroc": 0.8456,
    "auprc": 0.8123,
    "accuracy": 0.7800,
    "method": "confidence"
  },
  ...
}
```


