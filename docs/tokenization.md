# Tokenization strategies (MEDS-Torch)

Source: Oufattole, Bergamaschi, Renc, Kolo, McDermott, Stultz. *MEDS-Torch: An ML
Pipeline for Inductive Experiments for EHR Medical Foundation Models.* NeurIPS 2024
Workshop on Time Series in the Age of Large Models.
<https://openreview.net/forum?id=uwnK2LZtjX>

MEDS-Torch does not originate any of the three; it unifies them under one
preprocessing path: *"Everything In Code (EIC) [16], Triplet [7, 19], Text Code [5]."*
That single codebase is why we use it — the three differ in exactly one thing
(how an observation becomes a model input) and share everything else.

All three consume MEDS: one row per event, `subject_id, time, code, numeric_value`.

---

## 1. Everything In Code (EIC)

Every modality is discretised into one categorical vocabulary. Values are binned
per code into deciles, and **the bin is fused into the code string**:

    code=lab//A, numeric_value=1.0   ->   lab//A//_Q_1

The fusion is literal — `meds_torch/utils/quantile_binning.py:109`:

```python
code_quantile_concat = pl.concat_str(pl.col("code"), pl.lit("//_Q_"), pl.col("quantile"))
```

Bin edges are per-code deciles, set in `MIMICIV_INDUCTIVE_EXPERIMENTS/configs/eic_config.yaml`:

```yaml
fit_quantile_binning:
  - name: values/quantiles
    quantiles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

There is no input encoder: `input_encoder/eic_encoder.py` is a documented no-op
("*Placeholder encoder, does nothing as the x-transformer model handles the
embedding of tokens*") that passes raw integer code IDs to the backbone.

**Consequence — vocabulary.** Fusion multiplies the vocabulary by the number of
bins. Measured on synthetic input:

| distinct codes | unfused | EIC fused | ratio |
|---|---|---|---|
| 100 | 110 | 1,000 | 9.1x |
| 1,000 | 1,010 | 10,000 | 9.9x |
| 10,000 | 10,010 | 100,000 | 10.0x |

At our ~19-22k code inventory this implies a ~200k-entry embedding table, which is
a real parameter cost and a rare-token problem. Measure it before assuming it.

MEDS-Torch also ships `meds_torch/utils/split_quantiles.py` to split
`lab//A//_Q_1` back into a code and a separate quantile row — i.e. to *un*-fuse.
Fused is the native form.

## 2. Triplet

One event becomes a single **summed embedding**, not a vocabulary symbol:

    e_triplet = W_code * c  +  CVE_time(t)  +  CVE_value(v)

Paper: *"Triplet represents each observation as three separate embeddings (time
delta, code, numeric value) that are summed together. The continuous components
(time delta and numeric value) are embedded using learned feed-forward networks,
while codes use learned codebook embeddings."*

The input is continuous, but the **output head factors back apart**, so it is still
autoregressively generable: *"MSE loss is used over the time and numeric values
while cross entropy is applied for the code."* Codes are sampled from a softmax;
time and value are deterministic conditional means, so a rolled-out trajectory is
not a draw from a joint predictive distribution over events.

## 3. Text Code

Identical to Triplet except the code embedding comes from a language model over the
code's text description rather than a lookup table: *"Text Code is similar to Triplet
but maintains codes as text strings rather than numeric IDs, allowing semantic
embedding."* Time and value pathways are unchanged.

---

## Reported results

All three were run with autoregressive forecasting as a pretraining objective
(Table 1, "Forecasting / AR" block; fine-tuned discriminative AUC):

| AR forecasting | hosp-Mort | ICU-Mort | Discharge-Mort | Readmission |
|---|---|---|---|---|
| EIC | 84.8 +/- 0.1 | 76.9 +/- 0.5 | 83.6 +/- 0.3 | 69.9 +/- 0.2 |
| Text Code | 81.9 +/- 0.4 | 71.6 +/- 1.3 | 81.9 +/- 0.4 | 69.1 +/- 0.3 |
| Triplet | 74.0 +/- 0.6 | 65.1 +/- 2.2 | 76.0 +/- 1.8 | 64.4 +/- 0.5 |

EIC is the strongest AR configuration; Triplet the weakest, by ~11 AUC points on
in-hospital mortality.

**The paper runs no generative experiment.** AR is a pretraining objective only;
every reported number is fine-tuned discriminative AUC. Trajectory generation and
zero-shot labelling exist in the codebase but are implemented for **EIC only**
(`models/eic_forecasting.py`, `generate_trajectories.py`, `ZERO_SHOT_TUTORIAL/`),
and landed after the workshop paper.

## Why this matters for the missingness work

The perturbation is "withhold a fraction of clinical content", so the tokenizer
decides whether that operator is even well defined.

All three MEDS-Torch strategies are **one unit per observation**, so removal is
exact: delete one position and exactly one clinical fact leaves, code and value
together. Two-token schemes (a type token followed by a separate bin token, as in
ETHOS) force removal to delete matched pairs
to hide one fact — and every architecture must implement that identically or the
comparison is confounded.

EIC additionally keeps the stream a pure integer sequence, so all six backbones
receive an identical interface. Triplet and TextCode require the backbone to accept
precomputed embeddings, which is workable but is one more thing that must be held
constant across architectures.
