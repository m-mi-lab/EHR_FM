# Inference after finetuning — what is computed, on what, and what comes out

Every number in the results (`data/results/`, MLflow `ICLR-finetune` and `ICLR-missingness`) is produced
by the same inference path. This is that path, exactly as run. Code: `code/src/finetune.py`
(`evaluate`), `run_missingness.py` (`score_cell`), `model.py` (`forward_clf`).

## 1. The model that is being run

(The finetune stage itself — labels, cohorts, inputs, loop, output files — is `docs/finetune.md`.)

A finetuned checkpoint (`data/models/<arch>_ft/<task>/best.pt`) is the pretrained backbone plus a linear
head, with **every parameter updated** during finetuning (`AdamW(m.parameters())`, lr 5e-5, weight decay
default, gradient clipping 1.0, AMP fp16 during training). Training loss is `BCEWithLogitsLoss` with
`pos_weight = (1 − prevalence) / prevalence` on the training split; batch 64 sequences (gpt2: 4 × 16
accumulation); at most 8 epochs; early stop after 2 epochs without a tuning-AUROC improvement of 1e-4;
the checkpoint with the best tuning AUROC is the one evaluated and shipped. The backbone is the same
architecture and vocabulary as pretraining (`strict=True` load), so nothing about tokenization changes
between pretraining, finetuning and inference.

## 2. The input at inference

One sample per labelled admission (`data.EICTask`, `docs/inputs.md` §6):

- `code`: `[GENDER, RACE, MARITAL, AGE_T1, AGE_T2]` (age at the window's first post-birth event) followed by
  the last `CONTEXT[arch] − 5` tokens of the patient's timeline **strictly before `prediction_time`**,
  tail-padded with `[PAD]` (id 0). Context is architecture-specific (`model.CONTEXT`: 512 BiLSTM/BERT,
  1024 GPT-2, 2048 Mistral/Mamba, 4096 BigBird).
- `mask`: 1 where `code != [PAD]`. This is the attention/validity mask and the pooling weight.
- no label, no time-delta side channel, no static-value channel: time is in the stream as `TIME//DELTA`
  tokens, values are in the code ids as fused deciles.

The held-out split is used whole (mortality: 42,652 admissions); the tuning split is used for model
selection only.

## 3. The forward pass (`model._Base.forward_clf`)

```
h = encode(code, mask)                       # (B, L, d) contextual states from the backbone
if causal:  pooled = h[b, mask.sum(1) − 1]   # state at the LAST attended position
else:       pooled = Σ_t mask_t · h_t / Σ_t mask_t   # mean over attended positions
logit = Linear(d → 1)(dropout(pooled))       # dropout is inactive in eval mode
p = sigmoid(logit)
```

- **Bidirectional models (BiLSTM, BERT, BigBird)** mean-pool every attended position, so every token —
  including a `TIME` token — contributes equally to the summary.
- **Causal models (GPT-2, Mistral, Mamba)** read the hidden state at the last attended position, i.e. the
  representation of the whole prefix as seen from its final token. Whatever that token is, is where the
  prediction is read off.
- Positions: BERT and GPT-2 (own implementations) add a fixed positional-encoding table to the token
  embeddings and GPT-2 applies a causal triangular mask; Mistral uses rotary positions; BigBird learned
  absolute positions; BiLSTM and Mamba have no positional signal beyond order. HF backbones receive
  `attention_mask` with an empty-row guard (a fully padded row attends to position 0 instead of nothing);
  Mamba receives no mask at all — its recurrence runs over the padded tail, but the pooled position is the
  last real token, before the padding.
- Mode: `m.eval()`, `torch.no_grad()`. Precision: fp32 in `finetune.evaluate` and in the lab-box sweeps;
  TF32 matmuls in the CHPC sweeps (`--tf32 1`, recorded in every JSON and MLflow run).
- Batches of 64; sequences are already fixed-length, so batching changes nothing.

## 4. What comes out

- A probability `p ∈ (0, 1)` per admission. **No threshold is applied anywhere** and no calibration is
  performed; every reported metric is a ranking metric.
- `AUROC` (`sklearn.roc_auc_score`) and `AUPRC` (`average_precision_score`) over the held-out split.
- `finetune.py` writes `result.json` (test AUROC/AUPRC, prevalence, best tuning AUROC, epoch history) and
  `test_preds.npz` (`y`, `p`) and logs to MLflow experiment `ICLR-finetune`.

## 5. Inference under missingness (`run_missingness.py`)

The same checkpoint, the same forward pass. For each of the 36 cells (6 families × 6 doses)
the input tensors are rewritten by `perturb.py` under true removal (`apply_event_remove`) *after* windowing
and *before* the forward pass — the model is never re-finetuned, never told which cell it is in, never given
an imputed value. Per cell: three draws of the keep-mask (one at dose 0), each patient scored once per draw,
AUROC/AUPRC averaged over draws, and a 2,000-resample bootstrap over patients with **one shared resample
matrix for every cell**, so the reported `delta_auroc` against the clean baseline and its CI are paired.
Output: `data/results/missingness/<arch>/<task>.event.json` (one row per cell: family, regime, protocol, frac, beta, auroc, auroc_sd, auprc, realized dose,
CI, delta CI, n, checkpoint hash) and one MLflow run per (model, task) in `ICLR-missingness` with each
family's curve as a metric indexed by realized dose in percent.

The null arm (dose 0) must reproduce the clean AUROC in every family; the run aborts otherwise.

## 6. The two reference arms (`leak_check.py`)

*Token removal, deprecated on 2026-11-16; that material is not included in this branch.*

## 7. What inference is not doing

- No imputation, no re-fitting, no per-cell adaptation, no test-time augmentation, no ensembling over draws
  (draws are averaged at the metric level, not the prediction level).
- No threshold, no calibration, no decision rule — only rankings.
- No access to anything after `prediction_time`, to raw numeric values, to dates, or to labels.
- The window is not re-filled after removal: a record that loses 90 % of its content is scored as the
  shorter record it has become, from the same starting point.
