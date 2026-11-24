# ICLR submission — EHR missingness

Paths are relative to the repo root (override with `ICLR_ROOT`). Tracking: MLflow on
port 5001 -- `pretrain.py`, `finetune.py` and `run_missingness.py` log natively
(experiments `ICLR-pretrain`, `ICLR-finetune`, `ICLR-missingness`); one run per
invocation, so a restart is a new run, not an overwrite.

`ICLR-pretrain` logs on one x-axis for every model -- the optimizer step, which
is 32,768 tokens for each of them:

| cadence | metrics |
|---|---|
| every 200 steps | `train_loss_step`, `train_ppl_step`, `lr_step`, `grad_norm` (pre-clip, finite steps), `amp_skipped_steps` (fp16 overflows), `tokens_per_sec` (non-pad), `amp_scale` (fp16 runs), `epoch_progress` |
| every 1,000 steps | `val_loss_step`, `val_ppl_step` on a fixed 1,024-window tuning subset with fixed MLM masks |
| every epoch | `train_loss`, `train_ppl`, `val_loss`, `val_ppl` (full tuning split -- drives the plateau LR and early stopping), `lr`, `epoch_secs`, `epoch`, `best_val_loss`; `history.json` artifact |
| every 60 s | `system/*`: GPU utilisation / memory / power for both GPUs, CPU, RAM, disk (mlflow's own sampler) |
| at the end | `final_best_val_loss`, `final_best_val_ppl`, `epochs_run`, tag `stop_reason` |

Params record the full CLI plus `params_M`, `vocab`, `train_windows_per_epoch`,
`train_tokens`, `val_n`, `objective`, `window_sampling`, `prefix_len`; tags
`arch`, `objective`, `gpu`, `stage`.

**Every result produced before the 2026-09-08 rebuild is void.** The corpus it
used was missing 400,000,000 rows of `icu/chartevents` (a chunking bug in
`shard_events`) and mapped 10 of 26 MIMIC-IV tables. Nothing from it should be cited.

## Layout

    code/src/               data.py model.py pretrain.py finetune.py perturb.py
                            run_missingness.py figures.py
    code/make_labels.py     task labels with binding clock cutoffs
    configs/                event_configs_seq.yaml (21 tables)  extract_MIMIC.yaml
    docs/                   tokenization.md  missingness.md
    data/MEDS  data/eic  data/labels  data/models  data/results
    run_full_pipeline.sh    MIMIC-IV -> MEDS -> EIC
    run_pretrain.sh         <gpu> <arch>...   pretrain to convergence
    run_experiments.sh      <arch>            finetune -> missingness -> figures

## Corpus (measured 2026-09-08)

| | |
|---|---|
| MIMIC-IV tables mapped | **21 of 26** — excluded: `drgcodes` (no timestamp), `emar_detail`/`poe_detail` (join tables), `provider`/`caregiver` (staff ids) |
| MEDS rows / subjects | 735,696,749 / 364,627 |
| tokenized patients | 252,477 (train 201,932 / tuning 25,202 / held_out 25,343) |
| tokens | 745,260,121 |
| vocabulary | 3,926 = 3,890 MEDS codes (2,886 fused `//_Q_k` + 1,004 plain) + PAD + [MASK] + 34 demographic/age |
| labels | mortality & los_gt_7d 425,041 stays (held-out 42,652) · readmit_30d 544,365 (held-out 55,156) |

Extraction settings that matter: `row_chunksize: 500000000` (every table in one
chunk; the 200M default lost two of chartevents' three chunks) and
`infer_schema_length: 10000000` (outputevents.value is integral for its first
10k rows and then isn't).

## Models

Trained for the paper: bilstm, bert, gpt2 at GPT-2-medium geometry, and
mistral -- the Mistral architecture (RoPE, SwiGLU, RMSNorm, GQA) at BERT-base
scale, since no small Mistral was ever released. bigbird and mamba exist in
`model.py` behind the same interface but are not trained. One fused EIC token stream for all; context is
model-specific (its published geometry); every optimizer step sees 32,768
tokens.

| arch | params | context | objective | pooling | dropout / wd |
|---|---|---|---|---|---|
| bilstm | 20.2M (L=4) | 512 | MLM | mean | 0.1 / 0.01 |
| bert | 91.1M | 512 | MLM | mean | 0.1 / 0.01 |
| bigbird | 94.8M | 4096 block-sparse | MLM | mean | 0.1 / 0.01 |
| bilstms | 10.3M (d=512, L=4) | 512 | MLM | mean | 0.1 / 0.01 |
| bilstml | 33.2M (d=1024, L=4) | 512 | MLM | mean | 0.1 / 0.01 |
| berts | 29.2M (BERT-medium: L=8, d=512) | 512 | MLM | mean | 0.1 / 0.01 |
| bertl | 310.4M (BERT-large) | 512 | MLM | mean | 0.1 / 0.01 |
| gpt2 | 310M (medium: d=1024, L=24, H=16) | 1024 | next-token | last | 0.3 / 0.10 |
| gpt2s | 91.1M (small: d=768, L=12, H=12) | 1024 | next-token | last | 0.3 / 0.10 |
| gpt2l | 718M (large: d=1280, L=36, H=20) | 1024 | next-token | last | 0.3 / 0.10 |
| mistrals | 34.4M (Mistral arch: d=512, L=8, H=8) | 2048 | next-token | last | 0.3 / 0.10 |
| mistrall | 373M (Mistral arch: d=1024, L=24, H=16) | 2048 | next-token | last | 0.3 / 0.10 |
| mambas | 31.1M (Mamba: d=512, 16 blocks) | 2048 | next-token | last | 0.3 / 0.10 |
| mambal | 328M (mamba-370m geometry: d=1024, 48 blocks) | 2048 | next-token | last | 0.3 / 0.10 |
| mistral (Mistral-arch @ BERT-base scale) | 108.7M | 2048 | next-token | last | 0.3 / 0.10 |
| mamba (not trained) | 96.6M (L=24, mamba-130m depth) | 2048 | next-token | last | 0.3 / 0.10 |

Inputs follow ETHOS: every sample is a 5-token demographic prefix (sex, race,
marital, age at the window's first clinical event) plus a timeline window;
pretraining draws windows uniformly over all tokens in the corpus (one pass per
epoch), evaluation takes the last window before the prediction time. Window
draws are a pure function of (seed, epoch, index), so all architectures see the
same starts. `docs/inputs.md` has every step. Convergence: 500-epoch cap,
early-stop patience 5, plateau LR (halve after 3 flat epochs), AMP.
The causal pair gets more regularisation because next-token prediction has a
fixed target every epoch while MLM's random mask is its own regulariser.
HF Mistral exposes only `attention_dropout`; residual dropout (GPT-2's
`resid_pdrop`, on each attention/MLP/SSM branch before it is added to the
residual stream) is added by forward hook. Mamba runs AMP in bf16 -- its
selective scan overflows fp16 -- every other model in fp16.

Context window: pre-cutoff history at the mortality cutoff has median 986
tokens (p25 259, p75 2,920, mean 4,102 -- a long tail). The 512 window holds
the whole record for 37.6% of admissions and covers about half of the median
one; for 49% of admissions it sees less than half.

## Missingness

Per family, **true removal** (code name `event`): the withheld tokens are deleted as events, as if they had
been taken out of the record before it was tokenized. An event left with no content disappears with its
time-gap token, and the gaps of the surviving events are recomputed from the timestamps. Six intensities
x 3 draws x full held-out split, 95% patient bootstrap (2,000 resamples, shared index matrix) plus a paired
delta interval. Null arm must reproduce baseline exactly in every cell.

Token removal (delete the token, keep every time-gap token) and the leak check built on it are not
included in this branch.

C3 (culture -> antibiotic sensitivity) is the reason `microbiologyevents` was
mapped: the parent/child dependency is recorded in the table itself.

## Toolchain

The Python env (`PY`) is deliberately external — venvs hardcode
absolute paths in console-script shebangs, and it holds a resolved stack
(`meds` 0.3.3, torch 2.6.0+cu124, polars 1.14).
