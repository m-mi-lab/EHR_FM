# Finetuning — what goes in, what is trained, what comes out

The stage between a converged pretrained model and every downstream number. Code: `code/make_labels.py`
(labels), `code/src/data.py: EICTask` (inputs), `code/src/finetune.py` (loop and outputs). Inference on the
result is in `docs/inference.md`; the perturbed inputs of the sweep in `docs/inputs.md` §9.

## 1. Tasks and labels (`code/make_labels.py`)

One row per MIMIC-IV hospital admission (`hosp/admissions`, rows with both `admittime` and `dischtime`)
whose subject is in the tokenized corpus. Every label is defined strictly after a **binding prediction
time**, and the model never sees a token at or after it.

| task | label = 1 when | prediction time | cohort |
|---|---|---|---|
| `mortality` | in-hospital death (`hospital_expire_flag`) | admission + 24 h | admissions still open at 24 h (`admit + 24 h < dischtime`) |
| `los_gt_7d` | `dischtime − admittime` > 7 × 24 h | admission + 24 h | same |
| `readmit_30d` | the subject's next admission starts ≤ 30 days after this discharge (no next admission → 0) | discharge | all admissions |

Splits are **by subject**, taken from the current tokenization (`train` / `tuning` / `held_out`) and
regenerated whenever the corpus is rebuilt, so no admission's subject appears in two splits.

| task | admissions | train | tuning | held_out | prevalence |
|---|---|---|---|---|---|
| mortality | 425,041 | 339,544 | 42,845 | 42,652 | 2.35 % |
| los_gt_7d | 425,041 | 339,544 | 42,845 | 42,652 | 22.7 % |
| readmit_30d | 544,365 | 434,472 | 54,737 | 55,156 | 20.4 % |

Files: `data/labels/<task>.parquet` with columns `subject_id, hadm_id, split, prediction_time, label`, and
`data/labels/summary.json` with the counts above. The whole split is used: no subsampling in finetuning.

## 2. The input, one tensor pair per admission (`data.EICTask`)

- `code` (int64, length `CONTEXT[arch]`): the 5-token demographic prefix (sex, race, marital status, two
  age tokens for the age at the window's first post-birth event), then the **last `CONTEXT − 5` tokens of
  the subject's timeline strictly before `prediction_time`**, tail-padded with `[PAD]`. Strict `<` means
  that for mortality and LOS the model reads the first 24 h of the stay plus as much earlier history as fits,
  and for readmission the whole stay up to, but not including, the discharge token and the stay's billing
  diagnoses that sit at the discharge timestamp (`docs/inputs.md` §2, §6).
- `mask` (int64): 1 where `code` is not `[PAD]`.
- `label` (float): 0 or 1. `n_tokens` is also returned but only the (removed) token-removal leak check used it.

Inputs are identical for every architecture except the window length: 512 (BiLSTM, BERT families), 1024
(GPT-2 family), 2048 (Mistral, Mamba families), 4096 (BigBird). Loader: training split shuffled, batch 64
sequences (GPT-2-medium: 4 × 16 gradient accumulation), last partial batch dropped, 2 worker processes;
tuning and held-out unshuffled, batch 64, in file order. Seed 1234 for torch and numpy.


### 2a. Worked examples

Patient-level MIMIC-IV records (identifiers, dates, token sequences) are not reproduced in this
public repository, per the PhysioNet credentialed-access data use agreement.

During training the same forward pass is run on shuffled training admissions and the logit is compared with
the label by the weighted binary cross-entropy of §3; at evaluation the probabilities are kept and ranked
(AUROC, AUPRC). The model never sees a token at or after the prediction time in either phase.

## 3. What is trained, and how it stops

- **Model**: the pretrained backbone loaded strictly from `data/models/<arch>_pretrain/best.pt` (a silent
  partial load would finetune a random backbone) plus a linear head `d → 1`. **Every parameter is updated**.
- **Loss**: `BCEWithLogitsLoss` with `pos_weight = (1 − prevalence) / prevalence` of the training split
  (≈ 41 for mortality, ≈ 3.4 for LOS, ≈ 3.9 for readmission), so positives are not drowned out.
- **Optimiser**: AdamW, learning rate 5e-5, gradient-norm clip 1.0, mixed precision (autocast + GradScaler).
  No learning-rate schedule.
- **Epochs**: at most 8. After each epoch the model is scored on the tuning split (AUROC, AUPRC). If the
  AUROC beats the best so far by more than 1e-4 the weights are written to `best.pt`; otherwise a counter
  increments and the run **early-stops after 2 epochs without improvement**.
- **Then**: `best.pt` is reloaded and scored once on the held-out split. That is the number reported.

In practice runs stop after 3–5 epochs. Held-out AUROC of the finetunes so far:

| model | mortality | los_gt_7d | readmit_30d |
|---|---|---|---|
| bilstm | 0.934 | 0.837 | 0.777 |
| bilstms | 0.931 | 0.835 | 0.774 |
| bert | 0.939 | 0.843 | 0.783 |
| berts | 0.935 | 0.839 | 0.782 |
| gpt2 | 0.928 | 0.839 | 0.778 |
| mistrals | 0.933 | 0.842 | 0.785 |
| mambas | 0.936 | 0.846 | 0.789 |
| mamba | 0.941 | — | — |

## 4. What comes out

Per `(arch, task)`, in `data/models/<arch>_ft/<task>/`:

- `best.pt`: `{"model": state_dict, "vocab", "d", "layers", "heads", "arch"}` — the weights of the
  best-tuning-AUROC epoch. Written after every improving epoch, so **a `best.pt` without `result.json` is a
  killed run, not a finished one**; the CHPC sweep script re-finetunes in that case.
- `result.json`: `task, model, test_auroc, test_auprc, n_test, prevalence` (held-out), `val_best_auroc`,
  `history` (one entry per epoch: `epoch, val_auroc, val_auprc, secs`). Its presence is the "finished" flag.
- `test_preds.npz`: `y` (int8) and `p` (float32) for every held-out admission, in `EICTask` order, for
  re-scoring or calibration work without re-running the model.
- MLflow run in experiment `ICLR-finetune` (lab box :5001, CHPC mirrors :5002–:5004): tags
  `arch, task, stage=finetune, gpu`; params (every CLI argument, `train_n/val_n/test_n`, `prevalence`, the
  pretrain geometry); per-epoch `val_auroc, val_auprc, epoch_secs, best_val_auroc`; final `test_auroc,
  test_auprc, val_best_auroc, epochs_run`; `result.json` as an artifact.
- Log lines in `logs/exp_<arch>.log` (lab box) or `logs/sweep_<arch>.log` (CHPC): `epoch=k val_auroc=…
  val_auprc=… (secs)`, `early stop`, `RESULT <task>: test_auroc=… test_auprc=… n=…`.

Architectures pretrained on CHPC are finetuned there by `chpc/sweep.sh` with the same recipe and their
`best.pt` and `result.json` are copied to the lab box, so `data/models/*_ft/` on the lab box is the complete
set.

## 5. Who consumes them

- `run_missingness.py`: `best.pt` → the sweep (`docs/missingness.md`, `docs/inference.md` §5).
- `figures.py`: `result.json` for the task panels.

Nothing downstream re-trains anything: the finetuned weights are frozen from here on.
