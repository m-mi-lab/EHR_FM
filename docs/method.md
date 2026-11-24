# The method, end to end: ideas, approach, inference, missingness, results

Written 2026-10-08 against the rebuilt pipeline in this repo (code in `code/src`, results in
`data/results`). Every number is measured on held-out MIMIC-IV unless marked otherwise. The four
companion documents hold the step-by-step detail this one summarises: `docs/inputs.md` (tables to
tokens), `docs/finetune.md` (tasks, labels, training loop), `docs/inference.md` (the forward pass),
`docs/missingness.md` (the simulation design, worked examples, protected set).

---

## 1. The question

An EHR record is incomplete by construction, and a model that looks robust to missing data in a
paper may be leaning on exactly the information that disappears in deployment. We ask how
foundation models for patient records degrade when clinical content is withheld.

**What goes missing in a real record.** A lab that was never ordered. A vital that was not charted
overnight. A monitoring stream that stops at a ward transfer. Little of this is random: sicker
patients get more labs, stable patients fewer, and some values are absent because the clinician
already knew the answer. Rubin's regimes name the cases. *MCAR*: the absence depends on nothing.
*MAR*: it depends on things we can observe (the token's modality, the visit index, the clock).
*MNAR*: it depends on the hidden value itself or on the latent state that produced the record.

**Why deployment makes it urgent.** A model pretrained on dense academic-centre records meets
sparser, differently shaped records at another hospital or on another ward. The question is not
whether AUROC falls but how fast, under which kind of absence, and whether the model fails quietly
while still producing confident scores.

**Two questions.**

1. Across architectures and sizes trained on identical inputs, how does downstream performance
   degrade as content is withheld under MCAR, MAR and MNAR?
2. How much of a model's clean performance comes from clinical content, and how much from the
   scaffold around it (how many events there are, when they happen, where the patient was)?

## 2. The pipeline in one picture

```
MIMIC-IV v2.2 ──► MEDS rows ──► EIC tokens ──► pretrain (MLM or next-token)
 21 tables        736M rows      3,926 ids       best checkpoint, identical inputs for all 15 models
                                                        │
                                                        ▼
            finetune (mortality, LOS, readmission) ──► corrupt the held-out records ──► score
            backbone + linear head, every weight        6 families × 6 doses                  AUROC, paired
            updated, best tuning-AUROC epoch kept       same frozen, hash-checked weights     bootstrap CI
                                                        in every cell; only the tensor changes
```

Corruption happens only at test time, after finetuning. Nothing is retrained under missingness, so
a drop in a cell is attributable to the input alone.

## 3. Data and tokens (summary of `docs/inputs.md`)

| | |
|---|---|
| source | MIMIC-IV v2.2, 21 of 26 tables, via MEDS (one `(subject, time, code, value)` row per observation) |
| MEDS rows / subjects | 735,696,749 / 364,627 |
| filters | codes seen in < 10,000 subjects dropped; subjects with < 10 events dropped |
| tokenized patients | 252,477: train 201,932 / tuning 25,202 / held-out 25,343, split by subject |
| tokens | 745,260,121 |
| vocabulary | 3,926 ids: 2,886 fused `code//_Q_k`, 1,004 plain codes, 34 demographic, `[PAD]`, `[MASK]` |

**One token = code plus decile.** A lab with a numeric result becomes `code//_Q_k`, k the decile of
the value among training patients (deciles fit on train only; 405 lab stems carry them).
`LAB//50912//_Q_1` is a creatinine in the lowest tenth of values. Codes without a value (orders,
transfers, admission types, medication starts) stay as bare codes. There is no separate value
channel.

**Time is a token.** At every new timestamp a `TIME//DELTA//TOKEN//_Q_k` token is inserted, k from
22 fixed boundaries running from one minute to years. The model never sees a clock, only elapsed
time between events.

**One sample.** A 5-token demographic prefix (sex, race, marital status, two age digits), then the
last tokens of the patient's history strictly before the prediction time, tail-padded to the
model's context: 512 (BiLSTM, BERT), 1,024 (GPT-2), 2,048 (Mistral, Mamba), 4,096 (BigBird). For
62 % of mortality windows at 512 the history is longer than the window.

**Scaffold versus content.** The scaffold is the record's skeleton: the prefix, every TIME token,
and the admission, discharge, transfer, service, ICU, birth and death markers (110 token types, §8.7).
Content is everything else: labs, medications, orders, administrations, pharmacy, procedures,
diagnoses, microbiology. In a typical 512-token window content fills 77 % of the slots, TIME 19 %.

## 4. The model ladder

| family | sizes (parameters) | reads | mixes positions by | window | summary for prediction |
|---|---|---|---|---|---|
| BiLSTM | 10M · 20M · 33M | both directions | recurrent state, token to token | 512 | mean over positions |
| BERT | 29M · 91M · 310M | both directions | full attention | 512 | mean over positions |
| BigBird | 95M | both directions | sparse attention (local + global + random) | 4,096 | mean over positions |
| GPT-2 | 91M · 310M · 718M | left to right | causal attention | 1,024 | state at the last token |
| Mistral | 34M · 109M | left to right | sliding-window causal attention | 2,048 | state at the last token |
| Mamba | 31M · 97M · 328M | left to right | selective state space (gated recurrence) | 2,048 | state at the last token |

Fifteen models, three mechanisms (recurrence, attention, state space), both reading directions, one
long-context encoder. The 90M to 110M tier (bert, gpt2s, bigbird, mamba, mistral) is where families
compare at near-equal size.

## 5. Pretraining

- **Objective by direction.** Bidirectional models do masked-language modelling: 15 % of the tokens
  in a window are replaced by `[MASK]` and predicted. Causal models predict the next token.
- **Windows** are drawn from each patient's stream at the model's context length; the inputs are the
  same stream for every model.
- **Optimiser.** AdamW, learning rate 3e-4 with 500 warm-up steps (BigBird 1e-4; BERT-large 5e-5,
  2,000 warm-up steps, ε = 1e-6 after three diverged attempts), mixed precision (bf16 for the Mamba
  family, BigBird and BERT-large; fp16 elsewhere), per-model micro-batches.
- **Stopping rule.** Validation loss on the tuning split after each epoch; learning rate halves on a
  plateau; the run stops after five epochs without improvement; the best checkpoint is kept.
- Losses are not comparable across the two objectives, so models are never ranked by them.

## 6. Finetuning (summary of `docs/finetune.md`)

Every label is defined strictly after a binding prediction time, and the model never sees a token
at or after it. This cutoff is what makes the downstream numbers honest: the earlier cohort handed
models the whole stay, and length of stay could be read off the record's length.

| task | label = 1 when | prediction time | admissions (held-out) | prevalence |
|---|---|---|---|---|
| `mortality` | in-hospital death | admission + 24 h | 425,041 (42,652) | 2.35 % |
| `los_gt_7d` | stay > 7 days | admission + 24 h | 425,041 (42,652) | 22.7 % |
| `readmit_30d` | next admission ≤ 30 days after discharge | discharge | 544,365 (55,156) | 20.4 % |

For mortality and LOS the model reads the first 24 h of the stay plus as much earlier history as
fits. For readmission it reads the whole stay up to, but not including, the discharge token and the
billing diagnoses stamped at discharge.

**Recipe.** Pretrained backbone (`best.pt`, strict load) plus a linear head, every parameter
updated. `BCEWithLogitsLoss` with `pos_weight = (1 − prevalence) / prevalence`; AdamW 5e-5;
gradient clipping 1.0; batches of 64 sequences (accumulated from smaller micro-batches for the
large models); at most 8 epochs; early stop after 2 epochs without a tuning-AUROC improvement of
1e-4. The best epoch is reloaded and scored once on the held-out split. Runs stop after 3 to 5
epochs in practice.

## 7. Inference (summary of `docs/inference.md`)

The same forward pass produces every number in the results: clean scores and every missingness cell.

```
h      = encode(code, mask)                       # (B, L, d) contextual states
pooled = h[b, mask.sum(1) − 1]                    # causal: state at the LAST attended position
       | Σ_t mask_t · h_t / Σ_t mask_t            # bidirectional: mean over attended positions
logit  = Linear(d → 1)(pooled)
p      = sigmoid(logit)                            # one probability per admission
```

- Inputs: the `code` and `mask` tensors of §3, nothing else. No label, no time-delta side channel,
  no raw values, no dates.
- Mode `eval()`, no gradients, fp32 on the lab box and TF32 matmuls on CHPC (recorded per JSON).
- Outputs: a probability per admission; AUROC and AUPRC over the split. No threshold, no calibration,
  no decision rule; every reported metric is a ranking metric.
- What inference is not doing: no imputation, no re-fitting, no per-cell adaptation, no test-time
  augmentation, no ensembling over draws, no access to anything after the prediction time. A record
  that loses 90 % of its content is scored as the shorter record it has become.

## 8. Simulating missingness

### 8.1 Where we intervene

On the tensorized stream, after windowing and before the forward pass: `perturb.py` rewrites `code`
and `mask`, using each event's boundaries and timestamp (`EICTask(..., with_events=True)` returns them
with the window). The vocabulary stays byte-identical across every cell (perturbing before
vocabulary fitting would silently change ids between arms), and the choice of which tokens go is
drawn once per cell and draw. The window is cut before the removal and is not refilled.

### 8.2 The mechanism

One formula decides which eligible tokens are withheld, the amputation form of `mice::ampute`:

```
logit P(token j withheld | Z_j) = α + β · s(Z_j)
```

`Z` is whatever the family is allowed to look at, `s` is a score scaled to [−1, 1], and `β` sets
how strongly the choice depends on it. The regime is set entirely by what `Z` may be:

- `β = 0`: nothing depends on anything. **MCAR**, nested inside every family at zero contrast.
- `Z` an observed property (the token's modality): **MAR**.
- `Z` the value itself or the latent state that produced the record: **MNAR**.

For a requested dose `f`, `α` is solved by bisection on [−8, 8] so that the mean withholding
probability over the eligible pool equals `f`. Positivity holds by construction (no token has
retention probability exactly 0 or 1). The MNAR families run at `β = 4`.

### 8.3 The six families that run

`run_missingness.py: FAMILIES`. Every family draws from content tokens only (§8.7).

| id | regime | `Z` | eligible pool | score `s` | what it imitates |
|---|---|---|---|---|---|
| **A1** | MCAR | nothing | all content | none | uniform random loss of content: the reference curve |
| **A3** | MCAR | nothing, in contiguous runs | all content | none; Gilbert two-state channel, mean run length 5, exact count | a monitoring stream that drops out for stretches |
| **B1** | MAR | the token's modality | labs only | none | a lab system that does not link |
| **C1** | MNAR | how many times this code stem has already appeared | all content | repeats first: `clip(k, 0, 5)/5·2−1`, cap on the stem ignoring the decile | the clinician who stops re-ordering: suppression depends on the latent concern that also drives the outcome |
| **C2** | MNAR | the value's decile | labs with a value (67 % of content) | low deciles first: `(5.5 − d)/4.5`, `ampute` type LEFT | value-dependent withholding. A sensitivity probe, not a clinical mechanism: `_Q_1` sodium is profoundly abnormal and always reported |
| **C3** | MNAR | whether a culture grew an organism | antibiotic-sensitivity panels | +1 after an organism token, −1 otherwise | the reflex result that exists only because its parent did. Tiny pool, no measurable effect |
| **A2** | MCAR | nothing (whole prior visits) | complete prior visits in the window | none | prior encounters absent from this system (own file `<task>.visits_event.json`) |
| **B2** / **B2o** | MAR | the visit's recency rank | complete prior visits in the window | most recent first / oldest first, β = 4 | an unlinked recent (or old) encounter |

A prior visit is the run of tokens between two consecutive hospital-discharge markers, the second
included, so a withheld visit takes its ED arrival, admission, stay, discharge and time tokens with it;
the open visit holding the prediction time is never withheld. Doses are fractions of a record's eligible
visits, rounded half up.

**Which models, and why (decided 2026-10-08).** A complete prior visit is about 150 to 300 tokens, so
whether a window can hold several depends on the context length: three or more complete prior visits fit
for 1 % of mortality records at 512 tokens, 4 % at 1,024, 11 % at 2,048 and 19 % at 4,096 (8 %, 15 % and
21 % of readmission records). A visit-level family needs at least three prior visits in view for its doses
to be graded in visits rather than a single with-or-without contrast, so these families run only for the
models whose window can hold three: the GPT-2 family (1,024), the Mistral and Mamba families (2,048) and
BigBird (4,096). The BiLSTM and BERT families are excluded by design, not by budget: a 512-token window
holds three complete prior visits for 1 % of records, so there is nothing for the family to withhold, and
a result on that 1 % would not be a statement about the model. Within the eligible models the families
score the records with at least three complete prior visits in the window (`--min_visits 3`: about 1,700
mortality admissions at 1,024 tokens, 4,700 at 2,048, 8,100 at 4,096), and the JSON records the subset
size, the median visit count and the realized dose per cell. With 2.35 % mortality prevalence the smaller
subsets hold few deaths, so the mortality cells of this family carry wide intervals, especially for GPT-2;
LOS and readmission are better powered. The subset also grows with the window, so the comparison inside
this group is at each model's own window, as for every other family.

**For the paper (methods, visit-level families):** "Visit-level removal is evaluated only for the
architectures whose context window can contain at least three complete prior admissions (GPT-2 at 1,024
tokens, Mistral and Mamba at 2,048, BigBird at 4,096), on the subset of test admissions for which it does.
The 512-token models (BiLSTM, BERT) see three complete prior admissions in 1 % of records, so withholding
prior visits is not a well-defined intervention for them and they are omitted from this family rather
than reported on an unrepresentative subset."

Denominators differ: `f = 0.5` is half of all content for A1 and half of the labs for B1. The
realized fraction of that family's own pool is recorded per cell, and A1 is the matched-dose
reference curve.

### 8.4 What was designed, what was run, and what the earlier harness ran

The design in `docs/missingness.md` and `configs/missingness/` has **nine** families; the rebuilt grid
ran six until 2026-10-08, when the two visit-level families were added as an incremental run. The visit
families as drafted:

| id | regime | what it would do | why it is not in the grid |
|---|---|---|---|
| A2 | MCAR | withhold every token of randomly chosen prior visits, admission and discharge markers included | **implemented 2026-10-08** (§8.3); the October runs used token removal (not included) |
| B2 | MAR | withhold prior visits ordered by recency (newest-first B2 / oldest-first B2o) | **implemented 2026-10-08** (§8.3), as A2 |

Also designed and not run: the **matched-budget twin** for C2 (same code, random decile, which
isolates value-rank from code identity), and the **β sweep** at fixed dose with a breakdown value
(the grid uses β ∈ {0, 4} only).

The earlier harness (the August paper, `sections/methods.tex`) ran a different set on the old
cohort: F1 graded content removal, F2 per-modality withholding (labs / meds /
procedures, deterministic), F3 visit-level removal (most recent, oldest, random, all but
the last), F4 zeroing of the age and inter-event-time side streams. F2 maps onto B1, F1 onto A1.
F3 and F4 have no counterpart in the rebuild: F3 was the one whose finding ("the models lose the
visit trajectory, not the content") was circular under the old cohort bug, and F4's effect
("temporal streams carry 0.19 AUROC") was refuted once the harness was fixed. Visit-level removal
is therefore an open item (A2/B2), not a result.

### 8.5 The operator

**True removal** (`perturb.apply_event_remove`, code name `event`): the withheld tokens are deleted as
events, as if they had been taken out of the record before it was tokenized. An event left with no
content disappears together with its time-gap token, and the time-gap token of every surviving event is
re-binned from the timestamps with the tokenizer's 22 edges. That rule reproduces 100 % of the stored
time-gap tokens (189,677 held-out events, 2026-10-28). The draw of withheld tokens is seeded by family,
dose, β and draw number. The window is cut before the removal and is not refilled.

Token removal, which this section described until 2026-11-16 (delete the token, contract the record,
keep every time-gap token), is not included in this branch.

### 8.6 The grid, the draws, and what is recorded

- Doses `f ∈ {0, 0.1, 0.25, 0.5, 0.75, 0.9}` of each family's eligible pool; 6 families × 6 doses =
  **36 cells** per (model, task).
- Three draws of the keep-mask per cell (one at dose 0), seeded from `(family, f, β, draw)`; every
  patient scored once per draw; AUROC and AUPRC averaged over draws.
- A 2,000-resample patient bootstrap with **one shared resample matrix for every cell**, so
  `delta_auroc` against the clean baseline and its interval are paired.
- The dose-0 cell must reproduce the clean baseline logits; the run aborts otherwise (the
  null-arm check).
- A per-cell checkpoint (`<task>.event.json.partial`, added 2026-10-09): a sweep killed by a session limit resumes
  from the cells it finished instead of restarting. Draws are seeded per cell, so a resumed sweep equals an
  uninterrupted one up to GPU run-to-run noise (3e-4 AUROC on a 600-record BiLSTM test; far smaller on the
  full split). The null-arm check stays exact within a process and tolerates ≤ 2e-3 across the restart; rows
  scored before the restart carry `scored_before_restart: true`. The checkpoint is deleted when the JSON is written.
- Per cell: family, regime, protocol, `f`, `β`, AUROC, its SD over draws, AUPRC, realized dose,
  CI, delta CI, `n`, checkpoint SHA. Output `data/results/missingness/<arch>/<task>.event.json`
  (`protocol` is `event` in every row) and one MLflow run per (model, task) in `ICLR-missingness`.
- Assertions that abort a run (`assert_event_cell`): realized fraction matches the request over the
  family's own denominator; the withheld content is absent and the kept content is present in its
  order; the prefix is unchanged; only time-gap tokens differ from the kept sequence.

### 8.7 The protected set

110 vocabulary entries are never withheld by a family in any cell: TIME (24), TRANSFER_TO (16),
HOSPITAL_DISCHARGE (14), HOSPITAL_ADMISSION (9), SERVICE (7), ICU (2), MEDS_BIRTH and MEDS_DEATH,
the 34 demographic prefix tokens, `[PAD]` and `[MASK]`. A corrupted record always keeps who the
patient is and where they were. Only the clinical content between those anchors thins out. The
time-gap tokens follow the events: the one at the head of an event that loses all its content
disappears with the event, and the others are recomputed from the timestamps (§8.5). The transfer,
admission-type and service tokens carry prognosis on their own.

### 8.8 What 90 % looks like

*Token removal, deprecated on 2026-11-16; that material is not included in this branch.*

## 9. Leak checks

*Token removal, deprecated on 2026-11-16; that material is not included in this branch.*

## 10. Results so far (2026-10-08)

### 10.1 Clean held-out AUROC after finetuning

| model | mortality | LOS > 7 d | readmission |
|---|---|---|---|
| bilstms | 0.931 | 0.835 | 0.773 |
| bilstm | 0.934 | 0.837 | 0.776 |
| bilstml | 0.934 | 0.838 | 0.777 |
| berts | 0.935 | 0.839 | 0.782 |
| bert | 0.939 | 0.843 | 0.783 |
| gpt2s | 0.933 | 0.837 | 0.781 |
| gpt2 | 0.928 | 0.839 | 0.777 |
| mistrals | 0.933 | 0.842 | 0.785 |
| mistral | 0.929 | 0.840 | 0.781 |
| mambas | 0.936 | 0.846 | 0.789 |
| mamba | 0.941 | 0.848 | 0.786 |
| mambal | 0.943 | running | running |

The spread is 0.01 to 0.02 on every task, and size does not order it: the largest BiLSTM and the
medium GPT-2 and Mistral are not the best of their families.

### 10.2 Mean AUROC lost at dose 0.5, six families and the finished tasks pooled

*Token removal, deprecated on 2026-11-16; that material is not included in this branch.*

### 10.3 Mortality AUROC at 90 % of content withheld, family A1

*Token removal, deprecated on 2026-11-16; that material is not included in this branch.*

### 10.4 Across families

*Token removal, deprecated on 2026-11-16; that material is not included in this branch.*

## 11. Intuition

*Token removal, deprecated on 2026-11-16; that material is not included in this branch.*

## 12. Status and open questions (2026-10-08, 10:00 MDT)

The sweeps and leak checks counted here are the October token-removal runs (code not included).

**Status.** 108 of 150 pipeline pieces (15 models × pretrain, 3 finetunes, 3 leak checks, 3 sweeps)
are done. Complete: bilstms, bilstm, bilstml, berts, bert, gpt2s, gpt2, mistrals, mambas. Finishing:
mamba (readmission sweep, last family), mistral (LOS and readmission sweeps). mambal: pretrained
(best val 1.4478 at epoch 29), mortality finetune 0.943, leak check done, mortality sweep running,
readmission finetune running. Still pretraining: gpt2l (epoch 29, best 1.5842), bigbird (epoch 10,
best 1.0098), bertl from scratch (epoch 21, best 1.0243, improving every epoch). About 500 GPU-hours
of finetunes and sweeps remain for the four large models once their pretrains converge.

**Open questions.**

1. Visit-level removal (A2/B2) is running; the earlier harness ran it on the old cohort only.
2. How much does scale buy? The 310M to 718M models are the remaining evidence.
3. Sweep cost for the large models: full held-out split (20 to 27 h per sweep at 100M parameters)
   versus a 10,000-admission subsample (4 to 5× faster, intervals about twice as wide).
4. Provenance: mamba, mambas and mistrals resumed late in pretraining from `best.pt` with a fresh
   optimiser after a checkpoint bug. Re-pretrain from scratch or document.

## 13. Full-split recompute of the §11 figures (2026-10-08)

*Token removal, deprecated on 2026-11-16; that material is not included in this branch.*
