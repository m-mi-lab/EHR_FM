# Input construction — every step, exactly as run (2026-09-08, ETHOS-matched)

Source: MIMIC-IV v2.2 (PhysioNet credentialed access; path set via `MIMICIV_RAW_DIR`).
Everything below is what the code does, read from the configs and source, not a description of intent.

---

## 1. Raw tables → MEDS rows

`run_full_pipeline.sh` → `pre_MEDS_seq.py` (upstream MEDS_transforms MIMIC-IV example: joins dictionary
tables, writes parquet) → `MEDS_transform-runner` with `configs/extract_MIMIC.yaml`, five stages:

| stage | what it does | settings that matter |
|---|---|---|
| `shard_events` | reads each CSV, writes one parquet per table | `row_chunksize: 500000000` — every table in ONE chunk. The 200M default split `chartevents` (432,997,491 rows) into 3 chunks that all wrote to the same file; only the last 32,997,491 survived. `infer_schema_length: 10000000` — `outputevents.value` is integral for its first 10k rows and then isn't. |
| `split_and_shard_subjects` | assigns every subject to a split | `seed: 1`, `train 0.8 / tuning 0.1 / held_out 0.1`, `n_subjects_per_shard: 5000` → 75 shards. Deterministic given the subject set. |
| `convert_to_sharded_events` | applies `configs/event_configs_seq.yaml` | see §2 |
| `merge_to_MEDS_cohort` | one sorted `(subject_id, time, code, numeric_value)` table per shard | rows with null `time` are dropped (7.7% of microbiology) |
| `finalize_MEDS_data` | schema validation | |

Output: **735,696,749 rows, 364,627 subjects**.

## 2. The 21 event mappings (`configs/event_configs_seq.yaml`)

Each row of a raw table becomes one MEDS row with `code` = the joined parts and `time` = the named column.
`numeric_value` is set only where listed; everything else is a bare code.

| table | code | time | numeric |
|---|---|---|---|
| `hosp/patients` | `MEDS_BIRTH` | `year_of_birth` (**year only**, `%Y`) | |
| | `MEDS_DEATH` | `dod` | |
| `hosp/admissions` | `HOSPITAL_ADMISSION//<admission_type>` | `admittime` | |
| | `HOSPITAL_DISCHARGE//<discharge_location>` | `dischtime` | |
| `hosp/transfers` | `TRANSFER_TO//<eventtype>//<careunit>` | `intime` | |
| `hosp/services` | `SERVICE//<curr_service>` | `transfertime` | |
| `icu/icustays` | `ICU//admit`, `ICU//discharge` | `intime`, `outtime` | |
| `hosp/labevents` | `LAB//<itemid>` | `charttime` | `valuenum` |
| `icu/chartevents` | `LAB//<itemid>` (**same prefix as labevents**) | `charttime` | `valuenum` |
| `icu/outputevents` | `OUTPUT//<itemid>` | `charttime` | `value` |
| `icu/datetimeevents` | `DATETIME//<itemid>` | `charttime` | |
| `hosp/diagnoses_icd` | `DIAGNOSIS//<icd_code>` | **`hadm_discharge_time`** | |
| `hosp/procedures_icd` | `PROCEDURE//<icd_code>` | `chartdate` (date only) | |
| `icu/procedureevents` | `PROCEDURE//START//<itemid>`, `PROCEDURE//END//<itemid>` | `starttime`, `endtime` | |
| `hosp/prescriptions` | `MEDICATION//START//<ndc>`, `MEDICATION//STOP//<ndc>` | `starttime`, `stoptime` | |
| `hosp/emar` | `EMAR//<event_txt>` (Administered / Not Given / …) | `charttime` | |
| `hosp/pharmacy` | `PHARMACY//<route>` | `starttime` | |
| `hosp/poe` | `ORDER//<order_type>` | `ordertime` | |
| `icu/inputevents` | `INFUSION_START//<itemid>`, `INFUSION_END//<itemid>` | `starttime`, `endtime` | |
| `icu/ingredientevents` | `INGREDIENT_START//<itemid>`, `INGREDIENT_END//<itemid>` | `starttime`, `endtime` | `amount` (start only) |
| `hosp/microbiologyevents` | `MICRO//CULTURE//<spec_type_desc>` | `charttime` | |
| | `MICRO//ORGANISM//<org_name>` | `charttime` | |
| | `MICRO//AB//<ab_name>//<interpretation>` (S/I/R/D/Z) | `charttime` | |
| `hosp/omr` | `OMR//<result_name>` | `chartdate` (date only) | |
| `hosp/hcpcsevents` | `HCPCS//<hcpcs_cd>` | `chartdate` (date only) | |

Not mapped, and why: `drgcodes` (no timestamp column), `emar_detail` / `poe_detail` (join tables keyed on
`emar_id` / `poe_id`, no subject+time), `provider` / `caregiver` (staff-id dictionaries), `d_*` (lookup tables).

Two consequences of the timestamps that matter for leakage:

- **Diagnoses are stamped at discharge.** A stay's ICD codes sit at `hadm_discharge_time`, so for the 24-hour
  cutoff tasks the current stay's diagnoses are always after the cutoff and never seen. For readmission
  (cutoff = discharge) they sit exactly AT the cutoff and are excluded by the strict `<` in §7.
- **`MEDS_BIRTH` is at year resolution**, so age enters only through the first `TIME//DELTA` from birth, bucketed.

## 3. MEDS rows → EIC tokens (`eic_config.yaml`, 12 stages)

| # | stage | exactly what happens |
|---|---|---|
| 1 | `custom_time_token` | For each subject, at every unique non-null timestamp, ADD a row `TIME//DELTA//TOKEN` with `numeric_value` = years since the previous unique timestamp. The first timestamp gets `TIME//START//TOKEN` instead. Unit: years. |
| 2 | `count_codes` | `code/n_occurrences`, `code/n_subjects` |
| 3 | `custom_filter_measurements` | DROP every code seen in fewer than **10,000 subjects** — except codes matching `^MEDS_DEATH$ ^MEDS_BIRTH$ ^HOSPITAL_ADMISSION//.* ^HOSPITAL_DISCHARGE//.* ^ED_REGISTRATION//.* ^ED_OUT//.* ^ICU_ADMISSION//.* ^ICU_DISCHARGE//.* ^TIME.*$`, which are kept regardless. Note our ICU codes are `ICU//admit` / `ICU//discharge`, which do NOT match the exempt patterns and survived on frequency alone. |
| 4 | `filter_subjects` | DROP subjects with < 10 events or < 10 measurements |
| 5 | `fit_quantile_binning` | per code with numeric values, on **train** only: quantiles at 0.1 … 0.9 (9 boundaries) |
| 6 | `quantile_binning` | codes WITH a numeric value become `code//_Q_k`, k ∈ 1..10 (**1-indexed; there is no `_Q_0`**). `TIME//DELTA//TOKEN` uses 22 hand-set boundaries instead — 1 min, 5 min, 10 min, 30 min, 1 h, 3 h, 6 h, 12 h, 1 d, 2 d, 4 d, 7 d, 14 d, 30 d, 60 d, 120 d, 1 y, 2 y, 5 y, 10 y, 20 y, 40 y — giving `_Q_1 .. _Q_23`. Codes without a numeric value stay plain. |
| 7 | `reorder_measurments` | within one timestamp, codes are put in a fixed global order derived from `metadata/codes.parquet` |
| 8 | `fit_normalization` | value statistics (unused downstream — the value is already inside the code) |
| 9 | `fit_vocabulary_indices` | assigns integer ids; writes `metadata/codes.parquet` |
| 10 | `normalization` | maps codes → ids |
| 11 | `tokenization` | writes `schemas/` (per subject: list of unique times) and `event_seqs/` (per subject: `code` as list-of-lists — outer = event = one timestamp, inner = ids at that timestamp; `time_delta_days`; `numeric_value`) |
| 12 | `tensorization` | nested ragged tensors (not read by our code; we read `event_seqs` parquet directly) |

Output: **252,477 subjects** (train 201,932 / tuning 25,202 / held_out 25,343), **745,260,121 tokens**,
**3,890 codes** = 2,886 fused `//_Q_k` + 1,004 plain. 405 LAB stems carry deciles.

## 4. Vocabulary as the model sees it (`data.py: load_vocab`)

`itos = ["[PAD]"] + codes.parquet["code"] + [MASK] + demographic/age tokens` → **3,926 ids**:
id 0 = PAD; ids 1..3890 = the MEDS codes (already 1-based, nothing shifted); then `[MASK]` (a real,
learnable token — no longer the PAD id), `GENDER//{F,M,UNKNOWN}`, `RACE//{WHITE,BLACK,HISPANIC,ASIAN,
OTHER,UNKNOWN}`, `MARITAL//{MARRIED,SINGLE,WIDOWED,DIVORCED,UNKNOWN}`, `AGE_T1//0..9`, `AGE_T2//0..9`.
MEDS composition: LAB 3226 · MEDICATION 254 · INGREDIENT_START 61 · DIAGNOSIS 38 · OUTPUT 38 ·
INFUSION_END 30 · INFUSION_START 30 · PROCEDURE 24 · TIME 24 (1 START + 23 DELTA bins) · EMAR 23 ·
MICRO//AB 20 · PHARMACY 18 · TRANSFER_TO 16 · HOSPITAL_DISCHARGE 14 · ORDER 14 · DATETIME 13 ·
HOSPITAL_ADMISSION 9 · INGREDIENT_END 7 · MICRO//CULTURE 7 · SERVICE 7 · MICRO//ORGANISM 6 · OMR 5 ·
HCPCS 2 · ICU 2 · MEDS_BIRTH 1 · MEDS_DEATH 1.

## 5. Per-subject sequence (`data.py: _Corpus`)

`event_seqs` joined with `schemas` on `subject_id`. A subject is a chronological list of events; an event is
one unique timestamp holding the ids recorded then, in the §3.7 order — in practice
`[TIME token, clinical codes…]`. The flattened sequence is the concatenation of events in time order.
Held once per split as a flat `int32` array (~3 GB for train) with per-subject and per-event offsets and
event times; the former Python-list storage was ~30 GB per process and was OOM-killed.

## 6. Every sample: demographic prefix + timeline window (ETHOS)

    [ GENDER//x  RACE//x  MARITAL//x  AGE_T1//d  AGE_T2//d ]  +  [ timeline tokens … ]  +  [ 0-padding ]
      ^-- 5 tokens, attended, never masked/predicted/withheld       ^-- max_len − 5

The prefix is what ETHOS prepends as "patient context" (`static_data.pickle` + `_age_to_tokens`), built
here at dataset time from `patients.csv` / `admissions.csv`, not from MEDS:

- **gender** from `patients.gender`
- **race** = the mode over the subject's admissions, collapsed from 33 strings to 6 exactly as ETHOS's
  `race_priority_mapping`: WHITE / BLACK / ASIAN by prefix, anything `HISPANIC*` → HISPANIC,
  `UNKNOWN` / `UNABLE TO OBTAIN` / `PATIENT DECLINED` → UNKNOWN, everything else → OTHER
- **marital** = mode over admissions; null → UNKNOWN
- **age** at the window's first clinical event, clipped to [0, 99], as two digit tokens (tens, units) —
  ETHOS's `_age_to_tokens`. Birth year = `anchor_year − anchor_age`, the same definition as `MEDS_BIRTH`.
  "First clinical event" = first event in the window after birth + 180 days: `MEDS_BIRTH` is a token in
  *our* timeline (in ETHOS DOB is only in the static context), so without that rule every record that
  fits inside the window would start at birth and report age 0 — 33% of admissions.

Subjects with no admission row get RACE/MARITAL UNKNOWN. Verified: mean age 57.8 (p50 59) at the
readmission cutoff, zero age-0 records.

## 7. Pretraining windows (`data.py: EICSequences`, `pretrain.py`)

**Training = ETHOS random windows.** ETHOS concatenates all timelines and makes every token position a
valid window start (`__len__ = len(tokens) − timeline_size`), so the model sees the whole corpus and long
records contribute proportionally. Ours: a start position is drawn uniformly over all tokens in the split
(token-proportional), the window of `max_len − 5` tokens is clipped to that patient (windows never cross
patients), and one epoch is `n_tokens / window` windows — one corpus pass in expectation. The draw is
`sha256(seed | epoch | index)`, a pure function, so **every architecture sees the same window starts in
the same order**; only the window length differs by model (§8). This replaced "last 512 tokens", which
trained on ~30% of the corpus and is what made the causal models memorise in ten epochs.

**Validation = last window** (`windows="to_end"`) on the tuning split, so val loss is comparable across
epochs. `bs·accum·L = 32,768` tokens per optimizer step for every model. No augmentation of the tokens
themselves — the window choice is sampling, not perturbation.

### 7a. MLM — `bilstm`, `bert`, `bigbird` (`pretrain.py: mask_tokens`)
`sel = Bernoulli(0.15)` over real timeline positions, fresh every batch, **never on the prefix**;
`input[sel] = [MASK]` (learnable, id 3891); attention mask unchanged; `label[sel] = original`, everything
else −100; `CrossEntropyLoss(ignore_index=−100)`.

### 7b. Next-token — `gpt2`, `mistral`, `mamba` (`pretrain.py: shift_tokens`)
`label[t] = x[t+1]`; last position, padding, **and the prefix** = −100 (ETHOS: `y[:context_size] = −100`,
the context is read but not predicted). Causal attention; Mamba is causal by construction.

## 8. Model-specific context and batch (`model.py: CONTEXT / BATCH / ACCUM`)

**Optimiser exception (2026-10-01).** Every model pretrains with AdamW at lr 3e-4 (500 warm-up steps, then
plateau halving) and fp16 autocast, except the Mamba family (bf16, no loss scaler) and, since 2026-10-01,
**bigbird: bf16 and lr 1e-4; bertl: bf16, peak lr 5e-5 and 2,000 warm-up steps.** At 3e-4/fp16 bertl trained normally to step 10k (val 1.44) and then
overflowed (grad-norm 9.8e3, loss scale 65536 → 1, val 5.9), and bigbird never left the unigram floor
(loss 6.69 flat from step 200). 1e-4 is the published rate for both; at 1e-4/bf16 bertl diverged again (grad-norm 7.5e4 at step ~7k, loss 12.8), and
at 5e-5 with a 2,000-step warm-up it trained cleanly to epoch 4 (val 1.1433) and then diverged at epoch 6 (grad-norm
8.8e9). Every model already runs AdamW with β = (0.9, 0.95); a mid-training blow-up at a rate this low points at the
second-moment denominator, so **bertl additionally uses ε = 1e-6** (BERT's own setting; the roster default is 1e-8)
from epoch 5 on, resumed from its epoch-4 checkpoint with fresh optimiser state (2026-10-03). That fourth attempt
reached 1.1209 at epoch 5, regressed at 6 and diverged at 7 (val 7.27); on 2026-10-06 it was restarted from scratch on
the same schedule (lr 5e-5, 2,000 warm-up steps, ε = 1e-6, bf16).

| arch | context | micro-batch × accum | objective | pooling | dropout / wd |
|---|---|---|---|---|---|
| bilstm | 512 | 64 × 1 | MLM | mean | 0.1 / 0.01 |
| bert | 512 (BERT-base) | 64 × 1 | MLM | mean | 0.1 / 0.01 |
| bigbird | 4096 (published; block-sparse, block 64, 3 random blocks) | 2 × 4 | MLM | mean | 0.1 / 0.01 |
| bilstms | 512 (BiLSTM d=512, L=4) | 64 × 1 | MLM | mean | 0.1 / 0.01 |
| bilstml | 512 (BiLSTM d=1024, L=4) | 64 × 1 | MLM | mean | 0.1 / 0.01 |
| berts | 512 (BERT-medium: L=8, d=512) | 64 × 1 | MLM | mean | 0.1 / 0.01 |
| bertl | 512 (BERT-large) | 32 × 2 | MLM | mean | 0.1 / 0.01 |
| gpt2 | 1024 (GPT-2 medium geometry: d=1024, L=24, H=16) | 8 × 4 | next-token | last | 0.3 / 0.10 |
| gpt2s | 1024 (GPT-2 small geometry: d=768, L=12, H=12) | 4 × 8 | next-token | last | 0.3 / 0.10 |
| gpt2l | 1024 (GPT-2 large geometry: d=1280, L=36, H=20) | 4 × 8 (4090) / 8 × 4 (H200) | next-token | last | 0.3 / 0.10 |
| mistrals | 2048 (Mistral arch: d=512, L=8, H=8) | 16 × 1 | next-token | last | 0.3 / 0.10 |
| mistrall | 2048 (Mistral arch: d=1024, L=24, H=16) | 4 × 4 (4090) / 8 × 2 (H200) | next-token | last | 0.3 / 0.10 |
| mambas | 2048 (Mamba: d=512, 16 blocks) | 16 × 1 | next-token | last | 0.3 / 0.10 |
| mambal | 2048 (mamba-370m geometry: d=1024, 48 blocks) | 2 × 8 (4090) / 8 × 2 (H200) | next-token | last | 0.3 / 0.10 |
| mistral | 2048 (ETHOS scale) | 8 × 2 | next-token | last | 0.3 / 0.10 |
| mamba | 2048 (no fixed context; ETHOS scale; 24 blocks = mamba-130m depth) | 4 × 4 | next-token | last | 0.3 / 0.10 |

All at `d=768`, 12 layers (BiLSTM 4). Embedding `padding_idx=0`; BERT/GPT-2 add sinusoidal positions;
Mistral uses RoPE; BigBird learned positions; Mamba none. HF Mistral and Mamba expose no residual dropout,
so it is added by forward hook on each attention/MLP/SSM-mixer branch (x + dropout(f(norm(x))), GPT-2's `resid_pdrop`). Attention masking: `src_key_padding_mask` for the torch
encoders and `attention_mask` for HF, each with an empty-row guard; Mamba takes no mask (right-padding is
harmless to a causal scan). Pooling for classification: bidirectional → mean over real positions, causal →
last real position. Fine-tuning (`EICTask`) uses the same prefix + last `context − 5` tokens strictly
before `prediction_time`, age at the window's first clinical event (ETHOS inference dataset).

**Pretraining provenance (audited 2026-10-06 from every pretrain log).** Clean single runs, never resumed: bilstm,
bilstms, bilstml, bert, berts, gpt2, gpt2s, mistral. Resumed only from `last.pt` across CHPC session ends (an exact
continuation: weights, optimizer moments, scheduler and RNG state restored): gpt2l, mambal, bigbird (bigbird was also
reset from scratch once on 2026-10-01, see above). Resumed from `best.pt` with **fresh optimizer state** once, after a
checkpoint-corruption bug on 2026-09-27, at the learning rate the run had reached (recorded as `resumed_from_best_pt`
in the run's MLflow params): mamba at epoch 22 (converged at 56, best 1.4366 at 51), mambas at epoch 48 (converged at
the LR floor, best 1.4424), mistrals at epoch 46 (no later improvement on its pre-crash best 1.4763; stopped after 5
flat epochs). bertl: see the optimiser-exception note above.

## 9. Missingness input (`perturb.py`, `run_missingness.py`)

Start from the held-out `EICTask` tensors of §7 (full split, no subsampling). Per family, an eligible set
and a per-position score; one keep-mask per (family, dose, draw) with **exact per-patient dose**
(Efraimidis–Spirakis weighted sampling without replacement), rendered as:

- **true removal** (`apply_event_remove`, code name `event`) — the withheld tokens are deleted as events: an
  event left with no content vanishes with its time-gap token, and the time-gap token of every surviving
  event is re-binned from the timestamps. `EICTask(..., with_events=True)` returns each event's boundaries
  and timestamp for this. Tail-padded, mask recomputed; the window is not refilled.

Per-cell RNG seed is `sha256("fam|frac|beta|draw")` — Python's `hash()` of a str tuple is randomised per
process and was making draws non-reproducible. Invariants (`assert_event_cell`) abort the run: realized dose
within tolerance of requested; kept content unchanged and in order; the null arm must reproduce baseline in
every cell.

## 10. What the model never sees

- a numeric value as a number — only its train-set decile, fused into the code
- absolute time or dates — only bucketed deltas between consecutive timestamps
- insurance, language, or any demographic beyond the five-token prefix (sex, race, marital, age)
- any token after `prediction_time`
- the free-text `medication` string in EMAR, or the drug in `pharmacy` — mapped to `event_txt` and `route`
  respectively to keep cardinality bounded
- `drgcodes`, `emar_detail`, `poe_detail`, `provider`, `caregiver`
