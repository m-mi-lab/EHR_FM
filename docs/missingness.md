# Simulating Rubin's regimes on MEDS-Torch tokenized data

Operates on the EIC stream from `docs/tokenization.md`: one fused symbol per
observation, `code//_Q_k`, `k` the per-code decile.

## 1. Where we intervene

Directly on the **tensorized** output, not as a `MEDS_transforms` stage. After
tensorization each batch carries:

    batch["code"]   int ids, one per observation (EIC passes these raw to the backbone)
    batch["mask"]   validity/attention mask
    time_delta_days, static_indices, static_values

The perturbation rewrites `code` and `mask`, and reads each event's boundaries and
timestamp to do it. One reason this is the right seam:

1. **The vocabulary must be byte-identical across every cell.**
   `fit_vocabulary_indices` runs mid-chain; perturbing before it refits the
   vocabulary per cell and silently changes token ids between arms.

Given `keep`, the withheld tokens are deleted as events (`perturb.apply_event_remove`):
an event left with no content vanishes with its time-gap token, and the time-gap tokens
of the surviving events are re-binned from the timestamps. The window is not refilled.

**Decile lookup, built once at startup** — this is the concrete payoff of fused
tokenization:

    decile_of[i] = int(m.group(1)) if (m := re.search(r"//_Q_(\d+)$", vocab[i])) else None

`None` marks tokens carrying no value. Every value-conditioned rule below is then
a table lookup on the token id.

**Content vs scaffolding** is taken from the EIC config's own list rather than
invented (`custom_filter_measurements.additional_codes`):

    ^MEDS_DEATH$  ^MEDS_BIRTH$  ^HOSPITAL_ADMISSION//.*  ^HOSPITAL_DISCHARGE//.*
    ^ED_REGISTRATION//.*  ^ED_OUT//.*  ^ICU_ADMISSION//.*  ^ICU_DISCHARGE//.*  ^TIME.*$

Content = everything else. Scaffolding is protected except where a family
explicitly targets it (A2, B2).

## 2. One operator, three regimes

Use the standard amputation form (Schouten, Lugtig & Vink 2018 — this is
`mice::ampute`, so use its vocabulary rather than coining new terms):

    logit P(R_j = 0 | Z_j) = alpha + beta * s(Z_j)

`R_j` retention, `s(.)` a standardised score in `[-1,1]`, `beta` the contrast
strength. **The regime is set entirely by what `Z` is allowed to be.** `ampute`'s
`RIGHT`/`LEFT`/`MID`/`TAIL` types are the standard `s` shapes.

Three properties, all load-bearing:

- **`beta = 0` is MCAR, nested.** MCAR is not a separate arm, it is this family at
  zero contrast — so the mechanism effect is a contrast in one parameter at fixed
  dose, not a comparison of two differently-built arms.
- **Positivity holds by construction.** `expit` is strictly inside `(0,1)`, so no
  token has retention probability exactly zero. Deterministic families (drop *all*
  labs) violate this; they are classified but not ignorable, and positivity is a
  matter of identifiability/estimability, not ignorability (Zivich, Cole &
  Westreich 2022). Keep deterministic variants only as declared boundary cases.
- **Marginal calibration is a 1-D solve.** For target withheld fraction `f` and
  empirical score shares `q_z`: `sum_z q_z * expit(alpha + beta*s(z)) = f`. The
  left side is strictly increasing in `alpha`; bisect on `[-8,8]` exactly as
  `ampute` does. Fix `beta`, solve `alpha`.

**Sweep `beta` at fixed `f`, and report a breakdown value** — the `beta` at which
the conclusion changes. A single hand-chosen `beta` is not merely weak, it is
contrary to consensus practice for non-identifiable mechanisms (Scharfstein,
Rotnitzky & Robins 1999; NRC 2010). This is forced by Molenberghs et al. (2008):
*every MNAR model has an MAR counterpart with identical observed-data fit*, so no
single simulated mechanism can be argued to be the right one.

## 3. Families

Nine families, balanced across regimes. Every non-MCAR family ships with a
**matched-budget MCAR twin** (same per-patient count, random content), because the
raw drop is dominated by dose and only the excess over the twin is interpretable.

### MCAR — depends on nothing

| id | family | `Z` | operation |
|---|---|---|---|
| **A1** | graded random content | constant | withhold `f` of content tokens uniformly. **The reference curve.** |
| **A2** | random whole-visit | constant | withhold every token spanned by a randomly chosen visit; overrides scaffolding protection |
| **A3** | random burst / run | constant | two-state Markov (Gilbert 1960): mean run length `L` decoupled from marginal `f` via `p = 1/L` |

A3 fills a real gap: `ampute` assumes i.i.d. rows and **no longitudinal extension
exists**, so run-structured missingness is a methodological contribution, not a
nicety (cf. Mitra et al. 2023 on structured missingness).

### MAR — depends on observed data only

| id | family | `Z` | operation |
|---|---|---|---|
| **B1** | per-modality | observed code class (lab / med / proc) | graded withholding within one modality |
| **B2** | recency-ordered visit | observed visit index | withhold visits weighted by recency (oldest-first / newest-first) |


### MNAR — depends on the unobserved value or the latent state generating it

| id | family | `Z` | operation |
|---|---|---|---|
| **C1** | cadence / repeat suppression | latent clinician concern | per `(subject, code stem, day)` keep the first `k`, withhold the rest. Sweep `k in {1,2,3,4,inf}` |
| **C2** | low-value withholding | the value itself | withholding probability decreasing in the decile over ALL deciled lab tokens (`ampute` type `LEFT`). **219.9 eligible tokens/patient, 67.3% of content.** |


**C1 is MNAR via the latent state, not the value** — whether a repeat is ordered
depends on the clinician's posterior over instability, which also drives the
outcome. Cap on the **code stem, ignoring the decile suffix**: otherwise a patient
whose potassium swings across the day has each decile counted as a separate first
occurrence and escapes the cap, which is backwards.

**What C2 was, and why it changed.** C2 was a limit-of-detection arm restricted
to bottom-decile tokens of four curated assays (troponin `LAB//51003`, D-dimer
`LAB//50915`/`LAB//51196`, lactate `LAB//50813`). It had an honest clinical
mechanism and no statistical power: measured on held-out MIMIC-IV it reached
**1.70 tokens per patient, 0.53% of content**, so withholding 97% of its eligible
pool moved AUROC by **0.001** on all three tasks. Two further defects surfaced:
BNP and procalcitonin have **zero matches** in `d_labitems` and do not exist in
MIMIC-IV; and per-patient rounding `round(frac x 1.7)` withheld nothing for most
patients, so a requested 0.10 realized 0.015.

It is now value-conditioned withholding across all deciled lab tokens, which
reaches 67.3% of content.

**This is a sensitivity probe, not a clinical simulation.** Blanket
decile-conditioned withholding is not: `_Q_5` sodium is 140 (normal) while `_Q_1`
glucose is hypoglycaemia, a critical value that triggers *more* data; the coupling
between decile and clinical meaning differs per analyte. Real inpatient ordering
self-masks on the latent state — the decision not to look is made *before* the
value exists. Do not label C2 as censoring. The claim it supports is narrow:
performance under value-dependent withholding at matched marginal rate and known
contrast `beta`.

### The control that makes value-conditioned arms credible

For C2, the twin is **same code, random decile**: withhold the same count from
the same code, chosen at random. This holds code marginal, count and positional
distribution fixed so **only value-rank differs**. Without it the arm measures code
identity and calls it value-dependence — and in EIC specifically, deleting a fixed
`_Q_k` slice removes a fixed vocabulary slice, so it partly measures embedding
coverage, which differs across the six architectures.

## 4. Assertions

Abort, do not warn. Each corresponds to a defect already found in a prior harness.

1. **Realized fraction** matches request, computed over that family's own
   denominator.
2. **Kept content intact.** The withheld content is absent, the kept content is present in
   its order, the prefix is unchanged, and only time-gap tokens differ from the kept sequence
   (`perturb.assert_event_cell`).
3. **Pooling index validity.** `pool_idx == mask.sum()-1` **and**
   `code[pool_idx] != PAD`. This is the assertion that catches attention-mask
   inflation.
4. **Null-arm bit-exactness.** At `f=0` every family reproduces the clean
   baseline *logits* to 1e-6 — not the same AUROC. AUROC is a rank statistic and
   will hide a real divergence.
5. **Weight identity.** `sha256(state_dict)` equal across every arm in a cell. Path
   equality was never the invariant.
6. **Vocabulary hash** equal to the clean run's.
7. **Mechanism fired.** For C1-C2, the withheld score histogram differs from the
   retained one in the predicted direction. Without this, a silently-failed MNAR
   cell is MCAR wearing a label.
8. **Calibration held across the sweep.** Same realized `f` at every `beta`, else
   mechanism is confounded with dose.
9. **Monotonicity guard.** Spearman(dose, AUROC) <= 0 per (model, task, family);
   a positive value beyond draw noise is quarantined, not plotted.

**Denominators differ and must be stated.** `f` is a fraction of that family's
eligible pool: all content (A1), one modality (B1), numeric tokens (C2), visits
(A2/B2). An `f=0.5` cell is not the same quantity across families. Report
realized `rho_tok` (fraction of content withheld) alongside, and use A1 as the
matched-dose reference curve rather than asserting a shared axis.

Record per cell: family, regime, `Z`, protocol, `f`, `beta`, solved `alpha`,
realized fraction, realized score histogram, seed, checkpoint sha, vocabulary
hash, `n`.

## 5. What the taxonomy licenses

Not a ladder of harm. The precise statement is Rockenschaub et al. (2024),
Theorem 1: Bayes predictors are invariant across environments **iff missingness is
ignorable in both**. So:

> if both the source mechanism and our perturbation are ignorable, every point of
> ΔAUROC is an **approximation failure of the network**, not irreducible loss; if
> the perturbation is non-ignorable, part of the loss is irreducible and no model
> could have avoided it.

That decomposition is the honest, non-decorative use of Rubin here, and it is what
the MCAR/MAR-vs-MNAR contrast actually buys.

Two caveats to state rather than bury:

- **We study `M o M_real`, not `M`.** The perturbation is *overlaid* on data that is
  already the output of a real, unknown, near-certainly non-ignorable mechanism
  (Qian et al. 2024 on augmentation vs overlaying). The source-side condition of
  Theorem 1 may already fail before we perturb anything.
- **Mechanisms may converge.** Schouten & Vink (2021) show that where correlations
  are low **or very substantial**, strictly different mechanisms yield equivalent
  inferences — and EHR records (repeat labs, linked orders, code co-occurrence)
  sit in the high-correlation regime. Lead *with* this: if MCAR and MNAR arms
  converge at matched dose, that is the published expectation, and the interesting
  quantity is the recoverability governing it.


## 6. Worked examples

Patient-level MIMIC-IV records (identifiers, dates, token sequences) are not reproduced in this
public repository, per the PhysioNet credentialed-access data use agreement.


## 7. References

Rubin 1976 *Biometrika* 63(3):581-92 · Schouten, Lugtig & Vink 2018 *JSCS*
88(15):2909-30 (`mice::ampute`) · Schouten & Vink 2021 *Sociol Methods Res*
50(3):1243-58 · Molenberghs, Beunckens, Sotto & Kenward 2008 *JRSS-B* 70(2):371-88
· Scharfstein, Rotnitzky & Robins 1999 *JASA* 94(448):1096-120 · NRC 2010
*Prevention and Treatment of Missing Data in Clinical Trials* · Rockenschaub et al.
2024 arXiv:2406.16484 · Zhou, Balakrishnan & Lipton 2023 *AISTATS* PMLR 206:9577
· Qian et al. 2024 arXiv:2405.17508 · Sun & Tchetgen Tchetgen 2018 *JASA*
113(521):369-79 · Zhou, Little & Kalbfleisch 2010 *Statist Sci* 25(4):517-32 ·
Mitra et al. 2023 *Nat Mach Intell* 5:13-23 · Zivich, Cole & Westreich 2022
arXiv:2207.05010 · Gilbert 1960 *BSTJ* 39(5):1253-65 · Che et al. 2018 *Sci Rep*
8:6085 (GRU-D, informative missingness) · Agniel, Kohane & Weber 2018 *BMJ* 361:k1479
· Rusanov et al. 2014 *BMC Med Inform Decis Mak* 14:51 · Ramadan et al. 2025
*JAMA Netw Open* 8(12):e2550454 (ICD label leakage in MIMIC)


## 8. Measured facts (held-out MIMIC-IV, `data/`, rebuilt 2026-09-08)

Everything below is measured, not assumed.

| | |
|---|---|
| MIMIC-IV tables mapped | 21 of 26 |
| MEDS rows / subjects | 735,696,749 / 364,627 |
| tokenized patients | 252,477 (train 201,932 / tuning 25,202 / held_out 25,343) |
| tokens | 745,260,121 |
| vocabulary | 3,890 (2,886 fused `//_Q_k` + 1,004 plain) |
| pre-cutoff tokens / admission (mortality) | median 986 · p25 259 · p75 2,920 · mean 4,102; the model sees the last 512 |
| mean content tokens / patient | 2,995 |
| mean LAB tokens / patient | 2,154 |
| structural tokens | 74 |

**Vocabulary prefixes.** LAB 3226 · MEDICATION 254 · INGREDIENT_START 61 ·
DIAGNOSIS 38 · OUTPUT 38 · INFUSION_END 30 · INFUSION_START 30 · PROCEDURE 24 ·
TIME 24 · EMAR 23 · MICRO//AB 20 · PHARMACY 18 · TRANSFER_TO 16 ·
HOSPITAL_DISCHARGE 14 · ORDER 14 · DATETIME 13 · HOSPITAL_ADMISSION 9 ·
INGREDIENT_END 7 · MICRO//CULTURE 7 · SERVICE 7 · MICRO//ORGANISM 6 · OMR 5 ·
HCPCS 2 · ICU 2 · MEDS_BIRTH 1 · MEDS_DEATH 1.

**405 LAB stems carry deciles** (labevents and chartevents both map to `LAB//`),
plus `TIME//DELTA//TOKEN`. Medications, orders, procedures and diagnoses have no
numeric value, so every value-conditioned family is necessarily lab-only.

**Quantile bins are 1-indexed** (`_Q_1` .. `_Q_10`); the vocabulary contains zero
`_Q_0` entries. An earlier C2 tested `decile == 0` and therefore never fired.

**Structural set.** `MEDS_BIRTH MEDS_DEATH TIME TRANSFER_TO ICU HOSPITAL_ADMISSION
HOSPITAL_DISCHARGE SERVICE` (74 tokens). Before `hosp/admissions` and
`hosp/services` were mapped the corpus had no explicit admission event at all and
encounter structure was carried by `TRANSFER_TO` and `ICU` alone (18 tokens).

**Provenance.** The extraction used until 2026-09-08 silently dropped exactly
400,000,000 rows of `icu/chartevents` (three 200M chunks all written to one file)
and mapped 10 of 26 tables. Every number computed on it is void.

## 9. Ninety percent gone: what the record becomes, and what the finetuned model does with it

*Token removal, deprecated on 2026-11-16; that material is not included in this branch.*

### 9.1 The window before and after

*Token removal, deprecated on 2026-11-16; that material is not included in this branch.*

### 9.2 A record, clean and removed

*Token removal, deprecated on 2026-11-16; that material is not included in this branch.*

### 9.3 What the finetuned model does with it at inference

*Token removal, deprecated on 2026-11-16; that material is not included in this branch.*

### 9.4 A full 512-token window, clean and at 90 %

*Token removal, deprecated on 2026-11-16; that material is not included in this branch.*

### 9.5 The protected set, token by token

Every family draws its keep-mask from content tokens only. The 110 vocabulary entries below
(`perturb.STRUCTURAL`, matched by prefix, plus `[PAD]`) are never withheld by a family in any
cell. The remaining 3,816 tokens — LAB,
MEDICATION, EMAR, PHARMACY, ORDER, DIAGNOSIS, PROCEDURE, MICRO, INFUSION, INGREDIENT, OUTPUT, OMR,
DATETIME, HCPCS — are content.

| group | n | tokens |
|---|---|---|
| TIME | 24 | `TIME//START//TOKEN`; `TIME//DELTA//TOKEN//_Q_1` … `_Q_23` (elapsed-time buckets between consecutive events) |
| TRANSFER_TO | 16 | `//ED//Emergency Department`; `//admit//` Discharge Lounge, Emergency Department Observation, Labor & Delivery, Med/Surg, Medicine, Medicine/Cardiology, Neurology; `//transfer//` Cardiac Vascular Intensive Care Unit (CVICU), Discharge Lounge, Med/Surg, Medicine, Medicine/Cardiology, Neurology, PACU; `//discharge//UNKNOWN` |
| HOSPITAL_DISCHARGE | 14 | disposition: HOME, HOME HEALTH CARE, SKILLED NURSING FACILITY, REHAB, HOSPICE, DIED, AGAINST ADVICE, ACUTE HOSPITAL, CHRONIC/LONG TERM ACUTE CARE, HEALTHCARE FACILITY, PSYCH FACILITY, ASSISTED LIVING, OTHER FACILITY, UNK |
| HOSPITAL_ADMISSION | 9 | admission type: EW EMER., DIRECT EMER., URGENT, ELECTIVE, OBSERVATION ADMIT, EU OBSERVATION, DIRECT OBSERVATION, AMBULATORY OBSERVATION, SURGICAL SAME DAY ADMISSION |
| SERVICE | 7 | CMED, MED, NMED, NSURG, OBS, ORTHO, SURG |
| ICU | 2 | `ICU//admit`, `ICU//discharge` |
| MEDS_BIRTH, MEDS_DEATH | 2 | |
| demographic prefix | 34 | `GENDER//` F, M, UNKNOWN · `RACE//` WHITE, BLACK, HISPANIC, ASIAN, OTHER, UNKNOWN · `MARITAL//` MARRIED, SINGLE, WIDOWED, DIVORCED, UNKNOWN · `AGE_T1//0`–`9` · `AGE_T2//0`–`9` |
| special | 2 | `[PAD]`, `[MASK]` |

So under every cell the model keeps who the patient is (prefix) and where they were (ED arrival,
admission type, service, ward and ICU transfers, prior dispositions). The time-gap tokens follow the
events: the one at the head of an event that loses all its content disappears with the event, and the
others are recomputed from the timestamps.

Two consequences. `HOSPITAL_DISCHARGE//DIED` and `MEDS_DEATH` are protected but cannot occur in a
mortality window, which ends strictly before `prediction_time`. And the transfer, admission-type,
service and prior-disposition tokens carry prognosis on their own. The demographic prefix and the
encounter markers are never the thing being tested; the families test what the model does when the
clinical content between them thins out.
