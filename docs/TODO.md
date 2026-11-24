# TODO: run the remaining missingness families on every model (queued 2026-10-08)

Requested by Sud on 2026-10-08: once the current ladder (pretrains, finetunes,
sweeps for all 15 models) is finished, run the families that were designed but never implemented, on
all models, on CHPC. Nothing below is started. Order is by cost and by how much design is still open.

## What runs today

`run_missingness.py: FAMILIES` = A1, A3, B1, C1, C2, C3 (6 families × 6 doses = 36 cells
per model and task). Everything below is additional. Results must be **appended** to the existing
`data/results/missingness/<arch>/<task>.event.json` (cells keyed by family/protocol/frac/beta), so the run
script needs a `--families` argument and a skip-if-cell-exists rule before any of this starts;
otherwise the finished cells get recomputed.

## 1. A2 and B2: visit-level removal — DONE 2026-10-08 (implemented and running)

The visit families A2, B2 and B2o are in `run_missingness.py` (`--families A2,B2,B2o`, with `--min_visits 3`
for doses graded in visits). Under true removal they write `<task>.visits_event.json`. The October runs
(`<task>.visits.json`) used token removal; that variant is not included in this branch.

## 2. C2 matched-budget twin

- Same code, random decile: withhold the same count from the same lab stems as C2, chosen at random
  within the stem. Holds code marginal, count and positions fixed so only value-rank differs. Without it
  C2 partly measures code identity and embedding coverage, which differ across architectures.
- Cells: 1 twin × 6 doses = 6 per (model, task).

## 3. β sweep with a breakdown value

- For C1 and C2 at fixed dose 0.5: `β ∈ {0, 1, 2, 4, 8}`; report the β at which the
  conclusion (the model ranking) changes. Design rationale in
  `docs/missingness.md` §2.
- Cells: 2 families × 4 new β values = 8 per (model, task).

## Cost

Per (model, task): 6 (twin) + 8 (β sweep) = 14 new cells against the 36 that exist (the visit families add 18).
A cell takes 17 to 22 minutes for a 100M model on an H200 (full held-out split),
before the large-model multiplier. If the test set is
subsampled for the large models (open question in `docs/method.md` §12), apply the same subsample here.

## Mechanics when the time comes

1. Implement 2 and 3 in `perturb.py` / `run_missingness.py` with `--families`, the skip-existing
   rule, and the relaxed-inside-span assertion; run the null-arm check for every new family.
2. Validate on one small model (berts) on the lab box: realized fractions, mechanism-fired histograms,
   monotonicity guard, one worked record per family added to `docs/missingness.md` §6.
3. Add a `sweep2 <arch> <gpu> <tasks>` job kind (or extend `sweep.sh` to pass `--families`) so the CHPC
   supervisors run the new cells for every model whose main sweep exists, shipping the merged JSON back.
4. Figures: the main tree draws no figure of the sweep. The family panels that existed read the
   token-removal files and were removed with it.
