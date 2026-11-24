"""Run missingness families against a finetuned model under true removal (perturb.apply_event_remove).

One keep-mask per (family, frac, beta, draw). Per-patient dose is exact, so mechanism is
never confounded with dose.
"""
import argparse, json, hashlib, os, numpy as np, torch, mlflow
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import rankdata

NBOOT = 2000


def _auc(y, s):
    """Mann-Whitney AUROC with tie-averaged ranks; equals roc_auc_score exactly."""
    r = rankdata(s)
    npos = y.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0: return float("nan")
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def boot_vals(y, p, idx_matrix):
    """Per-replicate AUROCs. Returned rather than summarised so that a paired
    delta-CI can be formed against the baseline cell's replicates: the effect
    being claimed is a within-patient difference, and its variance is far below
    that of either endpoint."""
    out = np.full(len(idx_matrix), np.nan)
    for j, idx in enumerate(idx_matrix):
        yb = y[idx]
        if yb.min() == yb.max(): continue
        out[j] = _auc(yb, p[idx])
    return out


def boot_ci(y, p, idx_matrix):
    """Patient-level bootstrap CI from the SAME scores as the point estimate.

    One shared index matrix across every cell, so a band can never belong to a
    different experiment than its curve, and deltas between cells are
    computable per replicate."""
    vals = np.empty(len(idx_matrix)); k = 0
    for idx in idx_matrix:
        yb = y[idx]
        if yb.min() == yb.max(): continue
        vals[k] = _auc(yb, p[idx]); k += 1
    if k == 0: return float("nan"), float("nan")
    lo, hi = np.percentile(vals[:k], [2.5, 97.5])
    return float(lo), float(hi)
from data import EICTask, load_vocab
from model import build, CONTEXT
from perturb import build_lookups, keep_mask, visit_spans, visit_keep_mask
from perturb import apply_event_remove, assert_event_cell, time_ids
from data import PREFIX_LEN

FAMILIES = {
    # id: (regime, eligible_kind, score_kind, beta)
    "A1": ("MCAR", "content",   "none",        0.0),
    "A3": ("MCAR", "content",   "none",        0.0),   # burst: contiguous runs
    "B1": ("MAR",  "lab",       "none",        0.0),   # per-modality (LAB)
    "C1": ("MNAR", "content",   "repeat",      4.0),   # cadence: repeats first
    "C2": ("MNAR", "lab_valued", "low_value",  4.0),   # low-value withholding
    "C3": ("MNAR", "micro_ab",   "reflex",      4.0),   # reflex: culture -> sensitivity
}
# Visit-level families: whole PRIOR visits withheld, scaffold included. Scored on the multi-visit subset
# and written to <task>.visits_event.json, so they never share a file with the token families.
# See perturb.visit_spans for the visit definition.
VISIT_FAMILIES = {
    # id: (regime, eligible_kind, order, beta)
    "A2":  ("MCAR", "visits", "none",    0.0),   # random prior visits; matched-count twin of B2
    "B2":  ("MAR",  "visits", "recency", 4.0),   # most recent prior visit withheld first
    "B2o": ("MAR",  "visits", "oldest",  4.0),   # oldest prior visit withheld first
}
ALL_FAMILIES = {**FAMILIES, **VISIT_FAMILIES}
BOTTOM_DECILE = 1        # MEDS-Torch quantile bins are 1-indexed
CURATED = {"LAB//51003", "LAB//50915", "LAB//51196", "LAB//50813"}

def build_masks(itos):
    dec, struct, stems = build_lookups(itos)
    is_lab = np.array([s.startswith("LAB//") for s in stems])
    curated = np.array([s in CURATED for s in stems])
    # microbiology reflex pair: a sensitivity panel exists only where a culture
    # grew an organism, so the parent/child link is recorded in the data itself
    is_org = np.array([s.startswith("MICRO//ORGANISM") for s in stems])
    is_ab  = np.array([s.startswith("MICRO//AB") for s in stems])
    is_disch = np.array([s.startswith("HOSPITAL_DISCHARGE") for s in stems])   # visit boundaries (A2/B2)
    return dec, struct, stems, is_lab, curated, is_org, is_ab, is_disch

def eligible_and_score(fam, codes, valid, dec, struct, is_lab, curated, stems,
                       is_org=None, is_ab=None):
    if fam in ("A1", "A3"):
        return valid & ~struct[codes], None
    if fam == "B1":
        return valid & is_lab[codes], None
    if fam == "C1":                      # repeats of the same stem
        seen, rep = {}, np.zeros(len(codes))
        for i, c in enumerate(codes):
            if not valid[i] or struct[c]: continue
            s = stems[c]; rep[i] = seen.get(s, 0); seen[s] = seen.get(s, 0) + 1
        return valid & ~struct[codes], np.clip(rep, 0, 5) / 5.0 * 2 - 1
    if fam == "C2":
        # Value-conditioned withholding over ALL deciled lab tokens: withholding
        # probability decreases with the decile, i.e. mice::ampute type="LEFT".
        #
        # This REPLACES a limit-of-detection arm restricted to four curated
        # assays. That version was honest about its mechanism but reached only
        # 1.7 tokens per patient (0.53% of content), so withholding 97% of its
        # eligible pool moved AUROC by 0.001 -- no power to detect anything.
        # This reaches ~10% of content (31.6 bottom-decile lab tokens/patient).
        #
        # Name it for what it is: a value-conditioned sensitivity probe, NOT a
        # simulation of assay censoring. Decile does not equal clinical
        # normality -- _Q_1 sodium is 128 mEq/L, profoundly abnormal and always
        # reported -- so this arm tests MNAR-ness, not a clinical mechanism.
        d = dec[codes]
        elig = valid & is_lab[codes] & (d >= 1)
        sc = np.where(d >= 1, (5.5 - d) / 4.5, 0.0)   # low decile -> more likely withheld
        return elig, sc
    if fam == "C3":
        # Withhold the CHILD (antibiotic sensitivity) of a culture that grew.
        # Score is +1 for a sensitivity token preceded by an organism token in
        # the same record, -1 otherwise, so withholding concentrates on panels
        # whose parent is present -- the reflex the record actually encodes.
        elig = valid & is_ab[codes]
        seen_org = np.cumsum(valid & is_org[codes]) > 0
        sc = np.where(seen_org, 1.0, -1.0)
        return elig, sc
    raise ValueError(fam)

def burst_mask(codes, valid, elig, frac, rng, mean_run=5):
    """Withhold EXACTLY round(frac*n_eligible) tokens in contiguous runs.

    Gilbert two-state channel: mean run length is decoupled from the marginal
    rate, so clustering and dose vary independently. The previous version
    incremented its counter on empty retries instead of dropping tokens, so it
    silently undershot the requested dose -- caught by the realized-fraction
    assertion.
    """
    keep = valid.copy()
    idx = np.flatnonzero(elig)
    if idx.size == 0: return keep
    k = int(round(frac * idx.size))
    if k <= 0: return keep
    k = min(k, idx.size)
    dropped = 0
    guard = 0
    while dropped < k and guard < 10000:
        guard += 1
        start = int(rng.integers(0, idx.size))
        run = max(1, int(rng.geometric(1.0 / mean_run)))
        run = min(run, k - dropped)
        sel = idx[start:start + run]
        new_drop = sel[keep[sel]]
        if new_drop.size:
            keep[new_drop] = False
            dropped += int(new_drop.size)
    if dropped < k:                      # fall back to filling the remainder
        rest = idx[keep[idx]]
        need = k - dropped
        if rest.size:
            keep[rng.choice(rest, size=min(need, rest.size), replace=False)] = False
    return keep


@torch.no_grad()
def score_cell(m, ds, dev, fam, frac, beta, draw, lookups, bs=64, max_len=512, indices=None):
    dec, struct, stems, is_lab, curated, is_org, is_ab, is_disch = lookups
    # hashlib, NOT hash(): Python randomises str hashes per process
    # (PYTHONHASHSEED), so hash(("A1", ...)) differs between runs and the
    # draws were not reproducible. sha256 of the repr is stable everywhere.
    seed_src = f"{fam}|{round(frac, 3)}|{round(beta, 3)}|{draw}".encode()
    rng = np.random.default_rng(int(hashlib.sha256(seed_src).hexdigest()[:8], 16))
    ys, ps, realized = [], [], []
    buf_c, buf_m, buf_y = [], [], []
    def flush():
        if not buf_c: return
        c = torch.from_numpy(np.stack(buf_c)).to(dev)
        mm = torch.from_numpy(np.stack(buf_m)).to(dev)
        ps.append(torch.sigmoid(m.forward_clf(c, mm)).cpu().numpy())
        ys.append(np.array(buf_y)); buf_c.clear(); buf_m.clear(); buf_y.clear()
    for i in (range(len(ds)) if indices is None else indices):
        s = ds[i]; codes = s["code"].numpy(); valid = s["mask"].numpy().astype(bool)
        if fam in VISIT_FAMILIES:                      # whole prior visits
            order = VISIT_FAMILIES[fam][2]
            spans = visit_spans(codes, valid, is_disch)
            keep, k = visit_keep_mask(valid, spans, frac=frac, beta=beta,
                                      order=(None if order == "none" else order), rng=rng)
            yield_codes, yield_mask, _ = apply_event_remove(codes, keep, max_len, s["ev_start"], s["ev_time_us"], s["t_prev_us"], TIME_IDS[0], TIME_SET[0], PREFIX_LEN)
            if frac > 0 and spans: realized.append(float(k) / len(spans) if spans else 0.0)
            buf_c.append(yield_codes); buf_m.append(yield_mask); buf_y.append(s["label"].item())
            if len(buf_c) == bs: flush()
            continue
        elig, sc = eligible_and_score(fam, codes, valid, dec, struct, is_lab, curated, stems,
                                   is_org, is_ab)
        if frac > 0 and elig.sum() > 0:
            if fam == "A3":
                keep = burst_mask(codes, valid, elig, frac, rng)
            else:
                keep = keep_mask(codes, valid, frac=frac, beta=beta, score=sc,
                                 eligible=elig, rng=rng)
        else:
            keep = valid.copy()
        yield_codes, yield_mask, _ = apply_event_remove(codes, keep, max_len, s["ev_start"], s["ev_time_us"], s["t_prev_us"], TIME_IDS[0], TIME_SET[0], PREFIX_LEN)
        if frac > 0 and elig.sum() > 0:
            realized.append(assert_event_cell(codes, valid, elig, keep, yield_codes, TIME_SET[0], PREFIX_LEN, frac))
        buf_c.append(yield_codes); buf_m.append(yield_mask); buf_y.append(s["label"].item())
        if len(buf_c) == bs: flush()
    flush()
    y = np.concatenate(ys); p = np.concatenate(ps)
    return (float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
            float(average_precision_score(y, p)),
            float(np.mean(realized)) if realized else 0.0, y, p)

PROTOCOL = "event"   # written into every row: the code name of true removal
TIME_IDS = [None]   # bin -> TIME//DELTA token id
TIME_SET = [None]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True,
                    help="architecture name; results are written per model")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=str(Path(os.environ.get("ICLR_ROOT", Path(__file__).resolve().parents[2])) / "data" / "results" / "missingness"))
    ap.add_argument("--limit", type=int, default=None,
                    help="patients to score; default = the whole held-out split. "
                         "At 4000 the CI half-width on mortality is ~0.029, wider "
                         "than most effects being measured.")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=None, help="default: model.CONTEXT[arch]")
    ap.add_argument("--tf32", type=int, default=0,
                    help="1 = TF32 tensor-core matmuls (used for the H200 sweeps); 0 = plain fp32 as in the baseline sweeps")
    ap.add_argument("--families", default=",".join(FAMILIES),
                    help="comma list; the six token families by default, or the visit families A2,B2,B2o "
                         "(a run is either all-token or all-visit)")
    ap.add_argument("--suffix", default="",
                    help="output name <task>.<suffix>.json; default: event for the token families, visits_event for the visit families")
    ap.add_argument("--min_visits", type=int, default=1,
                    help="visit families: score only records with at least this many complete prior visits in the window "
                         "(3 makes the six doses graded in visits; 1 is a with/without contrast)")
    a = ap.parse_args()
    fams = {f: ALL_FAMILIES[f] for f in a.families.split(",")}
    visit_run = all(f in VISIT_FAMILIES for f in fams)
    assert visit_run or not any(f in VISIT_FAMILIES for f in fams), "mix of token and visit families"
    if not a.suffix: a.suffix = "visits_event" if visit_run else "event"
    out_name = f"{a.task}.{a.suffix}.json"
    if a.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    if a.max_len is None: a.max_len = CONTEXT[a.model]
    out = Path(a.out)/a.model; out.mkdir(parents=True, exist_ok=True)
    itos, stoi = load_vocab(); V = len(itos)
    TIME_IDS[0], TIME_SET[0] = time_ids(stoi)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert (Path(a.ckpt).parent / "result.json").exists(), f"unfinished finetune: no result.json next to {a.ckpt}"
    ck = torch.load(a.ckpt, map_location="cpu")
    m = build(ck.get("arch", a.model), V, d=ck["d"],
              layers=ck.get("layers", 4), heads=ck.get("heads", 8)).to(dev); m.load_state_dict(ck["model"]); m.eval()
    sha = hashlib.sha256(str(sorted(ck["model"].keys())).encode()).hexdigest()[:12]
    ds = EICTask(a.task, "held_out", a.max_len, a.limit, with_events=True)
    lookups = build_masks(itos)
    _rng = np.random.default_rng(12345)
    indices, extra = None, {}
    if visit_run:   # the multi-visit subset: records with >= 1 complete prior visit inside the window
        is_disch = lookups[7]
        n_vis = np.array([len(visit_spans(ds[i]["code"].numpy(), ds[i]["mask"].numpy().astype(bool), is_disch))
                          for i in range(len(ds))])
        indices = [int(i) for i in np.flatnonzero(n_vis >= a.min_visits)]
        assert len(indices) > 0, f"no record has {a.min_visits} complete prior visits inside the window"
        extra = {"subset": f"min_visits_{a.min_visits}", "min_visits": int(a.min_visits), "n_all": int(len(ds)),
                 "visits_median": float(np.median(n_vis[indices])), "visits_mean": float(np.mean(n_vis[indices]))}
        print(f"multi-visit subset (>= {a.min_visits} complete prior visits in the window): {len(indices)} of {len(ds)} "
              f"records (median {extra['visits_median']:.0f}, mean {extra['visits_mean']:.2f})", flush=True)
    n_pat = len(ds) if indices is None else len(indices)
    BOOT_IDX = _rng.integers(0, n_pat, size=(NBOOT, n_pat))   # shared across all cells
    FRACS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]
    rows, base, base_boot = [], {}, {}
    # Per-cell checkpoint: a sweep killed by a session limit resumes from the cells it finished. Cells are
    # order-independent (every draw is seeded from (family, dose, beta, draw); BOOT_IDX from a fixed seed), so
    # a resumed sweep writes the same JSON as an uninterrupted one. The checkpoint is dropped once the JSON is written.
    part = out / (out_name + ".partial")
    ck_id = f"{sha}:{os.path.getsize(a.ckpt)}:{int(os.path.getmtime(a.ckpt))}"
    done = {}
    if part.exists():
        try:
            P = json.loads(part.read_text())
            if P.get("ckpt_id") == ck_id and P.get("n") == n_pat and P.get("draws") == a.draws:
                done = {(r["family"], r["protocol"], r["frac"]): r for r in P["rows"]}
                base_boot = {tuple(k.split("|")): np.asarray(v) for k, v in P["base_boot"].items()}
                print(f"resuming from {part}: {len(done)} cells already scored", flush=True)
            else:
                print(f"ignoring {part}: written for a different checkpoint, subset or draw count", flush=True)
        except Exception as e:
            print(f"ignoring unreadable {part}: {e}", flush=True)
    def save_partial():
        part.write_text(json.dumps({"ckpt_id": ck_id, "n": n_pat, "draws": a.draws, "rows": rows,
                                    "base_boot": {"|".join(k): v.tolist() for k, v in base_boot.items()}}))
    for fam, (regime, _, _, beta) in fams.items():
        for frac in FRACS:
            if (fam, PROTOCOL, frac) in done:   # scored before a restart
                r = dict(done[(fam, PROTOCOL, frac)], scored_before_restart=True); rows.append(r)
                if frac == 0: base[(fam, PROTOCOL)] = r["auroc"]
                continue
            aucs, aps, reals, ps = [], [], [], []
            for d in range(1 if frac == 0 else a.draws):
                auc, ap_, rz, y, p = score_cell(m, ds, dev, fam, frac, beta, d, lookups,
                                                max_len=a.max_len, indices=indices)
                aucs.append(auc); aps.append(ap_); reals.append(rz); ps.append(p)
            # The reported point estimate is the MEAN OF PER-DRAW AUROCs, so the
            # bootstrap must resample the same statistic: within each patient
            # replicate, average the per-draw AUROCs. Bootstrapping the mean of
            # PREDICTIONS instead scores an ensemble, which is a different (and
            # higher) quantity -- it put the point estimate outside its own CI.
            bv = np.mean([boot_vals(y, pd, BOOT_IDX) for pd in ps], axis=0)
            ci_lo, ci_hi = (float(np.nanpercentile(bv, 2.5)),
                            float(np.nanpercentile(bv, 97.5)))
            if frac == 0:
                base_boot[(fam, PROTOCOL)] = bv
            dv = base_boot[(fam, PROTOCOL)] - bv          # paired, same replicates
            dci_lo, dci_hi = (float(np.nanpercentile(dv, 2.5)),
                              float(np.nanpercentile(dv, 97.5)))
            r = {"family": fam, "regime": regime, "protocol": PROTOCOL, "frac": frac,
                 "beta": beta, "auroc": float(np.mean(aucs)), "auroc_sd": float(np.std(aucs)),
                 "auprc": float(np.mean(aps)), "realized": float(np.mean(reals)),
                 "ci_lo": ci_lo, "ci_hi": ci_hi, "n_boot": NBOOT,
                 "delta_ci_lo": dci_lo, "delta_ci_hi": dci_hi,
                 "n": int(n_pat), "ckpt_sha": sha, "task": a.task,
                 "model": a.model, **extra}
            if frac == 0: base[(fam, PROTOCOL)] = r["auroc"]
            r["delta_auroc"] = base[(fam, PROTOCOL)] - r["auroc"]
            rows.append(r)
            print(f"  {fam} {PROTOCOL:6s} frac={frac:.2f} auroc={r['auroc']:.4f} "
                  f"[{ci_lo:.4f},{ci_hi:.4f}] realized={r['realized']:.3f} "
                  f"d={r['delta_auroc']:+.4f} [{dci_lo:+.4f},{dci_hi:+.4f}]", flush=True)
            save_partial()
    # null-arm check: frac=0 must agree across families
    z = {(r["family"], r["protocol"]): r["auroc"] for r in rows if r["frac"] == 0}
    fresh = sorted(set(round(v, 6) for k, v in z.items() if (k[0], k[1], 0.0) not in done))   # scored in this process: exact
    assert len(fresh) <= 1, f"NULL-ARM MISMATCH across cells scored in this run: {fresh}"
    vals = sorted(set(round(v, 6) for v in z.values()))   # across a restart GPU kernels are not bit-reproducible: tolerant
    assert vals[-1] - vals[0] <= 2e-3, f"NULL-ARM MISMATCH across the restart: {vals}"
    print(f"null arm OK: every family reproduces baseline {vals[0]:.4f}"
          + (f" (spread {vals[-1] - vals[0]:.1e} across the restart)" if len(vals) > 1 else ""), flush=True)
    (out / out_name).write_text(json.dumps(rows, indent=2))
    part.unlink(missing_ok=True)
    print(f"wrote {out}/{out_name}  ({len(rows)} cells, model={a.model}, tf32={a.tf32})", flush=True)
    # MLflow: one run per (model, task); each family's curve is a metric
    # indexed by realized dose in percent, so the sweep is browsable without the JSON
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    mlflow.set_experiment("ICLR-missingness")
    with mlflow.start_run(run_name=f"{a.model}/{a.task}/{a.suffix}"):
        mlflow.set_tags({"arch": a.model, "task": a.task, "stage": "missingness"})
        mlflow.log_params({"model": a.model, "task": a.task, "draws": a.draws, "n_boot": NBOOT,
                           "n_patients": int(n_pat), "families": ",".join(fams), "ckpt_sha": sha,
                           "tf32": int(a.tf32)})
        mlflow.log_metric("baseline_auroc", vals[0])
        for r in rows:
            step = int(round(r["realized"] * 100))
            mlflow.log_metrics({f"{r['family']}_{r['protocol']}_auroc": r["auroc"],
                                f"{r['family']}_{r['protocol']}_delta": r["delta_auroc"]}, step=step)
        mlflow.log_artifact(str(out / out_name))

if __name__ == "__main__": main()
