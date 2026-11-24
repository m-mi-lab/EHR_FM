"""Missingness on the tokenized EIC stream.

A keep-mask per draw says which tokens are withheld. apply_event_remove then deletes them as
events ("removal before tokenisation"): an event left with no content vanishes with its
time-gap token, and the time-gap tokens of the surviving events are re-derived from the timestamps.

Amputation follows mice::ampute: logit P(withhold) = alpha + beta*s(Z), with
alpha solved by bisection so the realized marginal rate matches the target.
beta = 0 is MCAR, nested -- so mechanism is a contrast in ONE parameter at
fixed dose.
"""
from __future__ import annotations
import re, numpy as np

PAD = 0
# Encounter scaffolding: never withheld by the content families. HOSPITAL_ADMISSION / HOSPITAL_DISCHARGE
# and SERVICE only exist since hosp/admissions and hosp/services were mapped
# (2026-09-08); before that the corpus had no explicit admission event at all and
# encounter structure was carried entirely by TRANSFER_TO and ICU.
STRUCTURAL = ("MEDS_DEATH", "MEDS_BIRTH", "TIME", "TRANSFER_TO", "ICU",
              "HOSPITAL_ADMISSION", "HOSPITAL_DISCHARGE", "SERVICE",
              # the ETHOS-style demographic prefix and the MLM token: never withheld
              "GENDER", "RACE", "MARITAL", "AGE_T1", "AGE_T2", "[MASK]")
_DEC = re.compile(r"//_Q_(\d+)$")


def build_lookups(itos):
    """decile_of[id] -> int|-1 ; is_struct[id] -> bool ; stem_of[id] -> str"""
    n = len(itos)
    dec = np.full(n, -1, dtype=np.int16)
    struct = np.zeros(n, dtype=bool)
    stems = []
    for i, c in enumerate(itos):
        m = _DEC.search(c)
        if m: dec[i] = int(m.group(1))
        struct[i] = c.startswith(STRUCTURAL) or c == "[PAD]"
        stems.append(_DEC.sub("", c))
    return dec, struct, np.array(stems, dtype=object)


def solve_alpha(scores, frac, beta, lo=-8.0, hi=8.0, tol=1e-4):
    """Bisect alpha so mean(expit(alpha + beta*s)) == frac. Same bracket as ampute."""
    if frac <= 0: return None
    s = np.asarray(scores, dtype=np.float64)
    f = lambda a: float(np.mean(1.0 / (1.0 + np.exp(-(a + beta * s)))) - frac)
    if f(lo) > 0 or f(hi) < 0:      # target unreachable within bracket
        return lo if f(lo) > 0 else hi
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) < 0: lo = mid
        else: hi = mid
        if hi - lo < tol: break
    return (lo + hi) / 2


def keep_mask(codes, valid, *, frac, beta=0.0, score=None, eligible=None, rng,
              exact_dose=True):
    """Boolean keep-mask over positions. `eligible`/`score` are per-position.

    exact_dose=True withholds EXACTLY round(frac * n_eligible) tokens per patient,
    chosen by weighted sampling without replacement with weights proportional to
    the amputation probability. This keeps the mechanism (beta shapes which
    tokens go) while removing dose as a channel: a per-patient Bernoulli draw has
    sd = sqrt(p(1-p)/n), which on a 50-token record is ~0.07 -- large enough that
    an MNAR arm and its MCAR twin would differ in HOW MUCH was withheld as well
    as HOW, and the mechanism effect would not be identified.
    """
    keep = valid.copy()
    if frac <= 0: return keep
    elig = valid if eligible is None else (valid & eligible)
    idx = np.flatnonzero(elig)
    if idx.size == 0: return keep
    s = np.zeros(idx.size) if score is None else np.asarray(score)[idx].astype(float)
    a = solve_alpha(s, frac, beta)
    p = 1.0 / (1.0 + np.exp(-(a + beta * s)))
    if not exact_dose:
        keep[idx] = rng.random(idx.size) >= p
        return keep
    k = int(round(frac * idx.size))
    if k <= 0: return keep
    k = min(k, idx.size)
    # Efraimidis-Spirakis weighted sampling without replacement
    w = np.clip(p, 1e-12, None)
    key = rng.random(idx.size) ** (1.0 / w)
    drop = idx[np.argsort(-key)[:k]]
    keep[drop] = False
    return keep


# ---- event removal: "removal before tokenisation" ------------------------------------------------------------
# The time-gap token at the head of every event is a pure function of the gap to the previous event and the
# tokenizer's bin edges (quantile_binning custom_quantiles for TIME//DELTA//TOKEN: years; bin = 1 + #edges < gap).
# Re-derived from timestamps this rule reproduces 100 % of the stored tokens (189,677 held-out events, 2026-10-28).
TIME_EDGES_YEARS = (1.90258e-06, 9.51293e-06, 1.902587e-05, 5.707762e-05, 0.00011415525, 0.00034246575, 0.0006849315,
                    0.00136986301, 0.00273972602, 0.00547945205, 0.0109589041, 0.01917808219, 0.03835616438,
                    0.08219178082, 0.16438356164, 0.32876712328, 1, 2, 5, 10, 20, 40)
SEC_PER_YEAR = 31556926.08          # meds_torch normalize_time_unit("years")
_EDGES = np.array(TIME_EDGES_YEARS)


def time_bin(gap_us):
    """1-based TIME//DELTA bin of a gap given in microseconds (the pipeline's own rule)."""
    return 1 + int((_EDGES < gap_us / 1e6 / SEC_PER_YEAR).sum())


def time_ids(stoi):
    """bin -> token id for TIME//DELTA//TOKEN//_Q_<bin>; set of all such ids."""
    ids = {k: stoi[f"TIME//DELTA//TOKEN//_Q_{k}"] for k in range(1, 40) if f"TIME//DELTA//TOKEN//_Q_{k}" in stoi}
    return ids, set(ids.values())


def apply_event_remove(codes, keep, L, ev_start, ev_time_us, t_prev_us, tid_of_bin, tid_set, prefix_len):
    """Delete the withheld tokens as events: an event that loses all its non-time tokens disappears with its
    time-gap token, and every surviving event's time-gap token is recomputed from the surviving timeline, so the
    output is what the tokenizer would have produced had those events never been in the record. Returns
    (codes, mask, n_events_dropped)."""
    n_tok = int((codes != PAD).sum())
    body = codes[prefix_len:n_tok]; kb = keep[prefix_len:n_tok]
    n_ev = len(ev_start); bounds = list(ev_start) + [len(body)]
    out = list(codes[:prefix_len]); prev_t = t_prev_us; dropped = 0
    for e in range(n_ev):
        a, b = int(bounds[e]), int(bounds[e + 1])
        if b <= a: continue
        toks, kp = body[a:b], kb[a:b]
        has_head = int(toks[0]) in tid_set
        content = toks[1:] if has_head else toks
        ck = kp[1:] if has_head else kp
        if len(content) and not ck.any():          # every non-time token of this event withheld: the event vanishes
            dropped += 1; continue
        if has_head and prev_t >= 0:
            out.append(tid_of_bin[time_bin(int(ev_time_us[e]) - prev_t)])
        elif has_head:
            out.append(int(toks[0]))                # no event before the window: the gap token stays as stored
        out.extend(int(x) for x in content[ck])
        prev_t = int(ev_time_us[e])
    out = np.array(out[:L], dtype=np.int64); res = np.zeros(L, dtype=np.int64); res[:len(out)] = out
    return res, (res != PAD).astype(np.int64), dropped


def assert_event_cell(orig, valid, eligible, keep, new_codes, tid_set, prefix_len, frac, tol=0.02):
    """Checks for the event protocol: withheld content absent, kept content present in order, prefix intact,
    only time-gap tokens differ from the kept sequence; returns the realised fraction of the eligible pool."""
    kept = orig[keep & valid]; new = new_codes[new_codes != PAD]
    assert np.array_equal(new[:prefix_len], orig[:prefix_len]), "prefix changed"
    a = [int(x) for x in kept[prefix_len:] if int(x) not in tid_set]
    b = [int(x) for x in new[prefix_len:] if int(x) not in tid_set]
    assert a == b, "kept non-time tokens changed under event removal"
    n_el = int(eligible.sum()); n_drop = int((eligible & ~keep).sum())
    realized = n_drop / max(n_el, 1)
    assert abs(realized - frac) < max(tol, 2.0 / max(n_el, 1)), f"realized {realized:.3f} != requested {frac:.3f}"
    return realized


# ---- visit-level families (A2 / B2 / B2o): withhold whole PRIOR visits, scaffold included ----
# A prior visit is the run of tokens between two consecutive HOSPITAL_DISCHARGE markers, the second
# marker included (ED arrival, admission, stay, discharge; its TIME tokens too). The open visit after
# the last discharge holds the prediction time and is never withheld; the partial segment before the
# first discharge in the window (its start may be cut off by the window) is not eligible either.
# Records with no eligible prior visit are not scored by these families (the multi-visit subset).

def visit_spans(codes, valid, is_disch):
    """[(start, end)) of every complete prior visit inside the window, oldest first."""
    d = np.flatnonzero(valid & is_disch[codes])
    return [(int(d[j - 1]) + 1, int(d[j]) + 1) for j in range(1, len(d))]


def visit_keep_mask(valid, spans, *, frac, beta=0.0, order=None, rng):
    """Withhold k = round-half-up(frac * n_spans) whole spans. order=None: uniform (A2, MCAR);
    'recency': the most recent prior visit is withheld first (B2, MAR on the visit index, beta > 0);
    'oldest': the reverse (B2o). Same amputation form and weighted sampling as keep_mask."""
    keep = valid.copy(); n = len(spans)
    if frac <= 0 or n == 0: return keep, 0
    k = min(n, int(np.floor(frac * n + 0.5)))
    if k <= 0: return keep, 0
    if order is None or beta == 0:
        s = np.zeros(n)
    else:
        r = np.arange(n, dtype=float) / max(n - 1, 1)        # 0 = oldest ... 1 = most recent
        s = 2 * r - 1 if order == "recency" else 1 - 2 * r
    a = solve_alpha(s, frac, beta) if beta else 0.0
    p = 1.0 / (1.0 + np.exp(-(a + beta * s)))
    w = np.clip(p, 1e-12, None)
    key = rng.random(n) ** (1.0 / w)
    for j in np.argsort(-key)[:k]:
        st, en = spans[j]; keep[st:en] = False
    return keep, int(k)
