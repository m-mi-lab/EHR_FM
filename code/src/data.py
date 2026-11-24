"""EIC datasets, ETHOS-style.

Every sample is  [ DEMOGRAPHIC PREFIX (5) ] + [ timeline window ]:
    GENDER//x  RACE//x  MARITAL//x  AGE_T1//d  AGE_T2//d
where age is at the FIRST token of the window, encoded as two digit tokens
(tens, units), exactly as ETHOS's `_age_to_tokens`. Prefix positions are never
masked, never predicted, and never withheld by the missingness families.

Pretraining windows are drawn ETHOS-style: a start position is sampled
uniformly over ALL tokens in the split (so long records contribute
proportionally), the window is clipped to that patient, and one epoch is one
pass over the corpus in expectation. Draws are a pure function of
(seed, epoch, index) so every architecture sees the same window starts.
Evaluation (EICTask) takes the last tokens before the prediction time, which is
ETHOS's inference dataset.
"""
from __future__ import annotations
import glob, hashlib, numpy as np, polars as pl, torch
from pathlib import Path
from torch.utils.data import Dataset

import os
import time
import json
# Project root and raw-MIMIC location come from the environment so the same
# code runs on the lab box (defaults) and on CHPC (ICLR_ROOT=..., MIMIC_HOSP=...).
ROOT = Path(os.environ.get("ICLR_ROOT", Path(__file__).resolve().parents[2]))
EIC = ROOT / "data" / "eic"
LAB = ROOT / "data" / "labels"
RAW = Path(os.environ.get("MIMIC_HOSP", ROOT / "raw" / "hosp"))
PAD = 0
MASK_TOKEN = "[MASK]"
PREFIX_LEN = 5

GENDERS  = ("F", "M", "UNKNOWN")
RACES    = ("WHITE", "BLACK", "HISPANIC", "ASIAN", "OTHER", "UNKNOWN")
MARITALS = ("MARRIED", "SINGLE", "WIDOWED", "DIVORCED", "UNKNOWN")


def collapse_race(s):
    """33 MIMIC-IV strings -> 6, the same collapse ETHOS applies
    (race_priority_mapping): keep the major group, everything else OTHER."""
    if s is None: return "UNKNOWN"
    s = s.upper()
    if s.startswith(("UNKNOWN", "UNABLE", "PATIENT DECLINED")): return "UNKNOWN"
    if s.startswith("HISPANIC"): return "HISPANIC"
    head = s.split(" - ")[0].split("/")[0].strip()
    return head if head in ("WHITE", "BLACK", "ASIAN") else "OTHER"


def load_vocab():
    """itos[i] is the code for id i. MEDS-Torch ids are 1-based with 0 = PAD;
    [MASK] and the demographic/age tokens are appended after the MEDS codes."""
    v = pl.read_parquet(EIC / "metadata/codes.parquet")["code"].to_list()
    extra = ([MASK_TOKEN]
             + [f"GENDER//{g}" for g in GENDERS]
             + [f"RACE//{r}" for r in RACES]
             + [f"MARITAL//{m}" for m in MARITALS]
             + [f"AGE_T1//{i}" for i in range(10)]
             + [f"AGE_T2//{i}" for i in range(10)])
    itos = ["[PAD]"] + v + extra
    return itos, {c: i for i, c in enumerate(itos)}


_DEMO = None
def _demographics(stoi):
    """subject_id -> (gender_id, race_id, marital_id, birth_year). Gender from
    patients; race/marital = mode over that subject's admissions; birth year =
    anchor_year - anchor_age, the same definition MEDS_BIRTH uses."""
    global _DEMO
    if _DEMO is not None: return _DEMO
    pat = pl.read_csv(RAW / "patients.csv", columns=["subject_id", "gender", "anchor_age", "anchor_year"],
                      infer_schema_length=100000)
    adm = pl.read_csv(RAW / "admissions.csv", columns=["subject_id", "race", "marital_status"],
                      infer_schema_length=100000)
    adm = adm.with_columns(pl.col("race").map_elements(collapse_race, return_dtype=pl.Utf8).alias("race"),
                           pl.col("marital_status").fill_null("UNKNOWN"))
    mode = (adm.group_by("subject_id")
               .agg(pl.col("race").mode().first().alias("race"),
                    pl.col("marital_status").mode().first().alias("marital")))
    df = pat.join(mode, on="subject_id", how="left").with_columns(
        pl.col("race").fill_null("UNKNOWN"), pl.col("marital").fill_null("UNKNOWN"),
        (pl.col("anchor_year") - pl.col("anchor_age")).alias("birth_year"))
    out = {}
    for sid, g, r, m, by in zip(df["subject_id"], df["gender"], df["race"], df["marital"], df["birth_year"]):
        g = g if g in GENDERS else "UNKNOWN"
        m = m if m in MARITALS else "UNKNOWN"
        out[int(sid)] = (stoi[f"GENDER//{g}"], stoi[f"RACE//{r}"], stoi[f"MARITAL//{m}"], int(by))
    _DEMO = out
    return out


def age_tokens(age_years, stoi):
    """ETHOS _age_to_tokens: age clipped to [0, 99], split into tens and units."""
    a = int(min(max(age_years, 0.0), 99.0))
    return stoi[f"AGE_T1//{a // 10}"], stoi[f"AGE_T2//{a % 10}"]


class _Corpus:
    """One split, held ONCE as a flat int32 array with per-subject and per-event
    offsets plus event times. ~3 GB for train. Python-list storage was ~30 GB.

    The full-split arrays are cached under data/eic/cache/<split>/ and loaded
    memory-mapped, so N training processes on one host share ONE copy through the
    page cache instead of holding N private ~10 GB copies (that is what a 64 GB
    CHPC session cannot afford with six runs). The first process builds the
    cache; concurrent starters wait for it."""
    def __init__(self, split, limit=None, subjects=None):
        files = sorted(glob.glob(str(EIC / f"tokenization/event_seqs/{split}/*.parquet")))
        cache = EIC / "cache" / split
        meta = json.dumps([[os.path.basename(f), os.path.getsize(f)] for f in files])
        if limit is None and subjects is None:
            self._wait_or_build(cache, meta, files, split)
            self._load_cache(cache)
        else:
            self._build(files, split, limit, subjects)
        self.n_tokens = int(self.sub_off[-1])

    _NAMES = ("flat", "sub_off", "subject_id", "evoff_flat", "evoff_idx", "times_flat")

    def _wait_or_build(self, cache, meta, files, split):
        done, building = cache / "meta.json", cache / ".building"
        if done.exists() and done.read_text() == meta: return
        cache.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(building, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)   # atomic: one builder
        except FileExistsError:
            print(f"corpus cache for {split!r} is being built by another process; waiting", flush=True)
            while not (done.exists() and done.read_text() == meta):
                time.sleep(10)
                if not building.exists() and not done.exists():         # builder died: take over
                    return self._wait_or_build(cache, meta, files, split)
            return
        try:
            print(f"building corpus cache for {split!r} at {cache}", flush=True)
            self._build(files, split, None, None)
            evoff_idx = np.zeros(len(self.ev_off) + 1, dtype=np.int64)
            np.cumsum([len(o) for o in self.ev_off], out=evoff_idx[1:])
            arrays = {"flat": self.flat, "sub_off": self.sub_off, "subject_id": self.subject_id,
                      "evoff_flat": np.concatenate(self.ev_off), "evoff_idx": evoff_idx,
                      "times_flat": np.concatenate(self.times)}
            for k, v in arrays.items(): np.save(cache / f"{k}.npy", v)
            done.write_text(meta)
        finally:
            building.unlink(missing_ok=True)

    @staticmethod
    def _preread(path, chunk=1 << 24):
        """Sequentially read a file once so its pages are in the page cache before it is memory-mapped: on a network
        filesystem one large streaming read is fast, thousands of random 4 KB page-fault reads are not."""
        with open(path, "rb") as fh:
            while fh.read(chunk): pass

    def _load_cache(self, cache):
        if not os.environ.get("ICLR_NO_PREREAD"):
            for k in self._NAMES: self._preread(cache / f"{k}.npy")
        a = {k: np.load(cache / f"{k}.npy", mmap_mode="r") for k in self._NAMES}
        self.flat, self.sub_off, self.subject_id = a["flat"], a["sub_off"], a["subject_id"]
        idx, eo, tf = a["evoff_idx"], a["evoff_flat"], a["times_flat"]
        n = len(self.sub_off) - 1
        # per-subject views into the shared maps (same interface as the in-memory build)
        self.ev_off = [eo[idx[i]:idx[i + 1]] for i in range(n)]
        self.times = [tf[idx[i] - i:idx[i + 1] - (i + 1)] for i in range(n)]   # ev_off has len(events)+1 per subject

    def _build(self, files, split, limit, subjects):
        sc_files = sorted(glob.glob(str(EIC / f"tokenization/schemas/{split}/*.parquet")))
        if subjects is not None:
            # filter while scanning (subject subsets; EICTask now indexes the cached split instead)
            keep = pl.Series("subject_id", list(subjects))
            ev = pl.concat([pl.read_parquet(f).filter(pl.col("subject_id").is_in(keep)) for f in files])
            sc = pl.concat([pl.read_parquet(f, columns=["subject_id", "time"]).filter(pl.col("subject_id").is_in(keep))
                            for f in sc_files])
        else:
            ev = pl.read_parquet(files)
            sc = pl.read_parquet(sc_files, columns=["subject_id", "time"])
        df = ev.join(sc, on="subject_id", how="inner")
        if limit: df = df.head(limit)
        self.subject_id = df["subject_id"].to_numpy()
        n = df.height
        ev_lens = df.select(pl.col("code").list.eval(pl.element().list.len()).alias("l"))["l"]
        self.ev_off = [None] * n; self.times = [None] * n
        sub_tok = np.zeros(n, dtype=np.int64)
        for i in range(n):
            l = np.asarray(ev_lens[i].to_list(), dtype=np.int64)
            o = np.zeros(len(l) + 1, dtype=np.int64); np.cumsum(l, out=o[1:])
            self.ev_off[i] = o; sub_tok[i] = o[-1]
        self.sub_off = np.zeros(n + 1, dtype=np.int64); np.cumsum(sub_tok, out=self.sub_off[1:])
        self.flat = np.zeros(int(self.sub_off[-1]), dtype=np.int32)
        col, tcol = df["code"], df["time"]
        for i in range(n):
            if sub_tok[i]:
                self.flat[self.sub_off[i]:self.sub_off[i + 1]] = np.fromiter(
                    (c for e in col[i].to_list() for c in e), dtype=np.int32, count=int(sub_tok[i]))
            self.times[i] = np.array(tcol[i].to_list(), dtype="datetime64[us]")

    def __len__(self): return len(self.sub_off) - 1

    def time_at(self, j, local_tok, after=None):
        """Timestamp of the event containing the j-th subject's local_tok-th
        token -- or, if `after` is given, of the first event at or beyond it
        whose time exceeds `after`. Used for age-at-window-start: MEDS_BIRTH is a
        token IN our timeline (in ETHOS it lives in the static context), so a
        window that begins at the record's start begins at birth and would
        report age 0 for every record that fits in the window (37.6% of them).
        Skipping to the first post-birth event restores ETHOS's semantics: age at
        the first clinical event the model reads."""
        e = int(np.searchsorted(self.ev_off[j], local_tok, side="right")) - 1
        e = max(e, 0); t = self.times[j]
        if after is not None:
            k = int(np.searchsorted(t[e:], after, side="right"))
            if e + k < len(t): return t[e + k]
        return t[e]


def _birth(demo, sid):
    by = demo.get(int(sid), (0, 0, 0, None))[3]
    return None if by is None else np.datetime64(f"{by}-01-01", "us") + np.timedelta64(180, "D")


def _prefix(demo, stoi, sid, t_start):
    g, r, m, by = demo.get(int(sid), (stoi["GENDER//UNKNOWN"], stoi["RACE//UNKNOWN"], stoi["MARITAL//UNKNOWN"], None))
    if by is None or t_start is None:
        a1, a2 = stoi["AGE_T1//0"], stoi["AGE_T2//0"]
    else:
        yrs = (t_start - np.datetime64(f"{by}-01-01", "us")) / np.timedelta64(1, "D") / 365.25
        a1, a2 = age_tokens(float(yrs), stoi)
    return np.array([g, r, m, a1, a2], dtype=np.int64)


def _pack(prefix, x, max_len):
    out = np.zeros(max_len, dtype=np.int64)
    out[:PREFIX_LEN] = prefix
    n = min(len(x), max_len - PREFIX_LEN)
    out[PREFIX_LEN:PREFIX_LEN + n] = x[-n:] if n else x[:0]
    return out


class EICSequences(Dataset):
    """Pretraining. windows="random" (train, ETHOS) or "to_end" (validation)."""
    def __init__(self, split="train", max_len=512, limit=None, windows="random", seed=1234):
        _, stoi = load_vocab()
        self.stoi, self.demo = stoi, _demographics(stoi)
        self.c = _Corpus(split, limit)
        self.max_len, self.L = max_len, max_len - PREFIX_LEN
        assert windows in ("random", "to_end"); self.windows = windows
        self.seed, self.epoch = seed, 0
        # one epoch ~ one pass over the corpus: as many windows as fit
        self.n = max(len(self.c), self.c.n_tokens // self.L) if windows == "random" else len(self.c)

    def set_epoch(self, e): self.epoch = int(e)
    def __len__(self): return self.n

    def __getitem__(self, i):
        c = self.c
        if self.windows == "random":
            h = hashlib.sha256(f"{self.seed}|{self.epoch}|{i}".encode()).digest()
            p = int.from_bytes(h[:8], "little") % c.n_tokens         # token-proportional start
            j = int(np.searchsorted(c.sub_off, p, side="right")) - 1
            lo, hi = int(c.sub_off[j]), int(c.sub_off[j + 1])
            start = min(p, max(lo, hi - self.L))                    # clip to the patient
        else:
            j = i; lo, hi = int(c.sub_off[j]), int(c.sub_off[j + 1]); start = max(lo, hi - self.L)
        x = c.flat[start:min(start + self.L, hi)]
        sid = c.subject_id[j]
        t0 = c.time_at(j, start - lo, after=_birth(self.demo, sid)) if hi > lo else None
        out = _pack(_prefix(self.demo, self.stoi, sid, t0), x, self.max_len)
        return {"code": torch.from_numpy(out), "mask": torch.from_numpy((out != PAD).astype(np.int64))}


class EICTask(Dataset):
    """Labelled admissions, truncated STRICTLY before prediction_time, with the
    demographic prefix; age is at the window's first token (ETHOS inference)."""
    def __init__(self, task: str, split="train", max_len=512, limit=None, with_events=False):
        self.with_events = with_events   # also return per-event boundaries and timestamps (variable length: no DataLoader collation)
        _, stoi = load_vocab()
        self.stoi, self.demo = stoi, _demographics(stoi)
        lab = pl.read_parquet(LAB / f"{task}.parquet").filter(pl.col("split") == split)
        if limit: lab = lab.head(limit)
        self.c = _Corpus(split)   # the memory-mapped split cache; task rows are picked by index below (no private copy)
        have = set(self.c.subject_id.tolist())
        lab = lab.filter(pl.col("subject_id").is_in(list(have)))
        row_of = {int(s): i for i, s in enumerate(self.c.subject_id)}
        self.idx = np.array([row_of[int(s)] for s in lab["subject_id"].to_list()], dtype=np.int64)
        self.pt = np.array(lab["prediction_time"].to_list(), dtype="datetime64[us]")
        self.y = lab["label"].to_list()
        self.max_len, self.L = max_len, max_len - PREFIX_LEN

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        c = self.c; j = int(self.idx[i])
        n_ev = int(np.searchsorted(c.times[j], self.pt[i], side="left"))
        end = int(c.sub_off[j] + c.ev_off[j][n_ev]); lo = int(c.sub_off[j])
        start = max(lo, end - self.L)
        x = c.flat[start:end]
        sid = c.subject_id[j]
        t0 = c.time_at(j, start - lo, after=_birth(self.demo, sid)) if end > lo else None
        out = _pack(_prefix(self.demo, self.stoi, sid, t0), x, self.max_len)
        item = {"code": torch.from_numpy(out),
                "mask": torch.from_numpy((out != PAD).astype(np.int64)),
                "label": torch.tensor(float(self.y[i])),
                "n_tokens": torch.tensor(len(x))}
        if self.with_events:
            # events overlapping [start, end): local token offsets of each event's first token within x (the first
            # event may be cut by the window start), its timestamp, and the timestamp of the event before the window
            eo = c.ev_off[j]; t = c.times[j]
            e0 = int(np.searchsorted(eo, start - lo, side="right")) - 1
            starts = np.maximum(eo[e0:n_ev] - (start - lo), 0).astype(np.int64)
            item["ev_start"] = starts
            item["ev_time_us"] = (t[e0:n_ev].astype("datetime64[us]").astype(np.int64))
            item["t_prev_us"] = int(t[e0 - 1].astype("datetime64[us]").astype(np.int64)) if e0 > 0 else -1
        return item
