"""Task labels with BINDING prediction-time cutoffs.

The previous pipeline set cutoffs to the full sequence length, so inputs
contained events after the outcome was determined -- token count alone reached
0.830 AUROC on LOS and readmission scored 0.954 with all clinical content
deleted. Here the cutoff is a CLOCK TIMESTAMP, the approach CLMBR and EHRSHOT
use, and every label is defined strictly after it.

  mortality    in-hospital death        cutoff = admit + 24h
  los_gt_7d    LOS > 7 days             cutoff = admit + 24h
  readmit_30d  re-admission <= 30 days  cutoff = discharge

A stay is kept only if the cutoff falls strictly inside the stay, so the
outcome is always in the future at prediction time.
"""
import polars as pl, glob, json, os
from pathlib import Path

ROOT = Path(os.environ.get("ICLR_ROOT", Path(__file__).resolve().parents[1]))
RAW = str(Path(os.environ.get("MIMIC_HOSP", ROOT / "raw" / "hosp")) / "admissions.csv")
EIC = ROOT / "data" / "eic"
OUT = ROOT / "data" / "labels"; OUT.mkdir(parents=True, exist_ok=True)
GAP_H = 24

adm = (pl.read_csv(RAW, try_parse_dates=True)
       .select("subject_id","hadm_id","admittime","dischtime","deathtime","hospital_expire_flag")
       .drop_nulls(["admittime","dischtime"])
       .sort(["subject_id","admittime"]))
print("admissions:", format(adm.height, ","))

adm = adm.with_columns([
    (pl.col("dischtime") - pl.col("admittime")).dt.total_hours().alias("los_h"),
    (pl.col("admittime") + pl.duration(hours=GAP_H)).alias("cut_early"),
    pl.col("admittime").shift(-1).over("subject_id").alias("next_admit"),
])
adm = adm.with_columns([
    ((pl.col("next_admit") - pl.col("dischtime")).dt.total_days() <= 30).fill_null(False).cast(pl.Int8).alias("readmit_30d"),
    (pl.col("los_h") > 7*24).cast(pl.Int8).alias("los_gt_7d"),
    pl.col("hospital_expire_flag").cast(pl.Int8).alias("mortality"),
])

# Splits are re-derived from the CURRENT tokenization every time. They are NOT
# stable across extractions: re-running MEDS re-partitions subjects, so a stored
# split column from an earlier run would put training patients into held_out.
# This file must be regenerated whenever the corpus is rebuilt.
splits = {}
for sp in ("train","tuning","held_out"):
    fs = sorted(glob.glob(str(EIC/f"tokenization/event_seqs/{sp}/*.parquet")))
    ids = pl.scan_parquet(fs).select("subject_id").collect()["subject_id"].to_list()
    for i in ids: splits[i] = sp
print("tokenized subjects:", format(len(splits), ","))
sp_df = pl.DataFrame({"subject_id": list(splits.keys()), "split": list(splits.values())})

adm = adm.join(sp_df, on="subject_id", how="inner")
print("admissions for tokenized subjects:", format(adm.height, ","))

tasks = {}
# early-cutoff tasks: the cutoff must land strictly before discharge
early = adm.filter(pl.col("cut_early") < pl.col("dischtime"))
print(f"stays surviving the {GAP_H}h cutoff (cutoff < discharge):", format(early.height, ","))
for t in ("mortality","los_gt_7d"):
    tasks[t] = early.select(
        "subject_id","hadm_id","split",
        pl.col("cut_early").alias("prediction_time"),
        pl.col(t).alias("label"))
# readmission is predicted AT discharge
tasks["readmit_30d"] = adm.select(
    "subject_id","hadm_id","split",
    pl.col("dischtime").alias("prediction_time"),
    pl.col("readmit_30d").alias("label"))

summary = {}
for t, df in tasks.items():
    df.write_parquet(OUT/f"{t}.parquet")
    per = {sp: int(df.filter(pl.col("split")==sp).height) for sp in ("train","tuning","held_out")}
    prev = float(df["label"].mean())
    summary[t] = {"n": int(df.height), "prevalence": round(prev,4), "per_split": per}
    print(f"  {t:12s} n={df.height:>8,}  prevalence={prev:.4f}  {per}")
(OUT/"summary.json").write_text(json.dumps(summary, indent=2))
print("\nwrote", OUT)
