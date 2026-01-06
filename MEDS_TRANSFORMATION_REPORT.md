# MEDS Transformation Status Report

**Generated:** 2026-01-05 23:09

## Executive Summary

The MEDS transformation pipeline **failed** during the `shard_events` stage due to an **Out of Memory (OOM)** error while processing the large `labevents.csv.gz` file (1.9GB compressed, ~125M rows).

---

## Pipeline Stage Status

| Stage | Status | Notes |
|-------|--------|-------|
| MIMIC-IV Download | ✅ Complete | 7GB downloaded from PhysioNet |
| Pre-MEDS Conversion | ✅ Complete | All files processed/symlinked |
| shard_events | ❌ Failed | OOM on labevents.csv.gz |
| split_and_shard_subjects | ⏸️ Pending | - |
| convert_to_sharded_events | ⏸️ Pending | - |
| merge_to_MEDS_cohort | ⏸️ Pending | - |
| extract_code_metadata | ⏸️ Pending | - |
| finalize_MEDS_metadata | ⏸️ Pending | - |
| finalize_MEDS_data | ⏸️ Pending | - |
| Tokenization | ⏸️ Pending | - |

---

## Data Status

### Raw MIMIC-IV Data (`data/mimic-iv-raw/`)
**Status:** ✅ Complete

| Directory/File | Size | Status |
|----------------|------|--------|
| `hosp/` | ~4.5GB | ✅ Present |
| `icu/` | ~2.5GB | ✅ Present |
| Metadata CSVs | ~600KB | ✅ Present |

**Largest files (potential memory issues):**
| File | Compressed Size | Estimated Rows |
|------|-----------------|----------------|
| `labevents.csv.gz` | 1.9GB | ~125M rows |
| `emar.csv.gz` | 485MB | ~27M rows |
| `poe.csv.gz` | 476MB | Unknown |
| `emar_detail.csv.gz` | 450MB | Unknown |

### Pre-MEDS Data (`data/mimic-iv-pre-meds/`)
**Status:** ✅ Complete

Generated parquet files:
- `hosp/patients.parquet` (1.1MB)
- `hosp/diagnoses_icd.parquet` (17.9MB)
- `hosp/drgcodes.parquet` (6.8MB)
- `hosp/d_icd_diagnoses.parquet` (1.5MB)
- `hosp/d_icd_procedures.parquet` (1.2MB)

Other files symlinked to raw data.

### MEDS Output (`data/mimic-iv-meds/`)
**Status:** ⚠️ Partial

Successfully sharded files:
| File | Rows | Status |
|------|------|--------|
| `transfers` | 1,890,972 | ✅ Complete |
| `emar` | 26,850,359 | ✅ Complete |
| `procedures_icd` | 669,186 | ✅ Complete |
| `drgcodes` | 604,377 | ✅ Complete |
| `labevents` | ~125,000,000 | ❌ Failed (OOM) |

---

## Error Details

### Error Type
**OOM Kill** - Process terminated by kernel (exit code 137)

### Location
Stage: `shard_events`
File: `/teamspace/studios/this_studio/EHR_FM/data/mimic-iv-raw/hosp/labevents.csv.gz`

### Error Message
```
[2026-01-05 23:09:47,866][MEDS_transforms.runner][INFO] - Command error:
Killed

ValueError: Stage shard_events failed via MEDS_extract-shard_events ... with return code 137.
```

### Root Cause
- `labevents.csv.gz` is 1.9GB compressed (~20+GB uncompressed)
- Contains ~125 million rows
- System has 15GB RAM
- Reading entire file into memory exceeded available RAM

---

## System Resources

| Resource | Value |
|----------|-------|
| Total RAM | 15GB |
| Available RAM | ~14GB |
| Swap | 9GB (1.4GB used) |
| N_WORKERS | 4 |

---

## Files Skipped by Event Config

The following files are intentionally skipped (not in event conversion config):
- `chartevents.csv.gz` (ICU chart events)
- `inputevents.csv.gz` (ICU inputs)
- `outputevents.csv.gz` (ICU outputs)
- `procedureevents.csv.gz` (ICU procedures)
- `prescriptions.csv.gz`
- `pharmacy.csv.gz`
- `microbiologyevents.csv.gz`
- `poe.csv.gz`, `poe_detail.csv.gz`
- Various metadata/dictionary files

---

## Recommended Solutions

### Option 1: Reduce Workers (Quick Fix)
Reduce parallel workers to free memory:
```bash
export N_WORKERS=1
./run_pipeline.sh --skip-training
```

### Option 2: Use SLURM Cluster
If available, use SLURM for distributed processing:
```bash
pip install "MEDS_transforms[slurm_parallelism]"
# Configure slurm_runner.yaml with your cluster
```

### Option 3: Pre-decompress Large Files
Decompress the largest files before processing:
```bash
cd data/mimic-iv-raw/hosp
gunzip labevents.csv.gz  # Creates ~20GB CSV
```
Then set `DO_UNZIP=false` (already unzipped).

### Option 4: Use Machine with More RAM
Recommend 32GB+ RAM for full MIMIC-IV processing.

### Option 5: Process in Chunks (Code Modification)
Modify the MEDS_transforms shard_events to process in smaller chunks.

---

## Disk Usage Summary

| Directory | Size |
|-----------|------|
| `mimic-iv-raw/` | ~7GB |
| `mimic-iv-pre-meds/` | ~30MB |
| `mimic-iv-meds/` | ~1.5GB (partial) |
| **Total** | ~8.5GB |

---

## Next Steps

1. **Choose a solution** from recommendations above
2. **Clean partial MEDS output** (optional):
   ```bash
   rm -rf data/mimic-iv-meds/shard_events/hosp/labevents/
   ```
3. **Retry the pipeline**:
   ```bash
   ./run_pipeline.sh --skip-training
   ```
4. **Monitor memory** during processing:
   ```bash
   watch -n 1 free -h
   ```

---

## Files Created by This Run

```
EHR_FM/
├── data/
│   ├── mimic-iv-raw/           # ✅ Complete (7GB)
│   │   ├── hosp/
│   │   └── icu/
│   ├── mimic-iv-pre-meds/      # ✅ Complete
│   │   ├── hosp/
│   │   └── icu/
│   └── mimic-iv-meds/          # ⚠️ Partial
│       └── shard_events/
│           └── hosp/           # 4/11 files complete
└── pipeline_run.log            # Full log
```

