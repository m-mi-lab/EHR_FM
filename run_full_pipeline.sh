#!/bin/bash
# Full rebuild: MIMIC-IV -> MEDS (with real 80/10/10 splits) -> EIC tokenization.
# The stock MIMIC-IV_Example config pins split_fracs train:1, which produced an
# empty tuning/held_out. ICLR/configs/extract_MIMIC.yaml overrides that.
set -uo pipefail
# Required: MIMICIV_RAW_DIR (PhysioNet MIMIC-IV root), MIMIC_EXAMPLE_DIR (MEDS MIMIC-IV_Example checkout),
#           EIC_CONFIG (meds-torch eic_config.yaml). Activate the env with MEDS_transform-runner first.
export POLARS_SKIP_CPU_CHECK=1 HYDRA_FULL_ERROR=1 N_WORKERS=16
A=${ICLR_ROOT:-$(cd "$(dirname "$0")" && pwd)}
RAW=${MIMICIV_RAW_DIR:?set MIMICIV_RAW_DIR}
OUT=$A/data
rm -rf $OUT/pre_MEDS $OUT/MEDS $OUT/eic
mkdir -p $OUT/pre_MEDS $OUT/MEDS $OUT/eic

export MIMICIV_RAW_DIR=$RAW
export MIMICIV_PRE_MEDS_DIR=$OUT/pre_MEDS
export MIMICIV_MEDS_EXTRACT_DIR=$OUT/MEDS
export EVENT_CONVERSION_CONFIG_FP=$A/configs/event_configs_seq.yaml

echo "=== [1/2] MEDS extract (80/10/10) $(date) ==="
cd "${MIMIC_EXAMPLE_DIR:?set MIMIC_EXAMPLE_DIR}" || exit 1
python pre_MEDS_seq.py input_dir=$RAW cohort_dir=$OUT/pre_MEDS || exit 1
MEDS_transform-runner "pipeline_config_fp=$A/configs/extract_MIMIC.yaml" || exit 1

echo "=== [2/2] EIC tokenization $(date) ==="
export MEDS_DIR=$OUT/MEDS MODEL_DIR=$OUT/eic
MEDS_transform-runner "pipeline_config_fp=${EIC_CONFIG:?set EIC_CONFIG}" || exit 1
echo "=== DONE $(date) ==="
