#!/bin/bash
# usage: run_experiments.sh <model> [gpu]
# Expects data/models/<model>_pretrain/best.pt to exist.
set -uo pipefail
MODEL=${1:?usage: run_experiments.sh <model> [gpu]}
export CUDA_VISIBLE_DEVICES=${2:-1}
PY=${PY:-python}
I=${ICLR_ROOT:-$(cd "$(dirname "$0")" && pwd)}
cd $I/code/src || exit 1
CK=$I/data/models/${MODEL}_pretrain/best.pt
[ -f "$CK" ] || { echo "FATAL: no pretrain checkpoint at $CK"; exit 1; }

for T in mortality los_gt_7d readmit_30d; do
  echo "--- finetune $MODEL/$T $(date) ---"
  $PY -u finetune.py --model $MODEL --task $T --epochs 8 --patience 2 ${FT_ARGS:-} || echo "FT $MODEL/$T FAILED"
done
for T in mortality los_gt_7d readmit_30d; do
  FT=$I/data/models/${MODEL}_ft/$T/best.pt
  [ -f "$FT" ] || { echo "skip $T"; continue; }
  echo "--- missingness $MODEL/$T ---"
  $PY -u run_missingness.py --model $MODEL --task $T --ckpt $FT --draws 3 \
    || echo "MISS $MODEL/$T FAILED"
done
$PY -u figures.py all
echo "=== DONE $MODEL $(date) ==="
