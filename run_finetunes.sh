#!/bin/bash
# usage: run_finetunes.sh <model> [gpu]   -- the finetune stage only (the sweeps run on CHPC via a `sweep` job)
set -uo pipefail
MODEL=${1:?usage: run_finetunes.sh <model> [gpu]}
export CUDA_VISIBLE_DEVICES=${2:-0}
PY=${PY:-python}
I=${ICLR_ROOT:-$(cd "$(dirname "$0")" && pwd)}
cd $I/code/src || exit 1
CK=$I/data/models/${MODEL}_pretrain/best.pt
[ -f "$CK" ] || { echo "FATAL: no pretrain checkpoint at $CK"; exit 1; }
for T in mortality los_gt_7d readmit_30d; do
  echo "--- finetune $MODEL/$T $(date) ---"
  $PY -u finetune.py --model $MODEL --task $T --epochs 8 --patience 2 ${FT_ARGS:-} || echo "FT $MODEL/$T FAILED"
done
echo "=== FINETUNES DONE $MODEL $(date) ==="
