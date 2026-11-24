#!/bin/bash
# Event-removal sweeps ("removal before tokenisation") on a lab-box card: the six token families under the
# event protocol into data/results/missingness/<model>/<task>.event.json. Skips tasks whose file exists.
#   usage: run_event_sweeps_lab.sh <model> <gpu> [task ...]
set -uo pipefail
MODEL=${1:?usage: run_event_sweeps_lab.sh <model> <gpu> [task ...]}
export CUDA_VISIBLE_DEVICES=${2:?usage: run_event_sweeps_lab.sh <model> <gpu> [task ...]}; shift 2
TASKS=${*:-mortality los_gt_7d readmit_30d}
PY=${PY:-python}
I=${ICLR_ROOT:-$(cd "$(dirname "$0")" && pwd)}
SRC=${SRC:-$I/code/src}   # code dir
cd $SRC || exit 1
for T in $TASKS; do
  FT=$I/data/models/${MODEL}_ft/$T/best.pt
  { [ -f "$FT" ] && [ -f "$I/data/models/${MODEL}_ft/$T/result.json" ]; } || { echo "skip $T (no finished finetune)"; continue; }
  if [ -f "$I/data/results/missingness/$MODEL/$T.event.json" ]; then echo "skip event $T (result exists)"; else
    echo "--- missingness-event $MODEL/$T $(date) ---"
    $PY -u run_missingness.py --model $MODEL --task $T --ckpt $FT --draws 3 \
      --families A1,A3,B1,C1,C2,C3 --suffix event || echo "MISS-event $MODEL/$T FAILED"; fi
done
echo "=== EVENT SWEEPS DONE $MODEL ($TASKS) $(date) ==="
