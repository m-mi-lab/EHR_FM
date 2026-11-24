#!/bin/bash
# Run the downstream evaluation (the true-removal missingness sweep) for one architecture on a
# CHPC PE GPU, using finetuned checkpoints copied from the lab box, and ship the results back.
#     nohup bash chpc/sweep.sh <arch> <gpu> [task ...] > logs/sweep_<arch>.log 2>&1 &
# Needs data/models/<arch>_ft/<task>/best.pt for each task (scp from the lab box first):
#     scp -r $LAB:$LABROOT/data/models/<arch>_ft ~/ICLR/data/models/
# If the finetuned checkpoint is missing but data/models/<arch>_pretrain/best.pt exists (the arch was
# pretrained on CHPC), the task is finetuned here first with the lab box's settings (run_experiments.sh),
# and the pretrain + finetune checkpoints are copied to the lab box so it holds every model.
# Matmuls run in TF32 (--tf32 1): fp32 on an H200 is no faster than a 4090; TF32 is ~8x.
# Results land in the lab box's data/results/missingness/<arch>/<task>.event.json after each task.
set -uo pipefail
ARCH=${1:?usage: chpc/sweep.sh <arch> <gpu> [task ...]}; GPU=${2:?usage: chpc/sweep.sh <arch> <gpu> [task ...]}; shift 2
TASKS=${*:-mortality los_gt_7d readmit_30d}
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT" || exit 1
LAB=${LAB:?set LAB=user@lab-host}; LABROOT=${LABROOT:?set LABROOT=<repo root on the lab box>}
export ICLR_ROOT="$ROOT" MIMIC_HOSP="$ROOT/raw/hosp" CUDA_VISIBLE_DEVICES="$GPU"
export MLFLOW_TRACKING_URI="sqlite:///$ROOT/mlflow_chpc${MLFLOW_SUFFIX:-}.db"
PY="$ROOT/.venv/bin/python"
case "$ARCH" in   # large models: the recipe's effective batch of 64 stays, but the micro-batch is the pretrain's so one H200 holds the activations (finetune.py divides the loss by --accum and steps every --accum batches)
  mambal|gpt2l) FT_ARGS=${FT_ARGS:-"--bs 16 --accum 4"};;
  bigbird)      FT_ARGS=${FT_ARGS:-"--bs 8 --accum 8"};;
esac
TL=$("$PY" -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')")
export LD_LIBRARY_PATH="$TL${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p logs
PRE="$ROOT/data/models/${ARCH}_pretrain/best.pt"
ssh -o BatchMode=yes -o ConnectTimeout=15 "$LAB" "mkdir -p $LABROOT/data/results/missingness/$ARCH $LABROOT/data/models/${ARCH}_pretrain" \
  || echo "warning: could not create result dirs on the lab box (results stay local until re-synced)"
ship() {   # ship <local file> <lab-relative dir>: copy one file to the lab box (dir created on the fly)
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$LAB" "mkdir -p $LABROOT/$2" 2>/dev/null
  scp -q -o BatchMode=yes "$1" "$LAB:$LABROOT/$2/" && echo "synced $2/$(basename "$1") to the lab box"; }
cd "$ROOT/code/src" || exit 1
echo "=== SWEEP $ARCH on GPU $GPU tasks: $TASKS  $(date) ==="
if [ -f "$PRE" ] && ! ssh -o BatchMode=yes -o ConnectTimeout=15 "$LAB" "test -f $LABROOT/data/models/${ARCH}_pretrain/best.pt" 2>/dev/null; then
  for F in best.pt history.json; do [ -f "$(dirname "$PRE")/$F" ] && ship "$(dirname "$PRE")/$F" "data/models/${ARCH}_pretrain"; done
fi
LOG="$ROOT/logs/sweep${FAMILIES:+x}_$ARCH.log"   # FAMILIES=<list> (extra families, own log and JSON) never shares a log with the regular sweep
for T in $TASKS; do
  CK="$ROOT/data/models/${ARCH}_ft/$T/best.pt"
  if [ -n "${FAMILIES:-}" ]; then   # extra families (A2,B2,B2o ...) into <task>.<suffix>.json; no finetune here
    SUF=${SUFFIX:-visits_event}; OUTJ="$ROOT/data/results/missingness/$ARCH/$T.$SUF.json"
    [ -f "$CK" ] && [ -f "$(dirname "$CK")/result.json" ] || { echo "skip $T (no finished finetune at $CK)"; continue; }   # best.pt without result.json = killed mid-run
    if [ -f "$OUTJ" ]; then echo "skip missingness-$SUF $T (result exists)"; else
      echo "--- missingness-$SUF $ARCH/$T families=$FAMILIES $(date) ---"
      "$PY" -u run_missingness.py --model "$ARCH" --task "$T" --ckpt "$CK" --draws 3 --tf32 1 \
        --families "$FAMILIES" --suffix "$SUF" --min_visits "${MIN_VISITS:-1}" --out "$ROOT/data/results/missingness" || echo "MISS-$SUF $ARCH/$T FAILED"
    fi
    [ -f "$OUTJ" ] && scp -q -o BatchMode=yes "$OUTJ" "$LAB:$LABROOT/data/results/missingness/$ARCH/" && echo "synced missingness/$ARCH/$T.$SUF.json to the lab box"
    scp -q -o BatchMode=yes "$LOG" "$LAB:$LABROOT/logs/chpc/" 2>/dev/null
    continue
  fi
  if { [ ! -f "$CK" ] || [ ! -f "$(dirname "$CK")/result.json" ]; } && [ -f "$PRE" ]; then   # no FINISHED finetune (best.pt without result.json = killed mid-run): finetune here, same recipe as run_experiments.sh
    echo "--- finetune $ARCH/$T $(date) ---"
    "$PY" -u finetune.py --model "$ARCH" --task "$T" --epochs 8 --patience 2 --ckpt "$PRE" \
      --out "$ROOT/data/models/${ARCH}_ft" ${FT_ARGS:-} || echo "FT $ARCH/$T FAILED"
    for F in best.pt result.json; do [ -f "$(dirname "$CK")/$F" ] && ship "$(dirname "$CK")/$F" "data/models/${ARCH}_ft/$T"; done
  fi
  [ -f "$CK" ] || { echo "skip $T (no $CK)"; continue; }
  OUTJ="$ROOT/data/results/missingness/$ARCH/$T.event.json"
  if [ -f "$OUTJ" ]; then echo "skip missingness $T (result exists)"; else
    echo "--- missingness $ARCH/$T $(date) ---"
    "$PY" -u run_missingness.py --model "$ARCH" --task "$T" --ckpt "$CK" --draws 3 --tf32 1 \
      --out "$ROOT/data/results/missingness" || echo "MISS $ARCH/$T FAILED"
  fi
  [ -f "$OUTJ" ] && scp -q -o BatchMode=yes "$OUTJ" "$LAB:$LABROOT/data/results/missingness/$ARCH/" && echo "synced missingness/$ARCH/$T.event.json to the lab box"
  scp -q -o BatchMode=yes "$ROOT/logs/sweep_$ARCH.log" "$LAB:$LABROOT/logs/chpc/" 2>/dev/null
done
echo "=== SWEEP DONE $ARCH ${FAMILIES:+(families $FAMILIES) }$(date) ==="
scp -q -o BatchMode=yes "$LOG" "$LAB:$LABROOT/logs/chpc/" 2>/dev/null
