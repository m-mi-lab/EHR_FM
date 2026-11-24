#!/bin/bash
# Launch (or resume) one pretrain on a CHPC PE GPU session and keep the lab box
# informed. Run from the unpacked ICLR root, in the Jupyter terminal:
#     bash chpc/launch.sh <arch> [SEED=<n>]        e.g.  bash chpc/launch.sh mamba
# Several archs can share the H200 (~25-30 GB each at one 32,768-token micro-batch).
# Idempotent: every new 72 h session runs the same command; --resume 1 picks up
# data/models/<arch>_pretrain/last.pt (full optimizer/scheduler/RNG state).
set -uo pipefail
ARCH=${1:?usage: chpc/launch.sh <arch> [gpu]}; GPU=${2:-0}
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT" || exit 1
LAB=${LAB:?set LAB=user@lab-host}; LABROOT=${LABROOT:?set LABROOT=<repo root on the lab box>}   # lab box; key auth set up once
export ICLR_ROOT="$ROOT" MIMIC_HOSP="$ROOT/raw/hosp" PY="$ROOT/.venv/bin/python"
export MLFLOW_TRACKING_URI="sqlite:///$ROOT/mlflow_chpc${MLFLOW_SUFFIX:-}.db"      # local store (synced below)
# The Apptainer image carries its own torch libs on LD_LIBRARY_PATH; the mamba/causal-conv1d
# extensions must resolve libc10_cuda.so from the venv torch they were built against.
TL=$("$PY" -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')")
export LD_LIBRARY_PATH="$TL${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# one 32,768-token micro-batch per step: bs = 32768 / context
# (the big ones keep some accumulation so several runs fit on one H200 at once)
# Six co-tenants: keep micro-batches modest (activations are the variable part) and stop
# each caching allocator from hoarding: expandable segments release memory between processes.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 64 GB job cgroup: one loader worker per run (the mmap corpus makes window slicing cheap)
export ICLR_WORKERS=${ICLR_WORKERS:-1}
case "$ARCH" in bilstm|bert|bilstms|bilstml|berts|bertl) BS=64; ACC=1;; gpt2|gpt2s) BS=32; ACC=1;; gpt2l) BS=16; ACC=2;;
  mistral|mistrals|mamba|mambas) BS=16; ACC=1;; mistrall) BS=4; ACC=4;; mambal) BS=16; ACC=1;; bigbird) BS=8; ACC=1;; *) BS=16; ACC=1;; esac
export EXTRA_ARGS=${EXTRA_ARGS:-"--bs $BS --accum $ACC --resume 1"}
mkdir -p logs
exec 9>/tmp/iclr_launch2${MLFLOW_SUFFIX:-}.lock; flock 9   # node-local lock: two launchers cannot both pass the check below
if pgrep -f "[p]retrain.py --model $ARCH " >/dev/null || pgrep -f "[r]un_pretrain.sh [0-9]+ $ARCH$" >/dev/null; then echo "$ARCH pretrain already running"; else
  if command -v tmux >/dev/null 2>&1; then          # attach with: tmux attach -t pre_$ARCH   (Ctrl-b d detaches)
    tmux new-session -d -s "pre_$ARCH" "bash run_pretrain.sh $GPU $ARCH >> logs/pre_$ARCH.log 2>&1" 8>&- 9>&-
    echo "started $ARCH pretrain on GPU $GPU in tmux session pre_$ARCH -> logs/pre_$ARCH.log   [$EXTRA_ARGS]"
  else
    nohup bash run_pretrain.sh $GPU "$ARCH" >> "logs/pre_$ARCH.log" 2>&1 8>&- 9>&- &
    echo "started $ARCH pretrain on GPU $GPU (pid $!, nohup; no tmux in this container) -> logs/pre_$ARCH.log   [$EXTRA_ARGS]"
  fi
fi
# every 10 min: all logs, histories and the MLflow store go to the lab box
if ! pgrep -f "[c]hpc_sync_loop${MLFLOW_SUFFIX:-} " >/dev/null; then
  nohup bash -c 'exec -a "chpc_sync_loop${MLFLOW_SUFFIX:-} " bash -c "while true; do scp -q -o BatchMode=yes -o ConnectTimeout=15 '"$ROOT"'/logs/pre_*.log '"$ROOT"'/mlflow_chpc*.db '"$LAB"':'"$LABROOT"'/logs/chpc/ 2>/dev/null; for d in '"$ROOT"'/data/models/*_pretrain; do scp -q -o BatchMode=yes -o ConnectTimeout=15 \$d/history.json '"$LAB"':'"$LABROOT"'/logs/chpc/\$(basename \$d)_history.json 2>/dev/null; done; sleep 600; done"' >/dev/null 2>&1 8>&- 9>&- &
  echo "sync loop started (every 10 min -> $LAB:$LABROOT/logs/chpc/)"
fi
echo "follow with:  tail -f logs/pre_$ARCH.log"
