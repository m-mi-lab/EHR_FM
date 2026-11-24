#!/bin/bash
# Pretrain one or more architectures to convergence.
#   usage: ./run_pretrain.sh <gpu> <arch> [arch ...]
#
# Inputs are IDENTICAL for every architecture: same corpus, same splits, same
# ETHOS-style window draws (fixed seed, re-drawn per epoch), same demographic
# prefix, no augmentation. What differs per model is its published context
# length (model.CONTEXT), its objective (MLM for the bidirectional bilstm/bert/
# bigbird, next-token for the causal gpt2/mistral/mamba) and its AMP dtype
# (bf16 for mamba, fp16 otherwise).
#
# Regularisation is dropout + weight decay only. The causal models need more
# of it: next-token prediction has a fixed target every epoch, whereas MLM's
# random mask is its own regulariser.
set -uo pipefail
GPU=${1:?usage: run_pretrain.sh <gpu> <arch>...}; shift
export CUDA_VISIBLE_DEVICES=$GPU
PY=${PY:-python}                          # interpreter (CHPC: the project .venv)
I=${ICLR_ROOT:-$(cd "$(dirname "$0")" && pwd)}                 # project root (CHPC: wherever the bundle was unpacked)
export ICLR_ROOT="$I"
EXTRA_ARGS=${EXTRA_ARGS:-}                              # e.g. "--bs 16 --accum 1 --resume 1" on a 140 GB GPU
cd "$I/code/src" || exit 1

# Context and batch are MODEL-SPECIFIC (model.CONTEXT / model.BATCH, bs*L = 32768):
#   bilstm 512  bert 512  gpt2/gpt2s/gpt2l 1024  mistral/mistrals/mistrall 2048  bigbird 4096  mamba/mambas/mambal 2048
# Objective is model-specific: bidirectional -> MLM, causal -> next-token.
# Measured peak memory (GB, AMP, model-specific context) + margin. Two queues
# share a GPU; each model waits until that much is free before starting, so
# queues self-arrange without OOMing each other.
# The guard sees FREE memory at start, but a co-tenant that is still loading
# has not peaked yet -- mamba (8.9 GB at bs 16) passed the check next to a
# loading mistral and both then OOMed. Mamba (24 blocks) runs at bs 4x4 so the
# pair fits regardless of who starts first.
need_gb() { case "$1" in bilstm|bilstms) echo 5;; bilstml|berts) echo 8;; bertl) echo 22;; bert|bigbird) echo 18;; gpt2) echo 20;; gpt2s) echo 7;; gpt2l) echo 22;; mistrals) echo 6;; mistrall) echo 20;; mambas) echo 5;; mambal) echo 20;; mistral) echo 15;; mamba) echo 8;; esac; }
wait_mem() {
  local need=$(( $1 * 1024 )) free
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | head -1)
    [ -n "$free" ] && [ "$free" -ge "$need" ] && return 0
    echo "  waiting for $1 GB free on GPU $GPU (have $((free/1024)) GB) $(date +%H:%M)"; sleep 60
  done
}
for M in "$@"; do
  case "$M" in
    bilstm)                 D=768;  L=4;  H=8;  DROP=0.1; WD=0.01 ;;   # deep LSTMs do not train; 4 is standard
    bert|bigbird)           D=768;  L=12; H=12; DROP=0.1; WD=0.01 ;;   # MLM: the random mask regularises
    bilstms)                D=512;  L=4;  H=8;  DROP=0.1; WD=0.01 ;;   # BiLSTM ladder, bottom: d=512, 4 layers
    bilstml)                D=1024; L=4;  H=8;  DROP=0.1; WD=0.01 ;;   # BiLSTM ladder, top: d=1024, 4 layers (deeper LSTMs do not train)
    berts)                  D=512;  L=8;  H=8;  DROP=0.1; WD=0.01 ;;   # BERT-medium geometry (L=8, H=512): bottom of the BERT ladder
    bertl)                  D=1024; L=24; H=16; DROP=0.1; WD=0.01 ;;   # BERT-large geometry: top of the BERT ladder
    gpt2)                   D=1024; L=24; H=16; DROP=0.3; WD=0.10 ;;   # GPT-2 MEDIUM geometry (310M here); large OOMs a 4090 at bs 4
    gpt2s)                  D=768;  L=12; H=12; DROP=0.3; WD=0.10 ;;   # GPT-2 SMALL geometry (91M here): the size ablation of gpt2
    gpt2l)                  D=1280; L=36; H=20; DROP=0.3; WD=0.10 ;;   # GPT-2 LARGE geometry (~700M here); needs a big GPU
    mistrals)               D=512;  L=8;  H=8;  DROP=0.3; WD=0.10 ;;   # Mistral arch, ~34M: bottom of the Mistral ladder
    mistrall)               D=1024; L=24; H=16; DROP=0.3; WD=0.10 ;;   # Mistral arch, ~370M: top of the Mistral ladder
    mistral)                D=768;  L=12; H=12; DROP=0.3; WD=0.10 ;;   # causal: fixed target, needs more
    mamba)                  D=768;  L=24; H=12; DROP=0.3; WD=0.10 ;;   # mamba-130m geometry (96.6M here)
    mambas)                 D=512;  L=16; H=8;  DROP=0.3; WD=0.10 ;;   # Mamba, 31M: bottom of the Mamba ladder
    mambal)                 D=1024; L=48; H=16; DROP=0.3; WD=0.10 ;;   # mamba-370m geometry (328M here)
    *) echo "unknown arch $M"; continue ;;
  esac
  LR=3e-4; WARM=500   # shared schedule: 500 warm-up steps, then plateau halving
  OPT=""   # optimiser extras (default AdamW betas 0.9/0.999, eps 1e-8 for everyone else)
  case "$M" in bigbird) LR=1e-4 ;; bertl) LR=5e-5; WARM=2000; OPT="--adam_eps 1e-6" ;; esac   # bigbird never learned at 3e-4; bertl diverged at 3e-4 (fp16), 1e-4 (bf16) and 5e-5 (bf16, epoch 6) -> AdamW eps 1e-6 (BERT's), resumed from epoch 4 (2026-10-03)
  wait_mem "$(need_gb "$M")"
  echo "=== pretrain $M  d=$D L=$L H=$H  dropout=$DROP wd=$WD lr=$LR warmup=$WARM${OPT:+ $OPT}  cap=500 patience=5  $(date) ==="
  $PY -u pretrain.py --model "$M" --d $D --layers $L --heads $H \
      --epochs 500 --patience 5 --lr_patience 3 --lr $LR --warmup $WARM $OPT --amp 1 \
      --dropout $DROP --wd $WD --seed 1234 $EXTRA_ARGS \
      || echo "PRETRAIN $M FAILED"
  echo "=== $M pretrain finished $(date) ==="
done
echo "=== GPU$GPU PRETRAIN QUEUE DONE $(date) ==="
