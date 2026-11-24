#!/bin/bash
# One-time environment build. Run INSIDE the CHPC PE Jupyter terminal (the
# Apptainer container on the GPU node), from the unpacked ICLR root:
#     bash chpc/setup_chpc.sh
# Reproduces the lab-box stack exactly (the one every other model was trained
# with): Python 3.12, torch 2.6.0+cu124, transformers 4.39.3, mamba_ssm 2.2.4,
# causal_conv1d 1.5.0.post8 -- the kernel wheels are the ones built against
# torch 2.6's pip build (pre-C++11 ABI). Newer mamba_ssm wheels are labelled for
# torch 2.6 but are not (they need the new ABI and triton>=3.5), so versions are
# pinned and the wheels are installed with --no-deps.
# Everything lives on scratch (home quota is tight); ./.venv is a symlink to it.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT" || exit 1
SCR=${SCR:-/scratch/general/pevast/$USER}
VENV=${VENV:-$SCR/iclr_venv}
# --- Python 3.12 (conda env on scratch, from the user's miniconda) -------------
PY312=${PY312:-$SCR/py312/bin/python3.12}
if [ ! -x "$PY312" ]; then
  CONDA=$(command -v conda || ls -d $HOME/miniconda3/bin/conda 2>/dev/null | head -1)
  [ -x "$CONDA" ] || { echo "no conda found; set PY312=<path to a python3.12>"; exit 1; }
  echo "creating Python 3.12 env at $SCR/py312 (conda, ~1 min)"
  "$CONDA" create -y -q -p "$SCR/py312" python=3.12 >/dev/null || { echo "conda create FAILED"; exit 1; }
fi
echo "python: $("$PY312" -c 'import sys; print(sys.executable, sys.version.split()[0])')   venv: $VENV"
# --- venv --------------------------------------------------------------------
rm -rf "$VENV"; "$PY312" -m venv "$VENV" || exit 1
rm -f .venv; ln -sfn "$VENV" .venv
P="$VENV/bin/pip"
"$P" install -q --upgrade pip
"$P" install -q "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124 || { echo "torch install FAILED"; exit 1; }
"$P" install -q "transformers==4.39.3" "mlflow>=3" polars pyarrow scikit-learn "numpy<2" pandas nvidia-ml-py psutil packaging ninja einops || { echo "deps FAILED"; exit 1; }
"$P" install -q --no-deps \
  "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8%2Bcu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl" \
  "https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4%2Bcu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl" \
  || { echo "kernel wheels FAILED"; exit 1; }
# --- verify (venv torch libs first: the container's own libs sit on LD_LIBRARY_PATH) ---
TL=$("$VENV/bin/python" -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')")
LD_LIBRARY_PATH="$TL${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$VENV/bin/python" - <<'PY'
import torch, transformers, mlflow
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
      "bf16", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else "-")
print("transformers", transformers.__version__, "mlflow", mlflow.__version__)
import mamba_ssm, causal_conv1d; print("kernels ok: mamba_ssm", mamba_ssm.__version__, "causal_conv1d", causal_conv1d.__version__)
PY
echo "setup done"
