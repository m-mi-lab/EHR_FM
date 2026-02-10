## Hardware Specifications (User's System)

**Machine**: `pettinelab-lambda-vector-0`
- **GPUs**: 2x NVIDIA GeForce RTX 4090 (24GB VRAM each, 49GB total)
  - Driver: 550.127.05
  - Compute Capability: 8.9
- **CPU**: AMD Ryzen Threadripper 7960X (24 cores, 48 threads, up to 5.36 GHz)
- **RAM**: 251 GB (212 GB available)
- **Storage**: 1.8TB NVMe (1.5TB used, 245GB free)
- **OS**: Ubuntu 22.04.5 LTS (Jammy Jellyfish)
- **Python**: 3.13.9 (Miniconda)
- **PyTorch**: 2.5.1

**Training Context**:
- Default training runs on **2 GPUs** (can use `world_size=2` for DDP)
- Total VRAM: 49GB (comfortable for large batches)
- Multi-GPU training commands should use `torchrun --nproc_per_node=2`

---

## Quick orientation

This repo implements EHR_FM — an EHR foundation model built on a GPT2-like backbone with optional Mixture-of-Experts (MoE) layers.
Key folders/files to read first:
- `src/train.py` — training entrypoint (Hydra-managed, supports single-GPU and DDP).
- `src/model.py` — GPT2LMNoBiasModel which composes the base GPT shell and optional MoE blocks.
- `src/manager.py` — `MANAGER` / `MOEManager` used to accumulate auxiliary MoE losses (must be reset between forwards).
- `src/tokenizer/` — tokenization pipeline, `run_tokenization.py`, `vocabulary.py`, and `constants.py` (special tokens like `SpecialToken.DEATH`).
- `src/conf/` — Hydra configs (default `config.yaml`, model and train presets live under `src/conf/model` and `src/conf/train`).
- `scripts/` — evaluation and inference scripts (`infer.py`, `infer_trajectory.py`, `run_infer.sh`).

## Big-picture architecture (short)
- Data -> `src/tokenizer` produces tokenized timelines and a `Vocabulary` saved in `data/tokenized_datasets`.
- Training (`python -m src.train`) reads Hydra config (`src/conf/config.yaml`) and builds:
  - A base GPT2 transformer (via `transformers.GPT2Config`) and
  - Optional MoE layers driven by the Hydra `cfg.model` (n_experts, top_k, aux_loss, router_z_loss).
- MoE auxiliary losses are collected by `MANAGER` and added to the main loss in training; watch `src/manager.py` semantics when changing routing or loss handling.
- Checkpoints saved by training include `vocab_stoi`, `model_configs`, and `model` state dict — inference scripts expect this shape.

## How developers run things (concrete commands & gotchas)
- Local dev / quick run (Hydra overrides):
  - Install and run in a dev env: `pip install -e .[jupyter]` (see `pyproject.toml`).
  - Train (2 GPUs default on this machine): `python -m src.train` or override: `python -m src.train model=gpt2_small_monolith train=monolith_million world_size=2`
  - Single GPU: `python -m src.train world_size=1`
  - Tokenize data: `python3 -m src.tokenizer.run_tokenization --config-name=tokenization_train` (see README).
- Inference (deterministic direct-prob):
  - `python scripts/infer.py --test hosp_mortality --model /path/to/best_model.pt --data-dir /path/to/tokenized --output infer_results`
  - Trajectory-based inference (sampling): `python scripts/infer_trajectory.py --test hosp_mortality --model /path/to/best_model.pt --output infer_results`
- Docker compose: `docker-compose up -d --build` then `docker exec -it ehr_fm_training bash` and run `python -m src.train` (see `README.md` for MLflow/TensorBoard ports).

## Hydra conventions used here
- Hydra config path: `config_path="conf"` in `src/train.py` — configs live at `src/conf/` (defaults: `train: 4_exp`, `model: gpt2_small_4exp`).
- Override examples: `python -m src.train model=gpt2_small_4exp train=4_exp world_size=1 seed=123`
- The training code logs the full resolved Hydra config to checkpoints as `hydra_config_full` — use that to reproduce runs.

## Project-specific patterns to follow (do not break)
- Vocabulary padding: model code expects padded vocab sizes (multiple of 64). When creating/loading Vocabulary, code pads to 64-aligned size before building GPT2Config.
- Checkpoint format: inference scripts look for `vocab_stoi` and `model_configs` keys in checkpoint dict — preserve these when changing saving code.
- MoE loss handling: `MOEManager` stores non-detached auxiliary losses; code relies on explicit resets per forward pass (see `MANAGER.reset_all()`). Avoid changing that lifecycle without verifying memory behavior.
- DDP setup: `src/train.py` sets `MASTER_PORT=23556` by default. For multi-node or cluster runs, ensure MPI/SLURM settings and `MASTER_ADDR/MASTER_PORT` are consistent.

## Integration points & external dependencies
- MLflow: training will call `mlflow.set_tracking_uri()` if `MLFLOW_TRACKING_URI` is set; Docker compose runs an MLflow server (see `docker-compose.yml`).
- TensorBoard logs: training writes to `outputs/.../tensorboard_logs` and `tensorboard` service reads `outputs/`.
- MEDS_transforms: data transforms are wired via the `MEDS_transforms` dependency and `scripts/MEDS_transforms/` helper code.
- Check `pyproject.toml` for required packages (torch, transformers, hydra-core, mlflow, pytorch-lightning, polars, etc.).

## Quick debugging tips for common failures
- If vocabulary/token mismatch on load -> check checkpoint `vocab_stoi` and `Vocabulary` ordering; ensure padding to multiples of 64.
- If DDP hangs or fails to init -> try `world_size=1` or set `MASTER_ADDR`/`MASTER_PORT` and verify GPUs visible via `nvidia-smi` (should show 2x RTX 4090).
- If MoE training consumes memory unexpectedly -> ensure `MANAGER.reset_all()` is called between forwards and that aux losses are not stored indefinitely.
- **CRITICAL BUG FIXED (2026-02-06)**: batch_size_per_gpu calculation on line 195 of `src/train.py` was dividing by both world_size AND gradient_accumulation_steps (wrong!). Now fixed to use cfg.train.batch_size directly.
- If training loss doesn't decrease -> check learning rate (should be 6e-4, NOT 1e-2) and batch size calculation.

## CRITICAL: Do NOT create markdown files
- User EXPLICITLY does not want unnecessary markdown summary/documentation files created
- Do NOT create files like CONFIG_SUMMARY.md, DEBUGGING_REPORT.md, SUMMARY.md, etc. unless explicitly requested
- Keep documentation in existing files only (README.md, this file, inline comments)

## Where to look for examples
- Training run example and outputs: `outputs/YYYY-MM-DD/...` (see README for layout).
- Example inference scripts: `scripts/infer.py` (direct probability) and `scripts/infer_trajectory.py` (sampling trajectories).
- Tokenizer and datasets: `src/tokenizer/` and `src/tokenizer/datasets/*` (dataset formats used throughout training and inference).

If anything above is unclear or you'd like me to expand an area (e.g., add a checklist for reproducing a training run, or capture specific Hydra option names), tell me which part and I'll iterate.
