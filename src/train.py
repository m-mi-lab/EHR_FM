import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import os
import time
import math
import inspect
from pathlib import Path
from collections import namedtuple, defaultdict
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import GPT2Config
from loguru import logger
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter
from contextlib import nullcontext
import mlflow

from src.tokenizer.constants import SpecialToken as ST
from src.tokenizer.datasets import TimelineDataset
from src.tokenizer.vocabulary import Vocabulary 
from src.metrics import estimate_loss
from src.manager import MANAGER # MOEManager instance
from src.model import GPT2LMNoBiasModel, ModelOutput

# ModelOutput is already defined in model.py, no need to redefine here if imported
# ModelOutput = namedtuple("ModelOutput", ["loss", "logits", "aux_loss", "router_z_loss"])


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    os.environ['MASTER_PORT'] = '23556'
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
    logger.info("Cleaned up process group.")

def get_lr(it, cfg_training: DictConfig):
    if it < cfg_training.warmup_iters:
        return cfg_training.lr * it / cfg_training.warmup_iters
    if it > cfg_training.lr_decay_iters:
        return cfg_training.min_lr
    decay_ratio = (it - cfg_training.warmup_iters) / (cfg_training.lr_decay_iters - cfg_training.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg_training.min_lr + coeff * (cfg_training.lr - cfg_training.min_lr)

def configure_optimizers(model: torch.nn.Module, weight_decay, learning_rate, betas, device_type):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    # num_decay_params = sum(p.numel() for p in decay_params) # Logged later
    # num_nodecay_params = sum(p.numel() for p in nodecay_params) # Logged later
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    return optimizer

def make_infinite_loader(loader):
    while True:
        yield from iter(loader)

def _flatten_dict(d, parent_key='', sep='.'):
    """Flatten nested dictionary for MLflow parameter logging."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)):
            # Convert lists/tuples to strings for MLflow
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def train_worker(rank, world_size, cfg: DictConfig):
    logger.remove()
    log_format = (
        f"<green>{{time:YYYY-MM-DD HH:mm:ss}}</green> | "
        f"<level>{{level: <8}}</level> | "
        f"RANK={rank} | "
        f"<cyan>{{name}}</cyan>:<cyan>{{function}}</cyan>:<cyan>{{line}}</cyan> - <level>{{message}}</level>"
    )
    logger.add(lambda msg: print(msg, end=""), colorize=True, format=log_format, level="INFO")

    if rank == 0:
        log_file_path = Path(cfg.train.out_dir) / "train.log"
        logger.add(log_file_path, rotation="10 MB", level="DEBUG")
        logger.info(f"Logging to file: {log_file_path}")

    logger.info(f"Running DDP training worker on rank {rank}.")
    setup(rank, world_size)
    logger.info(f"Rank {rank}/{world_size} initialized process group.")

    writer = None
    mlflow_run = None
    if rank == 0:
        tb_log_dir = Path(cfg.train.out_dir) / "tensorboard_logs"
        writer = SummaryWriter(log_dir=str(tb_log_dir))
        logger.info(f"Tensorboard logs will be saved to: {tb_log_dir}")

        profiler_log_dir = Path(cfg.train.out_dir) / "profiler_traces"
        profiler_log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Profiler traces will be saved to: {profiler_log_dir}")

        # Initialize MLflow
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow_experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "EHR_FM")
        mlflow_artifact_uri = os.getenv("MLFLOW_ARTIFACT_URI", None)
        try:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(mlflow_experiment_name)
            # Start run with explicit artifact location if configured
            run_kwargs = {}
            if mlflow_artifact_uri:
                # Set the artifact location for this run
                # Note: This requires the experiment to be configured with the artifact location
                # or we pass it when creating the experiment
                experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
                if experiment is None:
                    mlflow.create_experiment(mlflow_experiment_name, artifact_location=mlflow_artifact_uri)
                    mlflow.set_experiment(mlflow_experiment_name)
            mlflow_run = mlflow.start_run()
            logger.info(f"MLflow tracking initialized: {mlflow_tracking_uri}, experiment: {mlflow_experiment_name}")
            if mlflow_artifact_uri:
                logger.info(f"MLflow artifact URI: {mlflow_artifact_uri}")
            
            # Log hyperparameters
            flat_config = OmegaConf.to_container(cfg, resolve=True)
            mlflow.log_params(_flatten_dict(flat_config))
            
            # Log tags
            mlflow.set_tag("model_type", "MoE" if (hasattr(cfg.model, 'n_experts') and cfg.model.n_experts > 1) else "Standard GPT")
            mlflow.set_tag("world_size", str(world_size))
            mlflow.set_tag("dtype", cfg.train.dtype)
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}. Continuing without MLflow logging.")
            mlflow_run = None

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    logger.info(f"Rank {rank} using device: {device}")

    try:
        vocab_dir = Path(cfg.data.tokenized_dir)
        vocab = Vocabulary.from_path(vocab_dir)
        vocab_size = (len(vocab) // 64 + 1) * 64 if len(vocab) % 64 != 0 else len(vocab)
        logger.info(f"Rank {rank} loaded vocabulary with size {len(vocab)} (padded: {vocab_size}).")
        tokens_of_interest = {stoken.value: vocab.encode(stoken.value) for stoken in [ST.DEATH, ST.ADMISSION, ST.DISCHARGE] if stoken.value in vocab.stoi}
        logger.info(f"Rank {rank} Tokens of Interest: {tokens_of_interest}")
    except Exception as e:
        logger.exception(f"Rank {rank} FAILED to load vocabulary from {cfg.data.tokenized_dir}: {e}")
        cleanup()
        return

    try:
        full_dataset = TimelineDataset(
            input_dir=cfg.data.tokenized_dir,
            n_positions=cfg.model.n_positions, # This should be base n_positions
            is_encoder_decoder=False,
        )
        logger.info(f"Rank {rank} loaded full dataset with {len(full_dataset)} samples.")

        val_split_count = int(cfg.data.val_split_fraction * len(full_dataset))
        train_split_count = len(full_dataset) - val_split_count
        generator = torch.Generator().manual_seed(cfg.seed) # Use main seed
        train_indices, val_indices = torch.utils.data.random_split(
            range(len(full_dataset)), [train_split_count, val_split_count], generator=generator
        )
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        logger.info(f"Rank {rank} - Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.seed)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=cfg.seed)

        # FIXED: batch_size in config is per-GPU batch size, not divided by world_size or gradient_accumulation_steps
        # gradient_accumulation_steps is a training loop concept, not for DataLoader
        batch_size_per_gpu = cfg.train.batch_size
        logger.info(f"Rank {rank} - Batch size per GPU: {batch_size_per_gpu}")
        logger.info(f"Rank {rank} - Effective batch per GPU: {batch_size_per_gpu * cfg.train.gradient_accumulation_steps}")
        logger.info(f"Rank {rank} - Total effective batch (all GPUs): {batch_size_per_gpu * cfg.train.gradient_accumulation_steps * world_size}")

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_per_gpu, sampler=train_sampler, num_workers=cfg.train.num_workers, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size_per_gpu, sampler=val_sampler, num_workers=cfg.train.num_workers, pin_memory=True)
        train_dataloader = make_infinite_loader(train_dataloader)
        val_dataloader = make_infinite_loader(val_dataloader)
        logger.info(f"Rank {rank} created DataLoaders.")
    except Exception as e:
        logger.exception(f"Rank {rank} FAILED during data loading: {e}")
        cleanup()
        return

    # Model setup
    # Base GPT2Config
    base_model_config = GPT2Config(
         vocab_size=vocab_size,
         n_positions=cfg.model.n_positions,
         n_embd=cfg.model.n_embd,
         n_layer=cfg.model.n_layer,
         n_head=cfg.model.n_head,
         activation_function=cfg.model.activation_function,
         resid_pdrop=cfg.model.resid_pdrop,
         embd_pdrop=cfg.model.embd_pdrop,
         attn_pdrop=cfg.model.attn_pdrop,
         bias=cfg.model.bias,
         # n_inner can be part of cfg.model and then used by MLP if needed
         n_inner=getattr(cfg.model, 'n_inner', None), # Pass n_inner if specified in moe_config
    )
    logger.info(f"Rank {rank} created base GPT2 model config.")

    # The moe_hydra_config (cfg.model) will be passed directly and contains MoE specific params
    # It should also contain any parameters that Block/MLP/Router might need if they differ from base_config
    # e.g., if n_embd for MoE layers were different, it should be in cfg.model.
    # For now, assume n_embd, n_head, bias, activation_function, resid_pdrop are consistent from base_model_config
    # and cfg.model mainly provides n_experts, top_k, aux_loss params etc.
    # The model.py's GPT2LMNoBiasModel will use base_model_config for transformer shell and moe_hydra_config for MoE parts.
    
    model = GPT2LMNoBiasModel(base_gpt_config=base_model_config, moe_hydra_config=cfg.model).to(device)
    logger.info(f"Rank {rank} initialized model.")

    is_moe_model = hasattr(cfg.model, 'n_experts') and cfg.model.n_experts > 1 # Check the input Hydra config
    if rank == 0 and is_moe_model:
        logger.info("-----------------------------------------------------------------------------------------------")
        logger.info(f"MoE Model Configured: Number of Experts = {cfg.model.n_experts}, Top_k = {cfg.model.top_k}")
        if writer:
            writer.add_text('Model/Type', "Mixture of Experts", 0)
            writer.add_text('Model/NumExperts', str(cfg.model.n_experts), 0)
            writer.add_text('Model/TopK', str(cfg.model.top_k), 0)
            writer.add_text('Model/AuxLossEnabled', f"{getattr(cfg.model, 'use_aux_loss', False)} (Weight: {getattr(cfg.model, 'aux_loss_weight', 0.0)})", 0)
            writer.add_text('Model/RouterZLossEnabled', f"{getattr(cfg.model, 'use_router_z_loss', False)} (Weight: {getattr(cfg.model, 'router_z_loss_weight', 0.0)})", 0)
    elif rank == 0:
        logger.info("-----------------------------------------------------------------------------------------------")
        logger.info("Standard GPT Model Configured.")
        if writer:
            writer.add_text('Model/Type', "Standard GPT", 0)


    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=cfg.train.ddp_find_unused_params)
        logger.info(f"Rank {rank} wrapped model with DDP.")
        raw_model = model.module
    else:
        raw_model = model

    scaler = torch.amp.GradScaler(enabled=(cfg.train.dtype == 'float16'))
    optimizer = configure_optimizers(raw_model, cfg.train.weight_decay, cfg.train.lr, (cfg.train.beta1, cfg.train.beta2), device.type)
    logger.info(f"Rank {rank} configured optimizer (Weight Decay: {cfg.train.weight_decay}, LR: {cfg.train.lr}) and scaler (enabled: {scaler.is_enabled()}).")
    if rank == 0: # Log optimizer details only once
        logger.info(f"Optimizer details: Num decayed params: {sum(p.numel() for group in optimizer.param_groups if group['weight_decay'] > 0 for p in group['params'])}, Num non-decayed params: {sum(p.numel() for group in optimizer.param_groups if group['weight_decay'] == 0 for p in group['params'])}")
        logger.info(f"Using fused AdamW: {'fused' in optimizer.param_groups[0] and optimizer.param_groups[0]['fused'] if optimizer.param_groups else 'N/A'}")


    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.train.dtype]
    autocast_enabled = (cfg.train.dtype != 'float32') and (device.type == 'cuda') # Autocast only for CUDA
    ctx = torch.amp.autocast(device_type=device.type, dtype=ptdtype, enabled=autocast_enabled)
    logger.info(f"Rank {rank} using autocast with dtype: {ptdtype} (enabled: {autocast_enabled})")

    iter_num = 0
    best_val_loss = 1e9
    if cfg.train.resume_from_checkpoint:
        ckpt_path = Path(cfg.train.resume_from_checkpoint)
        if ckpt_path.exists():
            logger.info(f"Rank {rank} attempting to resume from checkpoint: {ckpt_path}")
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank} if torch.cuda.is_available() else device
            checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
            try:
                # raw_model.load_state_dict(checkpoint['model']) # This might fail if config changed slightly
                # Load state dict more robustly
                model_state_dict = checkpoint['model']
                # Filter out unexpected keys (e.g. if model structure changed)
                current_model_keys = set(raw_model.state_dict().keys())
                filtered_state_dict = {k: v for k, v in model_state_dict.items() if k in current_model_keys}
                missing_keys, unexpected_keys = raw_model.load_state_dict(filtered_state_dict, strict=False)
                if missing_keys:
                    logger.warning(f"Rank {rank} - Missing keys when loading model state: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Rank {rank} - Unexpected keys in checkpoint model state: {unexpected_keys}")

                optimizer.load_state_dict(checkpoint['optimizer'])
                iter_num = checkpoint['iter_num']
                best_val_loss = checkpoint['best_val_loss']
                if 'scaler' in checkpoint and scaler is not None:
                    scaler.load_state_dict(checkpoint['scaler'])
                logger.info(f"Rank {rank} successfully resumed from iteration {iter_num} with best_val_loss {best_val_loss:.4f}")
            except Exception as e:
                logger.exception(f"Rank {rank} FAILED to load state dicts from checkpoint: {e}. Starting from scratch.")
                iter_num = 0
                best_val_loss = 1e9
        else:
            logger.warning(f"Rank {rank} - Checkpoint path not found: {ckpt_path}. Starting from scratch.")


    def get_batch_worker(split):
        loader = train_dataloader if split == "train" else val_dataloader
        x, y = next(loader)
        # y labels are already Long by default from CrossEntropy, but ensure targets are long for cross_entropy
        y = y.to(device, non_blocking=True).long() 
        if isinstance(x, (list, tuple)): # Should not happen with TimelineDataset current __getitem__
            return tuple(t.to(device, non_blocking=True) for t in x), y
        return x.to(device, non_blocking=True), y

    logger.info(f"Rank {rank} starting training loop from iteration {iter_num}...")
    t0 = time.time()
    local_iter_num = 0 

    # for profiler
    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    # Profiler configuration - create ONCE before the loop to avoid memory leaks
    # Only enable profiling for a limited number of iterations to capture representative data
    profile_enabled = getattr(cfg.train, 'enable_profiling', False)
    profile_start_iter = getattr(cfg.train, 'profile_start_iter', 10)
    profile_iterations = getattr(cfg.train, 'profile_iterations', 17)  # wait(10) + warmup(2) + active(5)
    
    if rank == 0 and profile_enabled:
        profile_schedule = torch.profiler.schedule(wait=10, warmup=2, active=5, repeat=1)
        train_profiler = profile(
            activities=activities,
            schedule=profile_schedule,
            on_trace_ready=tensorboard_trace_handler(str(profiler_log_dir / "train")),
            record_shapes=True,
            profile_memory=True,
            with_stack=False  # Disable stack traces to reduce memory overhead
        )
        train_profiler.__enter__()
        logger.info(f"Profiler enabled for iterations {profile_start_iter} to {profile_start_iter + profile_iterations}")
    else:
        train_profiler = None

    while iter_num < cfg.train.max_iters:
        lr = get_lr(iter_num, cfg.train) if cfg.train.decay_lr else cfg.train.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % cfg.train.eval_interval == 0 and cfg.train.eval_iters > 0:
            if rank == 0:
                logger.info(f"Evaluating at step {iter_num}...")
                eval_model = raw_model


                # Validation timing without per-call profiler to avoid memory leaks
                val_start_time = time.time()
                losses = estimate_loss(eval_model, ctx, get_batch_worker, cfg.train.eval_iters, tokens_of_interest, cfg.model)
                val_end_time = time.time()

                val_duration = val_end_time - val_start_time
                val_throughput = (cfg.train.eval_iters * batch_size_per_gpu * world_size) / val_duration if val_duration > 0 else 0
                val_latency = (val_duration * 1000) / (cfg.train.eval_iters * batch_size_per_gpu * world_size) if (cfg.train.eval_iters * batch_size_per_gpu * world_size) > 0 else 0

                # Pass the moe_hydra_config to estimate_loss if it needs MoE specific details
                # losses = estimate_loss(eval_model, ctx, get_batch_worker, cfg.train.eval_iters, tokens_of_interest, cfg.model)


                val_loss = losses.get('loss/val', float('nan')) # main validation loss
                logger.info(f"--- Eval Results (iter {iter_num}) ---")
                if writer:
                    for k, v_val in losses.items():
                        parts = k.split('/')
                        tag_prefix = parts[-1].capitalize() if len(parts) > 1 else "Misc"
                        tag_suffix = '/'.join(parts[:-1]) if len(parts) > 1 else k
                        writer.add_scalar(f"{tag_prefix}/{tag_suffix}", v_val, iter_num)
                        logger.info(f"{k}: {v_val:.4f}")


                        writer.add_scalar(f"Val/latency_ms_per_sample", val_latency, iter_num)
                        writer.add_scalar(f"Val/throughput_samples_sec", val_throughput, iter_num)
                        if device.type == 'cuda':
                            writer.add_scalar(f"Val/peak_memory_MB_gpu{rank}", torch.cuda.max_memory_allocated(device) / (1024*1024), iter_num)
                            torch.cuda.reset_peak_memory_stats(device) # Reset for next measurement
                    writer.flush()
                else:
                     for k, v_val in losses.items():
                          logger.info(f"{k}: {v_val:.4f}")
                     logger.info(f"Val/latency_ms_per_sample: {val_latency:.4f}")
                     logger.info(f"Val/throughput_samples_sec: {val_throughput:.2f}")
                
                # Log to MLflow (only on rank 0)
                if mlflow_run:
                    try:
                        mlflow.log_metrics({f"val_{k}": v_val for k, v_val in losses.items()}, step=iter_num)
                        mlflow.log_metrics({
                            "val_latency_ms_per_sample": val_latency,
                            "val_throughput_samples_sec": val_throughput
                        }, step=iter_num)
                        if device.type == 'cuda':
                            mlflow.log_metrics({
                                f"val_peak_memory_MB_gpu{rank}": torch.cuda.max_memory_allocated(device) / (1024*1024)
                            }, step=iter_num)
                    except Exception as e:
                        logger.warning(f"Failed to log metrics to MLflow: {e}")
                logger.info(f"--------------------")

                if cfg.train.save_checkpoints and not math.isnan(val_loss) and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if writer:
                        writer.add_scalar('Loss/best_val_main', best_val_loss, iter_num)
                    
                    # Log best validation loss to MLflow
                    if mlflow_run:
                        try:
                            mlflow.log_metric("best_val_loss", best_val_loss, step=iter_num)
                        except Exception as e:
                            logger.warning(f"Failed to log best_val_loss to MLflow: {e}")
                    
                    # # Get config from raw_model (GPT2LMNoBiasModel instance)
                    # # It stores base_config and moe_config
                    # save_model_config = {
                    #     'base_config': OmegaConf.to_container(raw_model.base_config, resolve=True), # GPT2Config to dict
                    #     'moe_config': OmegaConf.to_container(raw_model.moe_config, resolve=True) # Hydra config for MoE
                    # }

                    checkpoint = {
                        'iter_num': iter_num,
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict() if scaler is not None else None,
                        'best_val_loss': best_val_loss,
                        'model_configs': cfg.model, # Store both configs
                        'vocab_stoi': vocab.stoi, # Save vocab string-to-int
                        'hydra_config_full': OmegaConf.to_container(cfg, resolve=True) # Full training hydra config
                    }
                    ckpt_path = Path(cfg.train.out_dir) / "best_model.pt"
                    logger.info(f"Saving checkpoint to {ckpt_path} (Val Loss: {best_val_loss:.4f})")
                    torch.save(checkpoint, ckpt_path)

                    recent_ckpt_path = Path(cfg.train.out_dir) / f"ckpt_{iter_num}.pt"
                    logger.info(f"Saving recent checkpoint to {recent_ckpt_path}")
                    torch.save(checkpoint, recent_ckpt_path)
                    
                    # Log checkpoints as artifacts to MLflow
                    if mlflow_run:
                        try:
                            mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")
                            mlflow.log_artifact(str(recent_ckpt_path), artifact_path="checkpoints")
                            logger.info(f"Uploaded checkpoints to MLflow: best_model.pt, ckpt_{iter_num}.pt")
                        except Exception as e:
                            logger.warning(f"Failed to log checkpoint to MLflow: {e}")
            if world_size > 1:
                dist.barrier()
            
            # Clear CUDA cache after validation to release fragmented memory
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        model.train()

        optimizer.zero_grad(set_to_none=True)
        
        accumulated_loss_value = 0.0 # For logging the accumulated main loss
        accumulated_raw_aux_loss = None
        accumulated_raw_router_z_loss = None
        iter_start_time = time.time()

        for micro_step in range(cfg.train.gradient_accumulation_steps):
            # DDP sync for last micro_step or if not using DDP
            sync_context = model.no_sync if (world_size > 1 and (micro_step < cfg.train.gradient_accumulation_steps - 1)) else nullcontext
            with sync_context():
                with ctx:
                    X, Y = get_batch_worker("train")
                    output: ModelOutput = model(X, Y) # output is ModelOutput
                    loss = output.loss 
                    loss = loss / cfg.train.gradient_accumulation_steps # Normalize loss for accumulation

                    accumulated_loss_value += loss.item() * cfg.train.gradient_accumulation_steps # De-normalize for logging sum
                    # Accumulate raw aux losses for logging (average over micro_steps)
                    if output.aux_loss is not None:
                        if accumulated_raw_aux_loss is None: accumulated_raw_aux_loss = 0.0
                        accumulated_raw_aux_loss += output.aux_loss.item()
                    if output.router_z_loss is not None:
                        if accumulated_raw_router_z_loss is None: accumulated_raw_router_z_loss = 0.0
                        accumulated_raw_router_z_loss += output.router_z_loss.item()
                    scaler.scale(loss).backward()

        if cfg.train.grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
    
        scaler.step(optimizer)
        scaler.update()
        iter_end_time = time.time()
        
        # Step the profiler if enabled (uses the single profiler created before the loop)
        if rank == 0 and train_profiler is not None:
            train_profiler.step()


        if rank == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1


            iter_duration = iter_end_time - iter_start_time
            samples_processed_this_iter = cfg.train.gradient_accumulation_steps * batch_size_per_gpu * world_size
            train_throughput_iter = samples_processed_this_iter / iter_duration if iter_duration > 0 else 0
            train_latency_iter = (iter_duration * 1000) / samples_processed_this_iter if samples_processed_this_iter > 0 else 0

            
            if iter_num % cfg.train.log_interval == 0:
                # Log the average loss over accumulation steps
                avg_loss_this_step = accumulated_loss_value / cfg.train.gradient_accumulation_steps
                logger.info(f"iter {iter_num}: train_loss {avg_loss_this_step:.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}")
                if writer:
                    writer.add_scalar('Loss/train_main', avg_loss_this_step, iter_num)
                    writer.add_scalar('Meta/learning_rate', lr, iter_num)
                    writer.add_scalar('Meta/iter_time_ms_overall', dt*1000, iter_num) # This is overall wall time
                    writer.add_scalar('Train/latency_ms_per_sample_iter', train_latency_iter, iter_num)
                    writer.add_scalar('Train/throughput_samples_sec_iter', train_throughput_iter, iter_num)
                    
                    if device.type == 'cuda':
                        writer.add_scalar(f"Train/peak_memory_MB_gpu{rank}", torch.cuda.max_memory_allocated(device) / (1024*1024), iter_num)
                        torch.cuda.reset_peak_memory_stats(device) # Reset for next measurement

                        
                    if is_moe_model: # Log raw (unweighted) aux losses averaged over micro_steps
                        if accumulated_raw_aux_loss is not None:
                            avg_raw_aux_loss = accumulated_raw_aux_loss / cfg.train.gradient_accumulation_steps
                            writer.add_scalar('MoE/aux_loss_raw_train', avg_raw_aux_loss, iter_num)
                            # Log weighted contribution as well for verification
                            weighted_aux = avg_raw_aux_loss * getattr(cfg.model, 'aux_loss_weight', 0.01)
                            writer.add_scalar('MoE/aux_loss_weighted_train', weighted_aux, iter_num)
                        if accumulated_raw_router_z_loss is not None:
                            avg_raw_router_z_loss = accumulated_raw_router_z_loss / cfg.train.gradient_accumulation_steps
                            writer.add_scalar('MoE/router_z_loss_raw_train', avg_raw_router_z_loss, iter_num)
                            weighted_rz = avg_raw_router_z_loss * getattr(cfg.model, 'router_z_loss_weight', 0.001)
                            writer.add_scalar('MoE/router_z_loss_weighted_train', weighted_rz, iter_num)
                    writer.flush()
                
                # Log to MLflow
                if mlflow_run:
                    try:
                        mlflow.log_metrics({
                            "train_loss": avg_loss_this_step,
                            "learning_rate": lr,
                            "iter_time_ms_overall": dt*1000,
                            "train_latency_ms_per_sample": train_latency_iter,
                            "train_throughput_samples_sec": train_throughput_iter
                        }, step=iter_num)
                        if device.type == 'cuda':
                            mlflow.log_metrics({
                                f"train_peak_memory_MB_gpu{rank}": torch.cuda.max_memory_allocated(device) / (1024*1024)
                            }, step=iter_num)
                        if is_moe_model:
                            if accumulated_raw_aux_loss is not None:
                                avg_raw_aux_loss = accumulated_raw_aux_loss / cfg.train.gradient_accumulation_steps
                                mlflow.log_metrics({
                                    "moe_aux_loss_raw_train": avg_raw_aux_loss,
                                    "moe_aux_loss_weighted_train": avg_raw_aux_loss * getattr(cfg.model, 'aux_loss_weight', 0.01)
                                }, step=iter_num)
                            if accumulated_raw_router_z_loss is not None:
                                avg_raw_router_z_loss = accumulated_raw_router_z_loss / cfg.train.gradient_accumulation_steps
                                mlflow.log_metrics({
                                    "moe_router_z_loss_raw_train": avg_raw_router_z_loss,
                                    "moe_router_z_loss_weighted_train": avg_raw_router_z_loss * getattr(cfg.model, 'router_z_loss_weight', 0.001)
                                }, step=iter_num)
                    except Exception as e:
                        logger.warning(f"Failed to log metrics to MLflow: {e}")

        iter_num += 1
        local_iter_num += 1

        if iter_num >= cfg.train.max_iters:
             logger.info(f"Rank {rank} reached max iterations ({cfg.train.max_iters}).")
             break

    # Clean up profiler if it was enabled
    if rank == 0 and train_profiler is not None:
        train_profiler.__exit__(None, None, None)
        logger.info("Profiler stopped and traces saved.")

    if rank == 0 and cfg.train.save_checkpoints:
        # save_model_config = {
        #     'base_config': OmegaConf.to_container(raw_model.base_config.to_dict_recursive(), resolve=True),
        #     'moe_config': OmegaConf.to_container(raw_model.moe_config, resolve=True)
        # }
        final_checkpoint = {
            'iter_num': iter_num,
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict() if scaler is not None else None,
            'best_val_loss': best_val_loss,
            'model_configs': cfg.model,
            'vocab_stoi': vocab.stoi,
            'hydra_config_full': OmegaConf.to_container(cfg, resolve=True)
        }
        final_ckpt_path = Path(cfg.train.out_dir) / "final_model.pt"
        logger.info(f"Saving final model checkpoint to {final_ckpt_path}")
        torch.save(final_checkpoint, final_ckpt_path)

    if rank == 0 and writer:
        writer.close()
    
    if rank == 0 and mlflow_run:
        try:
            mlflow.end_run()
            logger.info("MLflow run ended successfully.")
        except Exception as e:
            logger.warning(f"Failed to end MLflow run: {e}")

    cleanup()
    logger.info(f"Rank {rank} finished training worker.")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main_hydra(cfg: DictConfig):
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

    logger.info("--- Hydra Configuration ---")
    # Print a sanitized version if there are secrets, otherwise full
    logger.info(f"\n{OmegaConf.to_yaml(cfg, resolve=True)}") # Resolve interpolations for clarity
    logger.info("--------------------------")

    curr_dir = os.getcwd() # Hydra sets current working directory to output subdir
    logger.info(f"Hydra output directory: {curr_dir}")
    
    # Allow modifications to cfg now, like setting out_dir
    OmegaConf.set_struct(cfg, False) 
    cfg.train.out_dir = curr_dir
    OmegaConf.set_struct(cfg, True) # Re-lock if needed, though often not necessary after this point

    world_size = cfg.get("world_size", 1) # Default to 1 if not specified
    if world_size == 'auto': # 'auto' or similar to detect GPUs
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    elif isinstance(world_size, str) and world_size.isdigit():
        world_size = int(world_size)
    
    available_gpus = torch.cuda.device_count()

    if world_size > available_gpus and available_gpus > 0 : # Only warn if GPUs are available but fewer than requested
         logger.warning(f"Requested world_size {world_size} > available GPUs {available_gpus}. Setting to {available_gpus}.")
         world_size = available_gpus
    elif available_gpus == 0 and world_size > 1:
        logger.warning(f"No GPUs available. Requested world_size {world_size} is not possible for DDP. Running on CPU with world_size=1.")
        world_size = 1
    
    if world_size <= 1:
        if available_gpus == 0:
            logger.info("No GPUs detected. Running on CPU (world_size=1, rank=0).")
        else:
            logger.info("Running on a single GPU or world_size set to 1 (world_size=1, rank=0).")
        train_worker(0, 1, cfg)
    else:
         logger.info(f"Spawning {world_size} processes for DDP training.")
         mp.spawn(train_worker, args=(world_size, cfg), nprocs=world_size, join=True)

    logger.info("Main process finished.")


if __name__ == "__main__":
    main_hydra()