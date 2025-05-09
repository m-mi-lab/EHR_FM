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
from torch.utils.tensorboard import SummaryWriter

from src.tokenizer.constants import SpecialToken as ST
from src.tokenizer.datasets import TimelineDataset
from src.tokenizer.vocabulary import Vocabulary # Assuming Vocabulary can be loaded this way
from src.metrics import estimate_loss
from src.manager import MANAGER
from model import GPT2LMNoBiasModel

ModelOutput = namedtuple("ModelOutput", ["loss", "logits"])


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    # Use logger instead of print
    # logger.info is called inside train_worker after setup to ensure rank is known
    # print(f"Rank {rank}/{world_size} initialized process group with backend '{backend}'.") # Keep print here as logger might not be configured yet


def cleanup():
    dist.destroy_process_group()
    # Use logger instead of print
    logger.info("Cleaned up process group.")

def get_lr(it, cfg_training: DictConfig):
    # 1) linear warmup for warmup_iters steps
    if it < cfg_training.warmup_iters:
        return cfg_training.lr * it / cfg_training.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > cfg_training.lr_decay_iters:
        return cfg_training.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - cfg_training.warmup_iters) / (cfg_training.lr_decay_iters - cfg_training.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return cfg_training.min_lr + coeff * (cfg_training.lr - cfg_training.min_lr)

def configure_optimizers(model: torch.nn.Module, weight_decay, learning_rate, betas, device_type):
    param_dict = {pn: p for pn, p in model.named_parameters()} # start with all and filter later
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # weight decay for >=2 dims
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    # Log using logger after it's configured
    # print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    # print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    # print(f"using fused AdamW: {use_fused}") # Log later
    return optimizer

# --- Add make_infinite_loader back ---
def make_infinite_loader(loader):
    while True:
        yield from iter(loader)
# --- End Add ---


def train_worker(rank, world_size, cfg: DictConfig):
    # --- Configure Loguru ---
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
    # --- End Loguru Config ---

    logger.info(f"Running DDP training worker on rank {rank}.")
    setup(rank, world_size) # Initialize DDP
    # Now log the DDP setup message
    logger.info(f"Rank {rank}/{world_size} initialized process group.")


    # --- Initialize Tensorboard Writer (only rank 0) ---
    writer = None
    if rank == 0:
        tb_log_dir = Path(cfg.train.out_dir) / "tensorboard_logs"
        writer = SummaryWriter(log_dir=str(tb_log_dir))
        logger.info(f"Tensorboard logs will be saved to: {tb_log_dir}")
    # --- End Tensorboard Init ---

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    logger.info(f"Rank {rank} using device: {device}")

    # Load vocab
    try:
        vocab_dir = Path(cfg.data.tokenized_dir)
        vocab = Vocabulary.from_path(vocab_dir)
        vocab_size = (len(vocab) // 64 + 1) * 64 if len(vocab) % 64 != 0 else len(vocab)
        logger.info(f"Rank {rank} loaded vocabulary with size {len(vocab)} (padded: {vocab_size}).")
        tokens_of_interest = {stoken: vocab.encode(stoken) for stoken in [ST.DEATH, ST.ADMISSION, ST.DISCHARGE] if stoken in vocab.stoi}
        logger.info(f"Rank {rank} Tokens of Interest: {tokens_of_interest}")
    except Exception as e:
        logger.exception(f"Rank {rank} FAILED to load vocabulary from {cfg.data.tokenized_dir}: {e}")
        cleanup()
        return

    # Load dataset
    try:
        full_dataset = TimelineDataset(
            input_dir=cfg.data.tokenized_dir,
            n_positions=cfg.model.n_positions,
            is_encoder_decoder=False,
        )
        logger.info(f"Rank {rank} loaded full dataset with {len(full_dataset)} samples.")

        val_split_count = int(cfg.data.val_split_fraction * len(full_dataset))
        train_split_count = len(full_dataset) - val_split_count
        generator = torch.Generator().manual_seed(42)
        train_indices, val_indices = torch.utils.data.random_split(
            range(len(full_dataset)), [train_split_count, val_split_count], generator=generator
        )
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        logger.info(f"Rank {rank} - Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        batch_size_per_gpu = cfg.train.batch_size // (world_size * cfg.train.gradient_accumulation_steps)
        if batch_size_per_gpu < 1:
             logger.warning(f"Calculated batch_size_per_gpu is {batch_size_per_gpu}. Setting to 1.")
             batch_size_per_gpu = 1
        logger.info(f"Rank {rank} - Batch size per GPU: {batch_size_per_gpu}")


        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_per_gpu, sampler=train_sampler, num_workers=cfg.train.num_workers, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size_per_gpu, sampler=val_sampler, num_workers=cfg.train.num_workers, pin_memory=True)

        # --- Use make_infinite_loader ---
        train_dataloader = make_infinite_loader(train_dataloader)
        val_dataloader = make_infinite_loader(val_dataloader)
        # --- End Use ---
        logger.info(f"Rank {rank} created DataLoaders.")

    except Exception as e:
        logger.exception(f"Rank {rank} FAILED during data loading: {e}")
        import traceback
        traceback.print_exc()
        cleanup()
        return


    # Model setup
    if "_target_" in cfg.model:
         model_config = hydra.utils.instantiate(cfg.model, vocab_size=vocab_size, _recursive_=False)
         logger.info(f"Rank {rank} instantiated model config from target: {cfg.model._target_}")
    else:
         model_config = GPT2Config(
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
         )
         logger.info(f"Rank {rank} manually created model config.")

    model = GPT2LMNoBiasModel(model_config).to(device)
    logger.info(f"Rank {rank} initialized model.")

    # DDP Wrapping
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=cfg.train.ddp_find_unused_params)
        logger.info(f"Rank {rank} wrapped model with DDP.")
        raw_model = model.module
    else:
        raw_model = model

    # Optimizer and Scaler
    scaler = torch.amp.GradScaler(enabled=(cfg.train.dtype == 'float16'))
    optimizer = configure_optimizers(raw_model, cfg.train.weight_decay, cfg.train.lr, (cfg.train.beta1, cfg.train.beta2), device.type)
    logger.info(f"Rank {rank} configured optimizer (Weight Decay: {cfg.train.weight_decay}, LR: {cfg.train.lr}) and scaler (enabled: {scaler.is_enabled()}).")
    # Log optimizer details
    logger.info(f"Optimizer details: Num decayed params: {sum(p.numel() for group in optimizer.param_groups if group['weight_decay'] > 0 for p in group['params'])}, Num non-decayed params: {sum(p.numel() for group in optimizer.param_groups if group['weight_decay'] == 0 for p in group['params'])}")
    logger.info(f"Using fused AdamW: {'fused' in optimizer.param_groups[0] and optimizer.param_groups[0]['fused']}")


    # Autocast context
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.train.dtype]
    autocast_enabled = (cfg.train.dtype != 'float32')
    ctx = torch.amp.autocast(device_type=device.type, dtype=ptdtype, enabled=autocast_enabled)
    logger.info(f"Rank {rank} using autocast with dtype: {ptdtype} (enabled: {autocast_enabled})")

    # Checkpoint loading
    iter_num = 0
    best_val_loss = 1e9
    if cfg.train.resume_from_checkpoint:
        ckpt_path = Path(cfg.train.resume_from_checkpoint)
        if ckpt_path.exists():
            logger.info(f"Rank {rank} attempting to resume from checkpoint: {ckpt_path}")
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank} if torch.cuda.is_available() else device
            checkpoint = torch.load(ckpt_path, map_location=map_location)
            try:
                raw_model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                iter_num = checkpoint['iter_num']
                best_val_loss = checkpoint['best_val_loss']
                if 'scaler' in checkpoint:
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
        y = y.to(device, non_blocking=True)
        if isinstance(x, (list, tuple)):
            return tuple(t.to(device, non_blocking=True) for t in x), y
        return x.to(device, non_blocking=True), y

    logger.info(f"Rank {rank} starting training loop from iteration {iter_num}...")
    t0 = time.time()
    local_iter_num = 0 # Tracks iterations within this run

    # --- Training Loop ---
    while iter_num < cfg.train.max_iters:
        lr = get_lr(iter_num, cfg.train) if cfg.train.decay_lr else cfg.train.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluation and Checkpointing (Rank 0 Only)
        if iter_num % cfg.train.eval_interval == 0 and cfg.train.eval_iters > 0:
            if rank == 0:
                logger.info(f"Evaluating at step {iter_num}...")
                # Ensure raw_model is used if DDP is active
                eval_model = raw_model if world_size > 1 else model
                losses = estimate_loss(eval_model, ctx, get_batch_worker, cfg.train.eval_iters, tokens_of_interest)
                val_loss = losses.get('loss/val', float('nan'))
                logger.info(f"--- Eval Results (iter {iter_num}) ---")
                if writer:
                    for k, v in losses.items():
                        # Group metrics by split (train/val) and then metric type (loss, acc_top)
                        parts = k.split('/')
                        if len(parts) >= 2:
                             tag = f"{parts[-1].capitalize()}/{'/'.join(parts[:-1])}" # e.g., Val/loss, Val/acc_top/all/1023/k=1
                        else:
                             tag = f"Misc/{k}"
                        writer.add_scalar(tag, v, iter_num)
                        logger.info(f"{k}: {v:.4f}")
                    writer.flush()
                else:
                     for k, v in losses.items():
                          logger.info(f"{k}: {v:.4f}")
                logger.info(f"--------------------")

                if cfg.train.save_checkpoints and not math.isnan(val_loss) and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if writer:
                        writer.add_scalar('Loss/best_val', best_val_loss, iter_num) # Simplified tag
                    checkpoint = {
                        'iter_num': iter_num,
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'model_config': raw_model.config, # Use config from potentially wrapped model
                        'vocab': vocab.stoi,
                        'hydra_config': OmegaConf.to_container(cfg, resolve=True)
                    }
                    ckpt_path = Path(cfg.train.out_dir) / "best_model.pt"
                    logger.info(f"Saving checkpoint to {ckpt_path} (Val Loss: {best_val_loss:.4f})")
                    torch.save(checkpoint, ckpt_path)

                    recent_ckpt_path = Path(cfg.train.out_dir) / f"ckpt_{iter_num}.pt"
                    logger.info(f"Saving recent checkpoint to {recent_ckpt_path}")
                    torch.save(checkpoint, recent_ckpt_path)

            if world_size > 1:
                dist.barrier()

        # Training Step (All Ranks)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss_accum = 0.0

        for micro_step in range(cfg.train.gradient_accumulation_steps):
            # DDP handles gradient sync automatically in .backward() when no_sync isn't used
            with ctx:
                X, Y = get_batch_worker("train")
                output = model(X, Y)
                loss = output.loss
                loss = loss / cfg.train.gradient_accumulation_steps
                total_loss_accum += loss.item()

            model_config = model.module.config if world_size > 1 else model.config
            is_moe_model = hasattr(model_config, 'n_experts') and model_config.n_experts > 1
            if is_moe_model and model.training:
             if getattr(model_config, 'use_aux_loss', False):
                  aux_loss = MANAGER.aggregate_aux_loss()
                  if aux_loss is not None: # Ensure loss was recorded
                       loss += getattr(model_config, 'aux_loss_weight', 0.01) * aux_loss
                  MANAGER.reset_aux_loss() # Reset for next accumulation cycle

             if getattr(model_config, 'use_router_z_loss', False):
                  router_z_loss = MANAGER.aggregate_router_z_loss()
                  if router_z_loss is not None: # Ensure loss was recorded
                       loss += getattr(model_config, 'router_z_loss_weight', 0.001) * router_z_loss
                  MANAGER.reset_router_z_loss() # Reset for next accumulation cycle
            total_loss_accum += loss.item()
            scaler.scale(loss).backward()

        if cfg.train.grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # Logging (Rank 0 Only)
        if rank == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % cfg.train.log_interval == 0:
                lossf = total_loss_accum
                logger.info(f"iter {iter_num}: train_loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}")
                if writer:
                    writer.add_scalar('Loss/train', lossf, iter_num)
                    writer.add_scalar('Meta/learning_rate', lr, iter_num)
                    # Optionally log timing
                    writer.add_scalar('Meta/iter_time_ms', dt*1000, iter_num)
                    writer.flush()

        iter_num += 1
        local_iter_num += 1

        if iter_num >= cfg.train.max_iters:
             logger.info(f"Rank {rank} reached max iterations ({cfg.train.max_iters}).")
             break
    # --- End Training Loop ---

    # Final Checkpoint Saving (Rank 0 Only)
    if rank == 0 and cfg.train.save_checkpoints:
        final_checkpoint = {
            'iter_num': iter_num,
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'model_config': raw_model.config, # Use config from potentially wrapped model
            'vocab': vocab.stoi,
            'hydra_config': OmegaConf.to_container(cfg, resolve=True)
        }
        final_ckpt_path = Path(cfg.train.out_dir) / "final_model.pt"
        logger.info(f"Saving final model checkpoint to {final_ckpt_path}")
        torch.save(final_checkpoint, final_ckpt_path)

    if rank == 0 and writer:
        writer.close()

    cleanup()
    logger.info(f"Rank {rank} finished training worker.")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main_hydra(cfg: DictConfig):

    # --- Configure Loguru for main process ---
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")
    # --- End Loguru Config ---

    logger.info("--- Hydra Configuration ---")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")
    logger.info("--------------------------")


    curr_dir = os.getcwd()
    logger.info(f"Hydra output directory: {curr_dir}")
    OmegaConf.set_struct(cfg, False)
    cfg.train.out_dir = curr_dir
    OmegaConf.set_struct(cfg, True)


    world_size = cfg.get("world_size", torch.cuda.device_count())
    available_gpus = torch.cuda.device_count()

    if world_size > available_gpus:
         logger.warning(f"Requested world_size {world_size} > available GPUs {available_gpus}. Setting to {available_gpus}.")
         world_size = available_gpus

    # Adjust logic for spawning processes
    if world_size <= 1:
        if available_gpus == 0:
            logger.info("No GPUs detected. Running on CPU.")
            train_worker(0, 1, cfg) # Run directly on CPU (rank 0, world_size 1)
        else:
            logger.info("Running on a single GPU.")
            train_worker(0, 1, cfg) # Run directly on GPU 0 (rank 0, world_size 1)
            # # Alternatively, spawn 1 process if strict DDP setup is desired even for 1 GPU
            # logger.info("Running on a single GPU. Spawning one process.")
            # mp.spawn(train_worker, args=(1, cfg), nprocs=1, join=True)
    else:
         logger.info(f"Spawning {world_size} processes for DDP training.")
         mp.spawn(train_worker, args=(world_size, cfg), nprocs=world_size, join=True)


    logger.info("Main process finished.")


if __name__ == "__main__":
    main_hydra()