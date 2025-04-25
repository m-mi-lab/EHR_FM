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
import transformers.activations

from src.tokenizer.constants import SpecialToken as ST
from src.tokenizer.datasets import TimelineDataset
from src.tokenizer.vocabulary import Vocabulary # Assuming Vocabulary can be loaded this way
from src.metrics import estimate_loss


ModelOutput = namedtuple("ModelOutput", ["loss", "logits"])

class CausalSelfAttention(torch.nn.Module):
    def __init__(self, config, attention_weights: list | None = None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = torch.nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = torch.nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = torch.nn.Dropout(config.attn_pdrop)
        self.resid_dropout = torch.nn.Dropout(config.resid_pdrop)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.attn_pdrop
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash or attention_weights is not None:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Register buffer correctly
            bias = torch.tril(torch.ones(config.n_positions, config.n_positions)).view(
                    1, 1, config.n_positions, config.n_positions
                )
            self.register_buffer("bias", bias, persistent=False)

        self.attention_weights = attention_weights # Store passed list reference

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash and self.attention_weights is None:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # Use the registered buffer
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = torch.nn.functional.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
            if self.attention_weights is not None: # Check if list exists
                 self.attention_weights.append(att.detach().cpu())
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = torch.nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.activation = transformers.activations.get_activation(config.activation_function)
        self.c_proj = torch.nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = torch.nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(torch.nn.Module):
    def __init__(self, config, attention_weights: list | None = None):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, attention_weights=attention_weights)
        self.ln_2 = torch.nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2LMNoBiasModel(torch.nn.Module):
    def __init__(self, config: GPT2Config, return_attention=False):
        super().__init__()
        self.config = config # Store config
        self.return_attention = return_attention
        self.attention_weights = [] if return_attention else None

        self.transformer = torch.nn.ModuleDict(dict(
            wte=torch.nn.Embedding(config.vocab_size, config.n_embd),
            wpe=torch.nn.Embedding(config.n_positions, config.n_embd),
            drop=torch.nn.Dropout(config.embd_pdrop),
            # Pass the potentially existing list to Block
            h=torch.nn.ModuleList([Block(config, self.attention_weights) for _ in range(config.n_layer)]),
            ln_f=torch.nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        pos = torch.arange(0, config.n_positions, dtype=torch.long)
        self.register_buffer("pos", pos, persistent=False)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, exclude_embeddings=True):
        n_params = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, input_ids, labels=None) -> ModelOutput:
        b, t = input_ids.size()
        if t > self.config.n_positions: # Use stored config
             raise ValueError(f"Cannot forward sequence of length {t}, block size is only {self.config.n_positions}")

        if self.return_attention and self.attention_weights is not None:
            self.attention_weights.clear()

        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(self.pos[:t])
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if labels is not None:
            logits = self.lm_head(x)
            # Ensure labels are long type and handle ignore_index if necessary
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1).long(), ignore_index=-100)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return ModelOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def get_next_token(self, x: torch.Tensor, return_probs: bool = False, top_k: int | None = None):
        logits = self(x).logits # Call forward
        logits = logits[:, -1, :] # Get the last token logits
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf") # Apply top-k filtering
        probs = torch.nn.functional.softmax(logits, dim=-1) # Get probabilities
        next_token = torch.multinomial(probs, num_samples=1) # Sample next token
        if return_probs:
            return next_token, probs
        return next_token


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    # Use 'nccl' backend for NVIDIA GPUs
    # Use 'gloo' backend for CPU training or if nccl is not available
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    print(f"Rank {rank}/{world_size} initialized process group with backend '{backend}'.")


def cleanup():
    dist.destroy_process_group()
    print("Cleaned up process group.")

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
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")
    return optimizer

def make_infinite_loader(loader):
    while True:
        yield from iter(loader)
        
        
def train_worker(rank, world_size, cfg: DictConfig):
    print(f"Running DDP training worker on rank {rank}.")
    setup(rank, world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    print(f"Rank {rank} using device: {device}")

    # load vocab for all ranks / processes ? 
    try:
        vocab_dir = Path(cfg.data.tokenized_dir)
        vocab = Vocabulary.from_path(vocab_dir)
        vocab_size = (len(vocab) // 64 + 1) * 64 if len(vocab) % 64 != 0 else len(vocab)
        print(f"Rank {rank} loaded vocabulary with size {len(vocab)} (padded: {vocab_size}).")
        tokens_of_interest = {stoken: vocab.encode(stoken) for stoken in [ST.DEATH, ST.ADMISSION, ST.DISCHARGE] if stoken in vocab.stoi}
        print(f"Rank {rank} Tokens of Interest: {tokens_of_interest}")
    except Exception as e:
        print(f"Rank {rank} FAILED to load vocabulary from {cfg.data.tokenized_dir}: {e}")
        cleanup()
        return 

    try:
        full_dataset = TimelineDataset(
            input_dir=cfg.data.tokenized_dir,
            n_positions=cfg.model.n_positions,
            is_encoder_decoder=False,
        )
        print(f"Rank {rank} loaded full dataset with {len(full_dataset)} samples.")

        val_split_count = int(cfg.data.val_split_fraction * len(full_dataset))
        train_split_count = len(full_dataset) - val_split_count
        generator = torch.Generator().manual_seed(42) # Use a fixed seed for splitting
        train_indices, val_indices = torch.utils.data.random_split(
            range(len(full_dataset)), [train_split_count, val_split_count], generator=generator
        )
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        print(f"Rank {rank} - Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        # optimal??? batch size = batch_size_per_gpu * world_size * gradient_accumulation_steps
        batch_size_per_gpu = cfg.train.batch_size // (world_size * cfg.train.gradient_accumulation_steps)
        if batch_size_per_gpu < 1:
             print(f"WARNING: Calculated batch_size_per_gpu is {batch_size_per_gpu}. Setting to 1. Adjust total batch size, world size, or grad accum steps.")
             batch_size_per_gpu = 1
        print(f"Rank {rank} - Batch size per GPU: {batch_size_per_gpu}")


        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_per_gpu, sampler=train_sampler, num_workers=cfg.train.num_workers, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size_per_gpu, sampler=val_sampler, num_workers=cfg.train.num_workers, pin_memory=True)

        train_dataloader = make_infinite_loader(train_dataloader)
        val_dataloader = make_infinite_loader(val_dataloader)
        print(f"Rank {rank} created DataLoaders.")

    except Exception as e:
        print(f"Rank {rank} FAILED during data loading: {e}")
        import traceback
        traceback.print_exc()
        cleanup()
        return # Exit if data loading fails


    # check if _target_ is defined
    if "_target_" in cfg.model:
         model_config = hydra.utils.instantiate(cfg.model, vocab_size=vocab_size, _recursive_=False)
         print(f"Rank {rank} instantiated model config from target: {cfg.model._target_}")
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
         print(f"Rank {rank} manually created model config.")

    model = GPT2LMNoBiasModel(model_config).to(device)
    print(f"Rank {rank} initialized model.")

    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=cfg.train.ddp_find_unused_params) # Set find_unused_parameters based on config
        print(f"Rank {rank} wrapped model with DDP.")
        raw_model = model.module # Get underlying model for saving etc.
    else:
        raw_model = model # No DDP wrapping needed

    scaler = torch.amp.GradScaler(enabled=(cfg.train.dtype == 'float16'))
    optimizer = configure_optimizers(model, cfg.train.weight_decay, cfg.train.lr, (cfg.train.beta1, cfg.train.beta2), device.type)
    print(f"Rank {rank} configured optimizer and scaler (enabled: {scaler.is_enabled()}).")

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.train.dtype]
    autocast_enabled = (cfg.train.dtype != 'float32')
    ctx = torch.amp.autocast(device_type=device.type, dtype=ptdtype, enabled=autocast_enabled)
    print(f"Rank {rank} using autocast with dtype: {ptdtype} (enabled: {autocast_enabled})")

    # ctx = torch.amp.autocast(device_type=device.type, dtype=ptdtype, enabled=(cfg.train.dtype != 'float32'))
    # print(f"Rank {rank} using autocast with dtype: {ptdtype} (enabled: {ctx.is_enabled()})")

    # ckpt loading
    iter_num = 0
    best_val_loss = 1e9
    if cfg.train.resume_from_checkpoint:
        ckpt_path = Path(cfg.train.resume_from_checkpoint)
        if ckpt_path.exists():
            print(f"Rank {rank} attempting to resume from checkpoint: {ckpt_path}")
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank} if torch.cuda.is_available() else device
            checkpoint = torch.load(ckpt_path, map_location=map_location)
            try:
                raw_model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                iter_num = checkpoint['iter_num']
                best_val_loss = checkpoint['best_val_loss']
                if 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
                print(f"Rank {rank} successfully resumed from iteration {iter_num} with best_val_loss {best_val_loss:.4f}")
            except Exception as e:
                print(f"Rank {rank} FAILED to load state dicts from checkpoint: {e}. Starting from scratch.")
                iter_num = 0
                best_val_loss = 1e9
        else:
            print(f"Rank {rank} - Checkpoint path not found: {ckpt_path}. Starting from scratch.")


    def get_batch_worker(split):
        loader = train_dataloader if split == "train" else val_dataloader
        x, y = next(loader) # Sampler handles distribution
        y = y.to(device, non_blocking=True)
        if isinstance(x, (list, tuple)): # Handle potential tuple/list input from dataset
            return tuple(t.to(device, non_blocking=True) for t in x), y
        return x.to(device, non_blocking=True), y

    print(f"Rank {rank} starting training loop from iteration {iter_num}...")
    t0 = time.time()
    local_iter_num = 0 # Tracks iterations within this run

    while iter_num < cfg.train.max_iters:
        lr = get_lr(iter_num, cfg.train) if cfg.train.decay_lr else cfg.train.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % cfg.train.eval_interval == 0 and cfg.train.eval_iters > 0:
            if rank == 0:
                print(f"\nRank {rank} evaluating at step {iter_num}...")
                # Pass the underlying model (raw_model) to estimate_loss if needed
                # estimate_loss should handle the device placement internally or via get_batch_worker
                losses = estimate_loss(raw_model, ctx, get_batch_worker, cfg.train.eval_iters, tokens_of_interest)
                val_loss = losses.get('loss/val', float('nan'))
                print(f"--- Eval Results ---")
                for k, v in losses.items():
                     print(f"{k}: {v:.4f}")
                print(f"--------------------")


                if cfg.train.save_checkpoints and not math.isnan(val_loss) and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint = {
                        'iter_num': iter_num,
                        'model': raw_model.state_dict(), # Save the underlying model's state_dict
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(), # Save scaler state
                        'best_val_loss': best_val_loss,
                        'model_config': raw_model.config, # Save original model config
                        'vocab': vocab.stoi,
                        'hydra_config': OmegaConf.to_container(cfg, resolve=True) # Save full config
                    }
                    ckpt_path = Path(cfg.train.out_dir) / "best_model.pt"
                    print(f"Saving checkpoint to {ckpt_path} (Val Loss: {best_val_loss:.4f})")
                    torch.save(checkpoint, ckpt_path)

                    recent_ckpt_path = Path(cfg.train.out_dir) / f"ckpt_{iter_num}.pt"
                    print(f"Saving recent checkpoint to {recent_ckpt_path}")
                    torch.save(checkpoint, recent_ckpt_path)


            if world_size > 1:
                dist.barrier()

        model.train() # Ensure model is in training mode
        optimizer.zero_grad(set_to_none=True)
        total_loss_accum = 0.0 

        for micro_step in range(cfg.train.gradient_accumulation_steps):
            is_sync_step = (micro_step == cfg.train.gradient_accumulation_steps - 1)

            with ctx:
                X, Y = get_batch_worker("train")
                output = model(X, Y)
                loss = output.loss
                loss = loss / cfg.train.gradient_accumulation_steps
                total_loss_accum += loss.item() # Accumulate item for logging
            scaler.scale(loss).backward()

        if cfg.train.grad_clip > 0.0:
            scaler.unscale_(optimizer) # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer) # Optimizer step
        scaler.update() 

        if rank == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % cfg.train.log_interval == 0:
                # Note: total_loss_accum is already scaled by grad_accum_steps
                lossf = total_loss_accum # Log the average loss over the microsteps
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}")
                # Add WandB or TensorBoard logging here if desired

        iter_num += 1
        local_iter_num += 1


        if iter_num >= cfg.train.max_iters:
             print(f"Rank {rank} reached max iterations ({cfg.train.max_iters}).")
             break

    if rank == 0 and cfg.train.save_checkpoints:
        final_checkpoint = {
            'iter_num': iter_num,
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'model_config': raw_model.config,
            'vocab': vocab.stoi,
            'hydra_config': OmegaConf.to_container(cfg, resolve=True)
        }
        final_ckpt_path = Path(cfg.train.out_dir) / "final_model.pt"
        print(f"Saving final model checkpoint to {final_ckpt_path}")
        torch.save(final_checkpoint, final_ckpt_path)

    cleanup() # Clean up distributed processes
    print(f"Rank {rank} finished training worker.")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main_hydra(cfg: DictConfig):
    print("--- Hydra Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("--------------------------")
    
    
    curr_dir = os.getcwd()
    print(f"Resolved output directory: {curr_dir}")
    OmegaConf.set_struct(cfg, False) # Allow modification
    cfg.train.out_dir = curr_dir
    OmegaConf.set_struct(cfg, True) 
    
    
    world_size = cfg.get("world_size", torch.cuda.device_count()) # Get from config or detect
    if world_size > torch.cuda.device_count():
         print(f"Warning: Requested world_size {world_size} > available GPUs {torch.cuda.device_count()}. Setting to {torch.cuda.device_count()}.")
         world_size = torch.cuda.device_count()

    if world_size == 0:
        print("ERROR: No GPUs detected or requested. DDP training requires at least one GPU.")
        return
    elif world_size == 1:
         print("Running on a single GPU. Spawning one process.")
         mp.spawn(train_worker,
                  args=(world_size, cfg), # Pass world_size=1
                  nprocs=world_size,      # nprocs=1
                  join=True)
    else:
         print(f"Spawning {world_size} processes for DDP training.")
         mp.spawn(train_worker,
                  args=(world_size, cfg), # Pass world_size and loaded hydra config
                  nprocs=world_size,
                  join=True)

    if world_size > torch.cuda.device_count():
         print(f"Warning: Requested world_size {world_size} > available GPUs {torch.cuda.device_count()}. Setting to {torch.cuda.device_count()}.")
         world_size = torch.cuda.device_count()
         

             
    print("Main process finished.")


if __name__ == "__main__":
    main_hydra()