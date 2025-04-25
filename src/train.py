from tokenizer.constants import SpecialToken as ST
from tokenizer.datasets import TimelineDataset
import torch
from torch.utils.data import DataLoader, Subset

train_dataset = TimelineDataset(
    input_dir="/workspace/ehr_stuff/EHR_FM/data/tokenized_datasets/mimic_train",
    n_positions=2048,
    is_encoder_decoder=False,
)

vocab = train_dataset.vocab

vocab_size = (len(vocab) // 64 + 1) * 64 if len(vocab) % 64 != 0 else len(vocab)
tokens_of_interest = [ST.DEATH, ST.ADMISSION, ST.DISCHARGE]
tokens_of_interest = {stoken: vocab.encode(stoken) for stoken in tokens_of_interest}

print(tokens_of_interest) # from admissions table

val_size = int(6 * 1_000_000)
train_dataset, val_dataset = (
    Subset(train_dataset, indices=indices)
    for indices in torch.split_with_sizes(
        torch.arange(len(train_dataset)), [len(train_dataset) - val_size, val_size]
    )
)


def make_infinite_loader(loader):
    while True:
        yield from iter(loader)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
train_dataloader = make_infinite_loader(train_dataloader)

val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True)
val_dataloader = make_infinite_loader(val_dataloader)

# test the dataloader
batch = next(iter(train_dataloader))
print(batch[0].shape)
vocab.encode("MEDS_DEATH")
eval_iters = len(val_dataset) // (32 * 2048) + 1
vocab.decode(batch[0][0][200:250])



## modelling
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel, GPT2Config
config = GPT2Config(
    vocab_size=vocab_size,
    n_positions=2048,
    n_embd=64,
    n_layer=1, ## change this stuff later if the model is bad
    n_head=4,
    n_inner=None,
    activation_function="gelu",
    resid_pdrop=0, ## change this stuff later if the model is bad
    embd_pdrop=0, ## change this stuff later if the model is bad
    attn_pdrop=0, ## change this stuff later if the model is bad
    bias=False, # model doesn't perform well without bias
)


## model.py
import math
from collections import namedtuple
from functools import lru_cache

import torch
import torch.nn as nn
import transformers.activations
from torch.nn import functional as F
from transformers import GPT2Config

ModelOutput = namedtuple("ModelOutput", ["loss", "logits"])


class CausalSelfAttention(nn.Module):
    def __init__(self, config, attention_weights: list | None = None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.attn_pdrop
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash or attention_weights is not None:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.n_positions, config.n_positions)).view(
                    1, 1, config.n_positions, config.n_positions
                ),
                persistent=False,
            )
        self.attention_weights = attention_weights

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the
        # batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash and self.attention_weights is None:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            self.attention_weights.append(att.detach().cpu())
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.activation = transformers.activations.get_activation(config.activation_function)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, attention_weights: list | None = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, attention_weights=attention_weights)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2LMNoBiasModel(nn.Module):
    def __init__(
        self,
        config: GPT2Config,
        return_attention=False,
    ):
        super().__init__()
        self.config = config

        self.return_attention = return_attention
        self.attention_weights = [] if return_attention else None

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.n_positions, config.n_embd),
                drop=nn.Dropout(config.embd_pdrop),
                h=nn.ModuleList(
                    [Block(config, self.attention_weights) for _ in range(config.n_layer)]
                ),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        pos = torch.arange(0, config.n_positions, dtype=torch.long)
        self.register_buffer("pos", pos, persistent=False)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @lru_cache
    def num_parameters(self, exclude_embeddings=True):
        n_params = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, input_ids, labels=None) -> ModelOutput:
        _, t = input_ids.size()
        if self.return_attention:
            self.attention_weights.clear()

        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(self.pos[:t])
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if labels is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return ModelOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def get_next_token(self, x: torch.Tensor, return_probs: bool = False, top_k: int | None = None):
        logits = self(x).logits
        logits = logits[:, -1, :]
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        if return_probs:
            return next_token, probs
        return next_token
    
model = GPT2LMNoBiasModel(config)


## training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# scaler
scaler = torch.amp.GradScaler("float16")



## replace with hydra 
class TrainConfig:
    max_iters = 600000 
    lr = 6e-4 
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0 
    decay_lr = True
    warmup_iters = 2000 
    lr_decay_iters = max_iters 
    min_lr = lr / 10 
    eval_interval = 2000
    eval_iters = eval_iters
    log_interval = 10
    out_dir = "outs/"
    save_checkpoints = True
    device = device 
    dtype = 'float16'
    gradient_accumulation_steps = 2 * 8 # gpu times accumulation steps 
    ## add batch size to config


# config
cfg = TrainConfig()

from pathlib import Path
import numpy as np 

out_dir_path = Path(cfg.out_dir)
out_dir_path.mkdir(parents=True, exist_ok=True)


## from paper -- linear warmup, returns min lr, then applying cosine decay to lr
def get_lr(it):
    if it < cfg.warmup_iters:
        return cfg.lr * it / cfg.warmup_iters
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # starts at 1 and goes to 0
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


import inspect
## same as ethos and andrey karpathy's nanoGPT
def configure_optimizers(model: torch.nn.Module, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    sum(p.numel() for p in decay_params)
    sum(p.numel() for p in nodecay_params)
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and "cuda" in device_type
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    return optimizer


def get_batch(split) -> tuple[torch.Tensor | tuple, torch.Tensor]:
    data = train_dataloader if split == "train" else val_dataloader
    x, y = next(data)
    y = y.to(device, non_blocking=True)
    if isinstance(x, list):
        return (x[0].to(device, non_blocking=True), x[1].to(device, non_blocking=True)), y
    return x.to(device, non_blocking=True), y


from .metrics import estimate_loss



## auto cast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.dtype]
ctx = torch.amp.autocast(device_type=cfg.device.type, dtype=ptdtype) # Use device.type for cpu/cuda
print(f"Autocast context manager created for device '{cfg.device.type}' with dtype '{ptdtype}'.")

# Optimizer
optimizer = configure_optimizers(model, cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), cfg.device.type)
print("Optimizer configured.")



## add resume from ckpt logic!!!
iter_num = 0
best_val_loss = 1e9

print("Starting training loop...")
try:
    X, Y = get_batch("train")
except ValueError as e:
    print(f"Error getting initial batch: {e}")
    exit()
    
    
    
import time    
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0

while True:
    lr = get_lr(iter_num) if cfg.decay_lr else cfg.lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        
    if iter_num % cfg.eval_interval == 0 and cfg.eval_iters > 0:
        losses = estimate_loss(model, ctx, get_batch, cfg.eval_iters, tokens_of_interest)
        val_loss = losses.get('loss/val', float('nan')) 
        print(f"step {iter_num}: train loss {losses['loss/train']:.4f}, val loss {val_loss:.4f}")

        if not math.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            if cfg.save_checkpoints:
                checkpoint = {
                    "iter_num": iter_num,
                    "model": model.state_dict(), 
                    "optimizer": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": config, 
                    "vocab": vocab.stoi, 
                }
                ckpt_path = out_dir_path / "best_model.pt"
                ## recent ckpt saving logic
                print(f"Saving checkpoint to {ckpt_path} (Val Loss: {best_val_loss:.4f})")
                torch.save(checkpoint, ckpt_path)
                
            ## add metric calculation --> time over iteration etc. 
            ## add logging to wandb / tensorboard
            
    for micro_step in range(cfg.gradient_accumulation_steps):
        # ddp logic?? look into this. 
        with ctx:
            output = model(X, Y) # Forward pass
            loss = output.loss
            # Scale loss 
            loss = loss / cfg.gradient_accumulation_steps

        if micro_step < cfg.gradient_accumulation_steps - 1:
             X_next, Y_next = get_batch("train") # get next batch
        scaler.scale(loss).backward()
        if micro_step < cfg.gradient_accumulation_steps - 1:
            X, Y = X_next, Y_next
            
    if cfg.grad_clip != 0.0:
        # gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)


    scaler.step(optimizer) 
    scaler.update() 

    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0 
    t0 = t1 
    
    if iter_num % cfg.log_interval == 0:
        lossf = loss.item() * cfg.gradient_accumulation_steps
        # Model flops calculation?? 
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}")

    iter_num += 1
    local_iter_num += 1

    if iter_num > cfg.max_iters:
        print(f"Maximum iterations ({cfg.max_iters}) reached. Exiting training loop.")
        break
    
print("Training finished.")
# save final state
if cfg.save_checkpoints:
    final_checkpoint = {
        "iter_num": iter_num,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": best_val_loss, 
        "config": config,
        "vocab": vocab.stoi,
    }
    final_ckpt_path = out_dir_path / "final_model.pt"
    print(f"Saving final model checkpoint to {final_ckpt_path}")
    torch.save(final_checkpoint, final_ckpt_path)