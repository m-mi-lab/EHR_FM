import torch
import torch.nn as nn
from torch.nn import functional as F
from contextlib import nullcontext
import math
import transformers.activations
from transformers import GPT2Config
import torch.distributed as dist

from src.manager import MANAGER
from src.distributed import (
    AllGather,
    split_by_rank,
    gather_sizes,
    pad_dim_to,
    has_only_one_value,
    chunk_num
)

from collections import namedtuple
# Make sure aux_loss and router_z_loss from ModelOutput are the raw, unweighted losses for logging
ModelOutput = namedtuple("ModelOutput", ["loss", "logits", "aux_loss", "router_z_loss"])

# Helper functions for MoE
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    one_hot_classes = max(max_index + 1, max_length)
    return F.one_hot(indexes, one_hot_classes)[..., :max_length]

def cumsum_exclusive(t, dim=-2):
    """Exclusive cumsum along specified dimension"""
    assert dim < 0
    num_pad_dims = -dim - 1
    pre_padding = (0, 0) * num_pad_dims
    return F.pad(t, (*pre_padding, 1, -1)).cumsum(dim=dim)

def cast_tuple(el, length=1):
    return el if isinstance(el, tuple) else ((el,) * length)

def pack_one(t, pattern):
    """Pack a single tensor (helper for einops)."""
    from einops import pack
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    """Unpack a single tensor (helper for einops)."""
    from einops import unpack
    return unpack(t, ps, pattern)[0]

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

        # Don't store the list reference, but make a boolean flag
        self.store_attention = attention_weights is not None
        self.attention_weights = attention_weights

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash and not self.store_attention:
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
            # Only store if needed and list exists
            if self.store_attention and self.attention_weights is not None:
                # Use clone() to avoid reference issues
                self.attention_weights.append(att.detach().cpu())
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

# RMSNorm for experts
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.gamma * self.scale

# GEGLU activation for experts
class GEGLU(nn.Module):
    def __init__(self, dim, mult_bias=True):
        super().__init__()
        self.mult_bias = nn.Parameter(torch.ones(dim)) if mult_bias else 1.

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x * self.mult_bias

class MLP(torch.nn.Module):
    def __init__(self, config, is_distributed=None, allow_var_seq_len=False): 
        super().__init__()
        
        self.is_moe = hasattr(config, 'n_experts') and config.n_experts > 1
        
        if self.is_moe:
            # MoE parameters - using GEGLU architecture
            self.n_embd = config.n_embd 
            self.n_exp = config.n_experts
            hidden_mult = getattr(config, 'expert_hidden_mult', 4)
            self.dim_hidden = int(config.n_embd * hidden_mult * 2 / 3)
            self.bias = getattr(config, 'bias', False)
            self.mult_bias = getattr(config, 'expert_mult_bias', True)
            self.prenorm = getattr(config, 'expert_prenorm', False)

            # Expert networks: RMSNorm -> Linear -> GEGLU -> Linear
            if self.prenorm:
                self.norm = RMSNorm(config.n_embd)
            else:
                self.norm = None
            
            self.c_fc = nn.Parameter(torch.empty(self.n_exp, config.n_embd, self.dim_hidden * 2))
            self.c_proj = nn.Parameter(torch.empty(self.n_exp, self.dim_hidden, config.n_embd))
            
            if self.bias:
                self.c_fc_bias = nn.Parameter(torch.empty(self.n_exp, 1, self.dim_hidden * 2))
                self.c_proj_bias = nn.Parameter(torch.empty(self.n_exp, 1, config.n_embd))
            else:
                self.register_parameter('c_fc_bias', None)
                self.register_parameter('c_proj_bias', None)
            
            self.geglu = GEGLU(self.dim_hidden, mult_bias=self.mult_bias)
            self._init_weights()
            
            # Distributed settings
            self.is_distributed = is_distributed
            if self.is_distributed is None:
                self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1
            
            self.all_gather = AllGather()
            self.allow_var_seq_len = allow_var_seq_len
            
            # Device tracker for distributed training
            self.register_buffer('dummy', torch.ones(1), persistent=False)
        else:
            # Standard MLP parameters
            n_inner = 4 * config.n_embd  # Default value as fallback
            if hasattr(config, 'n_inner') and config.n_inner is not None:
                n_inner = config.n_inner

            self.c_fc = nn.Linear(config.n_embd, n_inner, bias=config.bias)
            self.activation = transformers.activations.get_activation(config.activation_function)
            self.c_proj = nn.Linear(n_inner, config.n_embd, bias=config.bias)            
        
        self.dropout = nn.Dropout(config.resid_pdrop)

    @property
    def device(self):
        if self.is_moe:
            return self.dummy.device
        return next(self.parameters()).device

    def _init_weights(self):
        if hasattr(self, 'c_fc') and isinstance(self.c_fc, nn.Parameter):
            dim = self.c_fc.shape[0] if len(self.c_fc.shape) > 2 else self.n_embd
            std = dim ** -0.5
            torch.nn.init.uniform_(self.c_fc, -std, std)
        if hasattr(self, 'c_proj') and isinstance(self.c_proj, nn.Parameter):
            dim = self.c_proj.shape[0] if len(self.c_proj.shape) > 2 else self.dim_hidden
            std = dim ** -0.5
            torch.nn.init.uniform_(self.c_proj, -std, std)
        if hasattr(self, 'c_fc_bias') and self.c_fc_bias is not None:
            dim = self.c_fc_bias.shape[1]
            std = dim ** -0.5
            torch.nn.init.uniform_(self.c_fc_bias, -std, std)
        if hasattr(self, 'c_proj_bias') and self.c_proj_bias is not None:
            dim = self.c_proj_bias.shape[1]
            std = dim ** -0.5
            torch.nn.init.uniform_(self.c_proj_bias, -std, std)

    def forward(self, x, is_distributed=None):
        if self.is_moe:
            """
            Forward pass for MoE with distributed support
            x: [B, n_experts, capacity, n_embd] for non-distributed
               [r*B, n_experts, capacity, n_embd] for distributed (after all-gather)
            """
            is_distributed = is_distributed if is_distributed is not None else self.is_distributed
            shape = x.shape
            seq_len = shape[-2]  # capacity dimension
            
            # For distributed: gather across batch dimension
            world_size = 1
            rank = 0
            
            if is_distributed:
                seq_sizes = gather_sizes(x, dim=-2)
                var_seq_len = not has_only_one_value(seq_sizes)
                
                assert self.allow_var_seq_len or not var_seq_len, \
                    'number of tokens per expert must be the same - set `allow_var_seq_len=True` to handle variable lengths'
                
                # Pad if variable sequence length
                if var_seq_len:
                    max_seq_size = seq_sizes.amax().item()
                    x = pad_dim_to(x, max_seq_size, dim=-2)
                
                # All-gather across batches
                x, batch_sizes = self.all_gather(x)
                total_batch_size = batch_sizes.sum().item()
                
                world_size = dist.get_world_size()
                rank = dist.get_rank()
            
            # Determine which experts this rank handles
            num_experts_per_rank = self.n_exp
            expert_slice = slice(0, self.n_exp)
            
            if is_distributed:
                if world_size <= self.n_exp:
                    # More experts than GPUs: each GPU handles subset of experts
                    num_experts_across_ranks = chunk_num(self.n_exp, world_size)
                    start_indices = cumsum_exclusive(torch.tensor(num_experts_across_ranks), dim=-1)
                    
                    num_experts_per_rank = num_experts_across_ranks[rank]
                    num_experts_batches_across_ranks = tuple(i * total_batch_size for i in num_experts_across_ranks)
                    
                    expert_start_index = start_indices[rank].item()
                else:
                    # More GPUs than experts: each expert handled by multiple GPUs (batch-split)
                    num_batch_chunks = world_size // self.n_exp
                    total_ranks_in_use = num_batch_chunks * self.n_exp
                    
                    expert_start_index = rank // num_batch_chunks
                    
                    batch_splits = chunk_num(total_batch_size, num_batch_chunks)
                    num_experts_batches_across_ranks = batch_splits * self.n_exp
                    
                    # Remaining machines process nothing
                    remain_ranks = world_size % self.n_exp
                    num_experts_batches_across_ranks += (0,) * remain_ranks
                    
                    num_experts_per_rank = int(rank < total_ranks_in_use)
                
                assert len(num_experts_batches_across_ranks) == world_size
                expert_slice = slice(expert_start_index, expert_start_index + num_experts_per_rank)
            
            # Rearrange to [n_experts, B, capacity, n_embd]
            from einops import rearrange
            x = rearrange(x, 'b e n d -> e b n d')
            
            if is_distributed:
                # Pack and split by rank
                x, expert_batch_packed_shape = pack_one(x, '* n d')
                x = x.split(num_experts_batches_across_ranks, dim=0)
                x, experts_per_rank_sizes = split_by_rank(x)
                
                if num_experts_per_rank > 0:
                    x = rearrange(x, '(e b) n d -> e b n d', e=num_experts_per_rank)
                else:
                    x = x.reshape(self.n_exp, *x.shape)
            
            # Get expert parameters for this rank
            c_fc_slice = self.c_fc[expert_slice]
            c_proj_slice = self.c_proj[expert_slice]
            c_fc_bias_slice = self.c_fc_bias[expert_slice] if self.c_fc_bias is not None else None
            c_proj_bias_slice = self.c_proj_bias[expert_slice] if self.c_proj_bias is not None else None
            
            # Apply expert networks
            if self.norm is not None:
                x = self.norm(x)
            
            # First linear: [n_experts_local, B, capacity, n_embd] -> [n_experts_local, B, capacity, dim_hidden*2]
            # x: [n_experts_local, B, capacity, n_embd], c_fc_slice: [n_experts_local, n_embd, dim_hidden*2]
            n_experts_local, B, capacity, n_embd = x.shape
            x_reshaped = x.reshape(n_experts_local * B, capacity, n_embd)
            c_fc_expanded = c_fc_slice.unsqueeze(1).expand(n_experts_local, B, n_embd, -1).reshape(n_experts_local * B, n_embd, -1)
            x = torch.bmm(x_reshaped, c_fc_expanded)
            x = x.reshape(n_experts_local, B, capacity, -1)
            
            if c_fc_bias_slice is not None:
                x = x + c_fc_bias_slice.unsqueeze(1)
            
            # GEGLU activation
            x = self.geglu(x)
            
            # Second linear: [n_experts_local, B, capacity, dim_hidden] -> [n_experts_local, B, capacity, n_embd]
            # x: [n_experts_local, B, capacity, dim_hidden], c_proj_slice: [n_experts_local, dim_hidden, n_embd]
            n_experts_local, B, capacity, dim_hidden = x.shape
            x_reshaped = x.reshape(n_experts_local * B, capacity, dim_hidden)
            c_proj_expanded = c_proj_slice.unsqueeze(1).expand(n_experts_local, B, dim_hidden, -1).reshape(n_experts_local * B, dim_hidden, -1)
            x = torch.bmm(x_reshaped, c_proj_expanded)
            x = x.reshape(n_experts_local, B, capacity, -1)
            
            if c_proj_bias_slice is not None:
                x = x + c_proj_bias_slice.unsqueeze(1)
            
            # All-gather results if distributed
            if is_distributed:
                x = rearrange(x, 'e b n d -> (e b) n d')
                x, _ = self.all_gather(x, sizes=experts_per_rank_sizes)
                x = unpack_one(x, expert_batch_packed_shape, '* n d')
            
            # Rearrange back to [B, n_experts, capacity, n_embd]
            x = rearrange(x, 'e b n d -> b e n d')
            
            if is_distributed:
                # Split back to per-rank batches
                x = x.split(batch_sizes.tolist())
                x, _ = split_by_rank(x)
                
                # Remove padding
                x = x[..., :seq_len, :]
            
            assert x.shape == shape
            return self.dropout(x)
        else:
            # Standard MLP forward pass
            x = self.c_fc(x)
            x = self.activation(x)
            x = self.c_proj(x)
            return self.dropout(x)


class Block(torch.nn.Module):
    def __init__(self, config, attention_weights: list | None = None): 
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, attention_weights=attention_weights)
        self.ln_2 = torch.nn.LayerNorm(config.n_embd, bias=config.bias)

        self.is_moe = hasattr(config, 'n_experts') and config.n_experts > 1
        if self.is_moe:
            self.router = Router(config) 
            # Enable distributed training and variable sequence length handling
            is_distributed = getattr(config, 'expert_distributed', None)
            allow_var_seq_len = getattr(config, 'allow_var_seq_len', False)
            self.experts = MLP(config, is_distributed=is_distributed, allow_var_seq_len=allow_var_seq_len)   
            self.config = config 
        else:
            self.mlp = MLP(config) 

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) 
        
        x_ln = self.ln_2(x) 
        
        if self.is_moe:
            # Route tokens to experts using threshold-based routing
            dispatch_tensor, combine_tensor = self.router(x_ln)
            # dispatch_tensor: [B, T, n_experts, capacity], combine_tensor: [B, T, n_experts, capacity]
            
            # Dispatch: einsum('b n d, b n e c -> b e c d', x, dispatch_tensor)
            # This efficiently routes tokens to experts
            expert_inputs = torch.einsum('b n d, b n e c -> b e c d', x_ln, dispatch_tensor)
            # expert_inputs: [B, n_experts, capacity, n_embd]
            
            # Process through experts with distributed support
            expert_outputs = self.experts(expert_inputs)
            # expert_outputs: [B, n_experts, capacity, n_embd]
            
            # Combine: einsum('b e c d, b n e c -> b n d', expert_outputs, combine_tensor)
            x_mlp_out = torch.einsum('b e c d, b n e c -> b n d', expert_outputs, combine_tensor)
        else:
            x_mlp_out = self.mlp(x_ln)
        
        x = x + x_mlp_out
        return x


class GPT2LMNoBiasModel(torch.nn.Module):
    def __init__(self, base_gpt_config: GPT2Config, moe_hydra_config, return_attention=False):
        super().__init__()
        self.base_config = base_gpt_config # Store base GPT2 config (vocab_size, n_embd, n_layer, n_head etc from GPT2Config)
        self.moe_config = moe_hydra_config # Store the moe config object (e.g., Hydra cfg.model, has n_experts, top_k etc.)
        self.return_attention = return_attention
        self.attention_weights = [] if return_attention else None

        # Determine n_embd from base_config as it's fundamental
        n_embd = self.base_config.n_embd
        
        # If MoE is active, Block needs parameters like n_embd, n_head etc.
        # These can come from moe_config if they are defined there and override base_config,
        # or moe_config can just be a DictConfig/OmegaConf object holding n_experts etc.
        # For Block initialization, we need a config object that it understands.
        # Let's create a combined config for Block if MoE is used.
        
        config_for_block = self.base_config
        if hasattr(self.moe_config, 'n_experts') and self.moe_config.n_experts > 1:
            # If moe_config is an OmegaConf object, convert to dict to easily merge/update
            # Or, directly pass moe_config to Block and let Block handle attribute access.
            # The current Block expects an object with attributes like n_embd, n_head, bias, resid_pdrop etc.
            # So, moe_config should provide these or they should be taken from base_config.
            # The simplest is to ensure moe_config (cfg.model) has all necessary fields for Block.
            config_for_block = self.moe_config


        self.transformer = torch.nn.ModuleDict(dict(
            wte=torch.nn.Embedding(self.base_config.vocab_size, n_embd),
            wpe=torch.nn.Embedding(self.base_config.n_positions, n_embd),
            drop=torch.nn.Dropout(self.base_config.embd_pdrop),
            h=torch.nn.ModuleList([Block(config_for_block, self.attention_weights) for _ in range(self.base_config.n_layer)]),
            ln_f=torch.nn.LayerNorm(n_embd, bias=self.base_config.bias),
        ))
        self.lm_head = torch.nn.Linear(n_embd, self.base_config.vocab_size, bias=self.base_config.bias)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.base_config.n_layer))

        pos = torch.arange(0, self.base_config.n_positions, dtype=torch.long)
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
        if t > self.base_config.n_positions:
            raise ValueError(f"Cannot forward sequence of length {t}, block size is only {self.base_config.n_positions}")

        if self.return_attention and self.attention_weights is not None:
            self.attention_weights.clear()

        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(self.pos[:t])
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # Initialize raw aux losses to None for returning
        raw_aux_loss = None
        raw_router_z_loss = None

        if labels is not None: # Training or evaluation with labels
            logits = self.lm_head(x)
            # Ensure labels are long type and handle ignore_index if necessary
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1).long(), ignore_index=-100)

            # Check if MoE is configured using self.moe_config
            is_moe_model = hasattr(self.moe_config, 'n_experts') and getattr(self.moe_config, 'n_experts', 0) > 1
            
            if is_moe_model and self.training: # Add aux losses only during training
                if getattr(self.moe_config, 'use_aux_loss', False):
                    raw_aux_loss = MANAGER.aggregate_aux_loss() # Get the sum of aux losses from all MoE layers
                    if raw_aux_loss is not None:
                        loss += getattr(self.moe_config, 'aux_loss_weight', 0.01) * raw_aux_loss
                    MANAGER.reset_aux_loss() # Reset for the next forward pass

                if getattr(self.moe_config, 'use_router_z_loss', False):
                    raw_router_z_loss = MANAGER.aggregate_router_z_loss() # Get router_z_loss
                    if raw_router_z_loss is not None:
                        loss += getattr(self.moe_config, 'router_z_loss_weight', 0.001) * raw_router_z_loss
                    MANAGER.reset_router_z_loss() # Reset for the next forward pass
        else: # Inference
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
            # Aux losses are typically not added or returned during inference

        return ModelOutput(loss=loss, logits=logits, aux_loss=raw_aux_loss, router_z_loss=raw_router_z_loss)

    @torch.no_grad()
    def get_next_token(self, x: torch.Tensor, return_probs: bool = False, top_k: int | None = None):
        output = self(x) # Call forward, labels will be None
        logits = output.logits # Get logits from ModelOutput
        # logits = logits[:, -1, :] # Already done in inference path of forward
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf") # Apply top-k filtering
        probs = torch.nn.functional.softmax(logits, dim=-1) # Get probabilities
        next_token = torch.multinomial(probs, num_samples=1) # Sample next token
        if return_probs:
            return next_token, probs
        return next_token


class Router(nn.Module):
    def __init__(self, config): 
        super().__init__()

        self.top_k = config.top_k 
        self.n_exp = config.n_experts
        assert self.top_k >= 2, 'must be 2 or more experts'
        assert self.top_k <= config.n_experts
        
        self.eps = 1e-9
        self.use_noisy_top_k = getattr(config, 'use_noisy_top_k', False)
        self.train_capacity_factor = getattr(config, 'train_capacity', 1.25) 
        self.eval_capacity_factor = getattr(config, 'eval_capacity', 2.0)   
        self.min_capacity = getattr(config, 'min_capacity', 4)
        self.router_use_full_prec = getattr(config, 'router_use_full_prec', False)
        self.straight_through_dispatch_tensor = getattr(config, 'straight_through_dispatch_tensor', True)

        self.use_aux_loss = getattr(config, 'use_aux_loss', False)
        self.use_router_z_loss = getattr(config, 'use_router_z_loss', False)
        
        # Threshold-based routing (default 0.2 for top-2, can be tuple for top-k)
        threshold_train = getattr(config, 'threshold_train', 0.2)
        threshold_eval = getattr(config, 'threshold_eval', 0.2)
        top_n_minus_1 = self.top_k - 1
        
        threshold_train = cast_tuple(threshold_train, top_n_minus_1)
        threshold_eval = cast_tuple(threshold_eval, top_n_minus_1)
        
        self.register_buffer('threshold_train', torch.tensor([self.eps, *threshold_train]))
        self.register_buffer('threshold_eval', torch.tensor([self.eps, *threshold_eval]))
        self.register_buffer('zero', torch.zeros((1,)), persistent=False)
        
        # Router weights
        self.w_g = nn.Linear(config.n_embd, config.n_experts, bias=False)
    
    def forward(self, x):
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        ctx = nullcontext() if not self.router_use_full_prec else torch.amp.autocast(device_type=device_type, enabled=False, dtype=torch.float32)

        with ctx:
            B, T, hidden_dim = x.shape
            num_tokens = B * T
            group_size = T  # sequence length
            
            # Get threshold and capacity factor based on training/eval
            suffix = 'train' if self.training else 'eval'
            threshold = getattr(self, f'threshold_{suffix}')
            capacity_factor = self.train_capacity_factor if self.training else self.eval_capacity_factor
            
            # Calculate expert capacity
            expert_capacity = min(group_size, int((group_size * capacity_factor) / self.n_exp))
            expert_capacity = max(expert_capacity, self.min_capacity)
            expert_capacity_f = float(expert_capacity)
            
            # Get router logits
            gate_logits = self.w_g(x)  # [B, T, n_experts]
            
            # Apply Gumbel noise during training if enabled
            maybe_noised_gate_logits = gate_logits
            if self.use_noisy_top_k and self.training:
                noise = gumbel_noise(gate_logits)
                maybe_noised_gate_logits = gate_logits + noise
            
            # Router z-loss to stabilize router logits
            if self.use_router_z_loss and self.training:
                router_z_loss = torch.logsumexp(gate_logits, dim=-1)
                router_z_loss = torch.square(router_z_loss)
                router_z_loss = router_z_loss.mean()
                MANAGER.add_router_z_loss(router_z_loss)
            
            # Get raw gates (probabilities)
            raw_gates = maybe_noised_gate_logits.softmax(dim=-1)  # [B, T, n_experts]
            
            # Find top-k experts per position
            top_k_values, top_k_indices = raw_gates.topk(self.top_k, dim=-1)  # [B, T, top_k]
            
            # Move top-k dimension to first: [top_k, B, T]
            gates = top_k_values.permute(2, 0, 1)  # [top_k, B, T]
            gate_indices = top_k_indices.permute(2, 0, 1)  # [top_k, B, T]
            
            # Create masks
            one_hot_gate_indices = F.one_hot(gate_indices, self.n_exp)  # [top_k, B, T, n_experts]
            mask = one_hot_gate_indices.float()
            mask_1 = mask[0]  # needed for balancing loss
            
            # Normalize top-k gate scores
            denom = gates.sum(dim=0, keepdim=True).clamp(min=self.eps)  # [1, B, T]
            gates = gates / denom  # [top_k, B, T]
            
            # Threshold-based probabilistic routing
            # Route to second expert and beyond with probability min(1., score / threshold)
            probs = torch.zeros_like(gates).uniform_(0., 1.)  # [top_k, B, T]
            threshold_expanded = threshold.view(-1, 1, 1)  # [top_k, 1, 1]
            should_route = probs < (gates / threshold_expanded.clamp(min=self.eps))  # [top_k, B, T]
            
            # Always route to first expert
            should_route[0, ...] = True
            
            # Apply routing mask
            mask = mask * should_route.unsqueeze(-1)  # [top_k, B, T, n_experts]
            
            # Compute positions within expert buffers using exclusive cumsum
            mask_cumsum = cumsum_exclusive(mask, dim=-2)  # [top_k, B, T, n_experts]
            
            # Compute assignment to experts
            positions = []
            prev_expert_count = torch.zeros(B, 1, self.n_exp, device=x.device, dtype=x.dtype)
            
            for n in range(self.top_k):
                position_in_expert = (mask_cumsum[n] + prev_expert_count) * mask[n]
                
                # Remove elements that don't fit capacity
                mask[n] = mask[n] * (position_in_expert < expert_capacity_f).float()
                
                # Count examples going to each expert for next iteration
                prev_expert_count = mask[n].sum(dim=1, keepdim=True) + prev_expert_count
                
                # Get position indices [B, T]
                position_in_expert = position_in_expert.sum(dim=-1)  # [B, T]
                positions.append(position_in_expert)
            
            positions = torch.stack(positions)  # [top_k, B, T]
            
            # Flatten mask: [top_k, B, T] - mostly ones, zeros where didn't fit
            mask_flat = mask.sum(dim=-1)  # [top_k, B, T]
            
            # Weighted assignment
            gates = gates * mask_flat  # [top_k, B, T]
            
            # Create combine tensor: [B, T, n_experts, expert_capacity]
            # Using einsum-like operations: k b n, k b n, k b n e, k b n c -> b n e c
            pos_one_hot_all = safe_one_hot(positions.long(), expert_capacity)  # [top_k, B, T, expert_capacity]
            
            # Expand dimensions for broadcasting
            gates_expanded = gates.unsqueeze(-1).unsqueeze(-1)  # [top_k, B, T, 1, 1]
            mask_flat_expanded = mask_flat.unsqueeze(-1).unsqueeze(-1)  # [top_k, B, T, 1, 1]
            one_hot_gate_expanded = one_hot_gate_indices.unsqueeze(-1)  # [top_k, B, T, n_experts, 1]
            pos_one_hot_expanded = pos_one_hot_all.unsqueeze(-2)  # [top_k, B, T, 1, expert_capacity]
            
            # Multiply all components: gates * mask_flat * one_hot_gate * pos_one_hot
            combine_tensor_k = gates_expanded * mask_flat_expanded * one_hot_gate_expanded * pos_one_hot_expanded
            # combine_tensor_k: [top_k, B, T, n_experts, expert_capacity]
            
            # Sum over top_k dimension
            combine_tensor = combine_tensor_k.sum(dim=0)  # [B, T, n_experts, expert_capacity]
            
            # Dispatch tensor (boolean version of combine_tensor)
            dispatch_tensor = combine_tensor.bool().type(x.dtype)
            
            # Straight-through estimator for gradients
            if self.straight_through_dispatch_tensor:
                dispatch_tensor = dispatch_tensor + combine_tensor - combine_tensor.detach()
            
            # Balance loss
            if self.use_aux_loss and self.training:
                density_1 = mask_1.mean(dim=(0, 1))  # [n_experts]
                density_1_proxy = raw_gates.mean(dim=(0, 1))  # [n_experts]
                balance_loss = (density_1_proxy * density_1).mean() * float(self.n_exp ** 2)
                MANAGER.add_aux_loss(balance_loss)
            
            return dispatch_tensor, combine_tensor
    
    def _get_capacity(self, tokens_per_batch):
        capacity_factor = self.train_capacity_factor if self.training else self.eval_capacity_factor
        capacity = math.floor(self.top_k * capacity_factor * tokens_per_batch / self.n_exp)
        capacity = max(capacity, self.min_capacity)
        return int(capacity)