import torch
import torch.nn as nn
from torch.nn import functional as F
from contextlib import nullcontext
import math
import transformers.activations
from transformers import GPT2Config

from src.manager import MANAGER

from collections import namedtuple
# Make sure aux_loss and router_z_loss from ModelOutput are the raw, unweighted losses for logging
ModelOutput = namedtuple("ModelOutput", ["loss", "logits", "aux_loss", "router_z_loss"])

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

class MLP(torch.nn.Module):
    def __init__(self, config): 
        super().__init__()
        
        self.is_moe = hasattr(config, 'n_experts') and config.n_experts > 1
        
        if self.is_moe:
            # MoE parameters
            self.n_embd = config.n_embd 
            self.n_exp = config.n_experts
            self.hidden_dim = getattr(config, 'n_inner', 4 * config.n_embd) 
            self.bias = getattr(config, 'bias', False)

            self.c_fc = nn.Parameter(torch.empty(self.n_exp, self.n_embd, self.hidden_dim))
            if self.bias:
                self.c_fc_bias = nn.Parameter(torch.empty(self.n_exp, 1, self.hidden_dim))
            else:
                self.register_parameter('c_fc_bias', None)

            self.activation = transformers.activations.get_activation(config.activation_function)

            self.c_proj = nn.Parameter(torch.empty(self.n_exp, self.hidden_dim, self.n_embd))
            if self.bias:
                self.c_proj_bias = nn.Parameter(torch.empty(self.n_exp, 1, self.n_embd))
            else:
                self.register_parameter('c_proj_bias', None)
                
            self._init_weights()
        else:
            # Standard MLP parameters

            n_inner = 4 * config.n_embd  # Default value as fallback
            if hasattr(config, 'n_inner') and config.n_inner is not None:
                n_inner = config.n_inner

            self.c_fc = nn.Linear(config.n_embd, n_inner, bias=config.bias)
            self.activation = transformers.activations.get_activation(config.activation_function)
            self.c_proj = nn.Linear(n_inner, config.n_embd, bias=config.bias)            
        
        self.dropout = nn.Dropout(config.resid_pdrop)

    def _init_weights(self):
        if hasattr(self, 'c_fc') and isinstance(self.c_fc, nn.Parameter):
            torch.nn.init.normal_(self.c_fc, mean=0.0, std=0.02)
        if hasattr(self, 'c_proj') and isinstance(self.c_proj, nn.Parameter):
            torch.nn.init.normal_(self.c_proj, mean=0.0, std=0.02)
        if hasattr(self, 'c_fc_bias') and self.c_fc_bias is not None:
            torch.nn.init.zeros_(self.c_fc_bias)
        if hasattr(self, 'c_proj_bias') and self.c_proj_bias is not None:
            torch.nn.init.zeros_(self.c_proj_bias)

    def forward(self, x):
        if self.is_moe:
            # For MoE, x is already batched by experts
            # Process each expert's tokens
            x = torch.bmm(x, self.c_fc)
            if self.c_fc_bias is not None:
                x = x + self.c_fc_bias
            x = self.activation(x)
            x = torch.bmm(x, self.c_proj)
            if self.c_proj_bias is not None:
                x = x + self.c_proj_bias
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
            self.experts = MLP(config)   
            self.config = config 
        else:
            self.mlp = MLP(config) 

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) 
        
        x_ln = self.ln_2(x) 
        
        if self.is_moe:
            batch_size, seq_len, hidden_dim = x_ln.shape
            num_tokens = batch_size * seq_len
            x_flat = x_ln.reshape(num_tokens, hidden_dim)

            # Route tokens to experts
            used_capacity, cb_weight, sec_mask = self.router(x_ln)
            
            # Prepare inputs for experts
            expert_inputs = torch.zeros(
                self.config.n_experts, 
                self.router._get_capacity(num_tokens), 
                hidden_dim, 
                device=x.device, 
                dtype=x.dtype
            )
            
            # Extract token indices for each expert
            for expert_idx in range(self.config.n_experts):
                token_indices, slot_indices = sec_mask[:, expert_idx].nonzero(as_tuple=True)
                if token_indices.numel() > 0:
                    expert_inputs[expert_idx, slot_indices] = x_flat[token_indices]
            
            # Process inputs through experts
            expert_outputs = self.experts(expert_inputs)
            
            # Combine expert outputs weighted by router probabilities
            combined_output = torch.zeros(num_tokens, hidden_dim, device=x.device, dtype=x.dtype)
            
            for expert_idx in range(self.config.n_experts):
                token_indices, slot_indices = sec_mask[:, expert_idx].nonzero(as_tuple=True)
                if token_indices.numel() > 0:
                    combined_output[token_indices] += cb_weight[token_indices, expert_idx, slot_indices].unsqueeze(-1) * expert_outputs[expert_idx, slot_indices]
            
            # Reshape output back to original dimensions
            x_mlp_out = combined_output.reshape(batch_size, seq_len, hidden_dim)
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
        assert self.top_k >= 1 and self.top_k <= config.n_experts
        
        self.use_noisy_top_k = getattr(config, 'use_noisy_top_k', False)
        self.train_capacity_factor = getattr(config, 'train_capacity', 1.25) 
        self.eval_capacity_factor = getattr(config, 'eval_capacity', 2.0)   
        self.min_capacity = getattr(config, 'min_capacity', 4)
        self.router_use_full_prec = getattr(config, 'router_use_full_prec', False)

        self.use_aux_loss = getattr(config, 'use_aux_loss', False)
        self.use_router_z_loss = getattr(config, 'use_router_z_loss', False)
        
        # Router weights
        self.w_g = nn.Linear(config.n_embd, config.n_experts, bias=False)
        self.w_noise = nn.Linear(config.n_embd, config.n_experts, bias=False) if self.use_noisy_top_k else None
    
    def forward(self, x):
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        ctx = nullcontext() if not self.router_use_full_prec else torch.amp.autocast(device_type=device_type, enabled=False, dtype=torch.float32)

        with ctx:
            B, T, hidden_dim = x.shape
            num_tokens = B * T
            x_flat = x.reshape(num_tokens, hidden_dim)

            # Get router logits
            logits = self.w_g(x_flat)
            
            # Apply noise during training if enabled
            if self.use_noisy_top_k and self.training:
                noise_logits = self.w_noise(x_flat) 
                noise = torch.randn_like(noise_logits) * F.softplus(noise_logits)
                logits = logits + noise

            # Router z-loss to stabilize router logits
            if self.use_router_z_loss and self.training:
                z_loss = torch.mean(torch.logsumexp(logits, dim=-1).pow(2))
                MANAGER.add_router_z_loss(z_loss)

            # Get top-k experts per token
            top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)
            
            # Create a mask for routing
            router_probs = torch.full_like(logits, float('-inf'))
            router_probs.scatter_(-1, top_k_indices, top_k_logits)
            router_probs = F.softmax(router_probs, dim=-1)

            # Calculate auxiliary load balancing loss if enabled
            if self.use_aux_loss and self.training:
                # Calculate fraction of tokens routed to each expert
                tokens_per_expert_fraction = F.one_hot(top_k_indices, num_classes=self.n_exp).sum(dim=1).float().mean(dim=0)
                # Calculate fraction of router probability assigned to each expert
                prob_per_expert_mean = router_probs.mean(dim=0)
                # Compute auxiliary loss
                aux_loss = self.n_exp * torch.sum(tokens_per_expert_fraction * prob_per_expert_mean)
                MANAGER.add_aux_loss(aux_loss)

            # Calculate capacity
            capacity = self._get_capacity(num_tokens)
            
            # Create expert masks and weights
            # [num_tokens, n_experts, capacity]
            exp_mask = F.one_hot(top_k_indices, num_classes=self.n_exp)
            exp_mask = exp_mask.view(num_tokens, self.top_k, self.n_exp)
            exp_mask = exp_mask.permute(1, 0, 2)  # [top_k, num_tokens, n_experts]

            # Calculate the position of each token in each expert's buffer
            # and ensure we respect capacity constraints
            exp_rank = exp_mask.reshape(self.top_k * num_tokens, self.n_exp)
            exp_rank = torch.cumsum(exp_rank, dim=0) - 1
            exp_rank = exp_rank.reshape(self.top_k, num_tokens, self.n_exp)

            # Mask out tokens that exceed capacity
            exp_mask = exp_mask * torch.lt(exp_rank, capacity)
            used_capacity = torch.sum(exp_mask, dim=(0, 1))  # [n_experts]

            # Calculate final rank for selected experts
            exp_rank = torch.sum(exp_mask * exp_rank, dim=-1)  # [top_k, num_tokens]

            # Use router probabilities to weight experts
            router_probs = router_probs.view(num_tokens, self.n_exp)[None, :]  # [1, num_tokens, n_experts]
            exp_weights = exp_mask * router_probs  # [top_k, num_tokens, n_experts]

            # Create one-hot encoding of ranks
            exp_rank_sc = F.one_hot(exp_rank.long(), num_classes=capacity)  # [top_k, num_tokens, capacity]

            # Combine weights and ranks to create final weight tensor
            cb_weight = torch.sum(exp_weights.unsqueeze(3) * exp_rank_sc.unsqueeze(2), dim=0)  # [num_tokens, n_experts, capacity]
            sec_mask = cb_weight.bool()  # [num_tokens, n_experts, capacity]

            return used_capacity, cb_weight, sec_mask
    
    def _get_capacity(self, tokens_per_batch):
        capacity_factor = self.train_capacity_factor if self.training else self.eval_capacity_factor
        capacity = math.floor(self.top_k * capacity_factor * tokens_per_batch / self.n_exp)
        capacity = max(capacity, self.min_capacity)
        return int(capacity)