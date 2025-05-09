import torch
import torch.nn as nn
from torch.nn import functional as F
from contextlib import nullcontext
import math
import transformers.activations
from transformers import GPT2Config

from src.manager import MANAGER

from collections import namedtuple
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
        # Check if this is a MoE config or standard config
        self.is_moe = hasattr(config, 'n_experts') and config.n_experts > 1
        
        if self.is_moe:
            # MoE specific setup
            self.n_embd = config.n_embd
            self.n_exp = config.n_experts
            self.hidden_dim = 4 * self.n_embd  # Typical expansion factor in transformer MLPs
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
            # Standard MLP setup
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.activation = transformers.activations.get_activation(config.activation_function)
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
            
        # Common dropout setup
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
            # MoE specific forward pass
            if len(x.shape) == 3:  # [batch, seq_len, hidden]
                # This would need additional logic to properly route tokens
                # For example, a batched version of expert routing
                raise NotImplementedError("Batched MoE forward is not implemented in this version")
            else:  # Assume input is properly formatted for experts: [n_experts, capacity, hidden]
                expert_tokens = x
                x = torch.bmm(expert_tokens, self.c_fc)
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
        x = self.ln_2(x)

        if self.is_moe:
            batch_size, seq_len, hidden_dim = x.shape
            num_tokens = batch_size * seq_len
            x_flat = x.view(num_tokens, hidden_dim)

            used_capacity, cb_weight, sec_mask = self.router(x_flat)
            expert_capacity = sec_mask.shape[-1]
            expert_out = torch.zeros_like(x_flat)
            dispatched_input = torch.zeros(self.config.n_experts, expert_capacity, hidden_dim, device=x.device, dtype=x.dtype)
            for token_idx in range(num_tokens):
                for expert_idx in range(self.config.n_experts):
                    for capacity_slot_idx in range(expert_capacity):
                        if sec_mask[token_idx, expert_idx, capacity_slot_idx]:
                            dispatched_input[expert_idx, capacity_slot_idx] = x_flat[token_idx]
            
            processed_by_experts = self.experts(dispatched_input)
            cb_weight_expanded = cb_weight.unsqueeze(-1)
            processed_by_experts_expanded = processed_by_experts.unsqueeze(0)
            weighted_expert_outputs = processed_by_experts_expanded * cb_weight_expanded
            final_output_flat = weighted_expert_outputs.sum(dim=(1,2))
            x_mlp_out = final_output_flat.view(batch_size, seq_len, hidden_dim)
        
        else:
            x_mlp_out = self.mlp(x)
        
        x = x + x_mlp_out
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

            ## moe loss
            is_moe_model = hasattr(self.config, 'n_experts') and getattr(self.config, 'n_experts', 0) > 1
            if is_moe_model:
                if getattr(self.config, 'use_aux_loss', False):
                    loss += getattr(self.config, 'aux_loss_weight', 0.01) * MANAGER.aggregate_aux_loss()
                    MANAGER.reset_aux_loss()
                if getattr(self.config, 'use_router_z_loss', False):
                    loss += getattr(self.config, 'router_z_loss_weight', 0.001) * MANAGER.aggregate_router_z_loss()
                    MANAGER.reset_router_z_loss()
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

class Router(nn.Module):
    def __init__(self, config):
        super().__init__()

        # router settings
        self.top_k = config.top_k
        self.n_exp = config.n_experts
        assert self.top_k >= 1 and self.top_k <= config.n_experts
        self.use_noisy_top_k = config.use_noisy_top_k
        self.train_capacity = config.train_capacity
        self.eval_capacity = config.eval_capacity
        self.min_capacity = config.min_capacity
        self.router_use_full_prec = config.router_use_full_prec

        # auxiliary / load balancing loss settings
        self.use_aux_loss = config.use_aux_loss
        self.use_router_z_loss = config.use_router_z_loss

        # linear projection for (noisy) softmax gating
        # no bias is used, see page 4 eq (4) in (https://arxiv.org/abs/1701.06538)
        self.w_g = nn.Linear(config.n_embd, config.n_experts, bias=False)
        self.w_noise = nn.Linear(config.n_embd, config.n_experts, bias=False) if self.use_noisy_top_k else None
    
    def forward(self, x):
        # optionally run the router in full precision to avoid instability during training
        # see discussion on pg. 9 here: https://arxiv.org/abs/2101.03961
        # setting enabled to False in autocast automatically puts everything in float32
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu' # for later use in torch.autocast
        ctx = nullcontext() if not self.router_use_full_prec else torch.amp.autocast(device_type=device_type, enabled=False)

        with ctx:
            B, T = x.size() if x.dim() == 2 else (x.size(0), 1)
            num_tokens = B * T

            # eq (4) in (https://arxiv.org/abs/1701.06538)
            logits = self.w_g(x)  # [B, T, n_exp] or [B, n_exp]
            if self.use_noisy_top_k:
                # optionally add noise into the router
                noise = F.softplus(self.w_noise(x))
                noise *= torch.randn_like(noise)
                logits += noise

            # router z loss, computed on logits (before softmax)
            # this loss prevents router logits from becoming too large
            if self.use_router_z_loss:
                z_loss = self.compute_router_z_loss(logits)
                MANAGER.add_router_z_loss(z_loss)

            # find top k experts for each token
            top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1) # [B, T, k] or [B, k]

            # normalize expert probabilities
            router_probs = torch.full_like(logits, float('-inf'))  # [B, T, n_exp] or [B, n_exp]
            router_probs.scatter_(-1, top_k_indices, top_k_logits)
            router_probs = F.softmax(router_probs, dim=-1)

            # compute auxiliary load balancing loss
            if self.use_aux_loss:
                aux_loss = self.compute_aux_loss(router_probs, top_k_indices)
                MANAGER.add_aux_loss(aux_loss)

            # compute expert capacity
            exp_capacity = self.get_capacity(num_tokens)

            # Reshape if needed to handle both 2D and 3D inputs
            if router_probs.dim() == 2:  # [B, n_exp]
                router_probs = router_probs.unsqueeze(1)  # [B, 1, n_exp]
                top_k_indices = top_k_indices.unsqueeze(1)  # [B, 1, k]

            # make a multi-hot mask of chosen experts
            exp_mask = F.one_hot(top_k_indices, num_classes=self.n_exp)  # [B, T, k, n_exp]
            exp_mask = exp_mask.view(num_tokens, self.top_k, self.n_exp)  # [B * T, k, n_exp]
            exp_mask = exp_mask.permute(1, 0, 2) # [k, B * T, n_exp]

            # compute cumulative sum
            exp_rank = exp_mask.reshape(self.top_k * num_tokens, self.n_exp)  # [k * B * T, n_exp]
            exp_rank = torch.cumsum(exp_rank, dim=0) - 1  # [k * B * T, n_exp]
            exp_rank = exp_rank.reshape(self.top_k, num_tokens, self.n_exp)  # [k, B * T, n_exp]

            # mask out entries beyond capacity
            exp_mask *= torch.lt(exp_rank, exp_capacity) # [k, B * T, n_exp]
            used_capacity = torch.sum(exp_mask, dim=(0, 1)) # [n_exp]

            # mask rank and get position in expert batch
            exp_rank = torch.sum(exp_mask * exp_rank, dim=-1)  # [k, B * T]

            # mask probabilities to only include selected experts
            router_probs = router_probs.view(num_tokens, self.n_exp)[None, :] # [1, B * T, n_exp]
            exp_weights = exp_mask * router_probs # [k, B * T, n_exp]

            # convert rank to one-hot
            exp_rank_sc = F.one_hot(exp_rank, num_classes=exp_capacity) # [k, B * T, exp_capacity]

            # create final weight vector for each token's experts
            cb_weight = torch.sum(exp_weights.unsqueeze(3) * exp_rank_sc.unsqueeze(2), dim=0)
            sec_mask = cb_weight.bool() # binary mask of selected experts 
            
            return used_capacity, cb_weight, sec_mask

    def compute_aux_loss(self, expert_probs, indices):
        """Compute auxiliary load balancing loss"""
        with torch.no_grad():
            one_hot_indices = F.one_hot(indices, num_classes=self.n_exp)
            if one_hot_indices.dim() == 4:  # [B, T, k, n_exp]
                one_hot_indices = torch.sum(one_hot_indices.float(), dim=2)  # [B, T, n_exp]
            elif one_hot_indices.dim() == 3:  # [B, k, n_exp]
                one_hot_indices = torch.sum(one_hot_indices.float(), dim=1)  # [B, n_exp]
            
            tokens_per_expert = torch.mean(one_hot_indices.float(), dim=tuple(range(one_hot_indices.dim() - 1)))

        prob_per_expert = torch.mean(expert_probs.float(), dim=tuple(range(expert_probs.dim() - 1)))
        return self.n_exp * torch.sum(prob_per_expert * tokens_per_expert)
    
    def compute_router_z_loss(self, logits):
        """Compute router z-loss to prevent logits from growing too large"""
        z_loss = torch.logsumexp(logits, dim=-1) ** 2.0
        return torch.mean(z_loss)

    def get_capacity(self, tokens_per_batch):
        """Calculate expert capacity based on batch size"""
        capacity_factor = self.train_capacity if self.training else self.eval_capacity
        capacity = math.floor(self.top_k * capacity_factor * tokens_per_batch / self.n_exp)
        capacity += capacity % 2  # make sure capacity is an even number
        capacity = max(capacity, self.min_capacity)  # use min capacity
        assert capacity > 0
        return int(capacity)