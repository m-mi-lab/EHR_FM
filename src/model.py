import torch
import torch.nn as nn
from torch.nn import functional as F
from contextlib import nullcontext
import math
import transformers.activations
from transformers import GPT2Config

from src.manager import MANAGER

from collections import namedtuple
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
        # Regular attention path
        x = x + self.attn(self.ln_1(x))
        x_ln = self.ln_2(x)
        
        if self.is_moe:
            batch_size, seq_len, hidden_dim = x_ln.shape
            num_tokens = batch_size * seq_len
            x_flat = x_ln.view(num_tokens, hidden_dim)

            # Route tokens through experts
            used_capacity, cb_weight, sec_mask = self.router(x_flat)
            
            # Process tokens efficiently in a single pass through experts
            expert_capacity = sec_mask.shape[-1]
            dispatched_input = torch.zeros(
                self.config.n_experts, expert_capacity, hidden_dim, 
                device=x.device, dtype=x.dtype
            )
            
            # More efficient token dispatch - avoid loops when possible
            if dispatched_input.shape[0] * dispatched_input.shape[1] <= 1024:  # Small enough for tensor ops
                # Use sec_mask to create mask for each expert
                for expert_idx in range(self.config.n_experts):
                    # Get mask for current expert
                    expert_mask = sec_mask[:, expert_idx, :]  # [num_tokens, capacity]
                    
                    # Use masked_select and reshape for efficient assignment
                    selected_tokens = torch.masked_select(
                        x_flat.unsqueeze(1).expand(-1, expert_capacity, -1),
                        expert_mask.unsqueeze(-1)
                    ).view(-1, hidden_dim)
                    
                    # Fill only up to the number of selected tokens
                    num_selected = selected_tokens.shape[0]
                    if num_selected > 0:
                        dispatched_input[expert_idx, :num_selected] = selected_tokens
            else:
                # Fall back to loop for very large models
                for token_idx in range(num_tokens):
                    for expert_idx in range(self.config.n_experts):
                        for capacity_slot_idx in range(expert_capacity):
                            if sec_mask[token_idx, expert_idx, capacity_slot_idx]:
                                dispatched_input[expert_idx, capacity_slot_idx] = x_flat[token_idx]
            
            # Process tokens through experts
            processed_by_experts = self.experts(dispatched_input)
            
            # Combine expert outputs efficiently
            # Instead of expanding tensors, use einsum for the weighted sum
            # cb_weight: [num_tokens, n_experts, capacity]
            # processed_by_experts: [n_experts, capacity, hidden_dim]
            # Result: [num_tokens, hidden_dim]
            final_output_flat = torch.einsum('nec,ech->nh', cb_weight, processed_by_experts)
            
            # Reshape back to batch form
            x_mlp_out = final_output_flat.view(batch_size, seq_len, hidden_dim)
        else:
            x_mlp_out = self.mlp(x_ln)
        
        x = x + x_mlp_out
        return x


class GPT2LMNoBiasModel(torch.nn.Module):
    def __init__(self, config: GPT2Config, moe_config, return_attention=False):
        super().__init__()
        self.config = config # Store config
        self.return_attention = return_attention
        self.attention_weights = [] if return_attention else None

        self.transformer = torch.nn.ModuleDict(dict(
            wte=torch.nn.Embedding(config.vocab_size, config.n_embd),
            wpe=torch.nn.Embedding(config.n_positions, config.n_embd),
            drop=torch.nn.Dropout(config.embd_pdrop),
            # Pass the potentially existing list to Block
            h=torch.nn.ModuleList([Block(moe_config, self.attention_weights) for _ in range(config.n_layer)]),
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

        # Initialize aux losses to None
        aux_loss = None
        router_z_loss = None

        if labels is not None:
            logits = self.lm_head(x)
            # Ensure labels are long type and handle ignore_index if necessary
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1).long(), ignore_index=-100)

            ## moe loss
            is_moe_model = hasattr(self.config, 'n_experts') and getattr(self.config, 'n_experts', 0) > 1
            if is_moe_model:
                if getattr(self.config, 'use_aux_loss', False):
                    aux_loss = MANAGER.aggregate_aux_loss()
                    if aux_loss is not None:
                        loss += getattr(self.config, 'aux_loss_weight', 0.01) * aux_loss
                    MANAGER.reset_aux_loss()
                if getattr(self.config, 'use_router_z_loss', False):
                    router_z_loss = MANAGER.aggregate_router_z_loss()
                    if router_z_loss is not None:
                        loss += getattr(self.config, 'router_z_loss_weight', 0.001) * router_z_loss
                    MANAGER.reset_router_z_loss()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return ModelOutput(loss=loss, logits=logits, aux_loss=aux_loss, router_z_loss=router_z_loss)

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
        # Run the router in full precision if needed
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        ctx = nullcontext() if not self.router_use_full_prec else torch.amp.autocast(device_type=device_type, enabled=False)

        with ctx:
            # x should be 2D tensor [num_tokens, hidden_dim]
            assert x.dim() == 2, f"Router input x must be 2D (x_flat), but got shape {x.shape}"
            num_tokens = x.shape[0]

            # Get router logits
            logits = self.w_g(x)  # [num_tokens, n_exp]
            
            # Add noise if configured
            if self.use_noisy_top_k and self.training:
                noise = F.softplus(self.w_noise(x))
                noise *= torch.randn_like(noise)
                logits += noise

            # Router z-loss to prevent logits from becoming too large
            if self.use_router_z_loss and self.training:
                z_loss = torch.mean(torch.logsumexp(logits, dim=-1) ** 2.0)
                MANAGER.add_router_z_loss(z_loss)

            # Find top-k experts per token
            top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)  # [num_tokens, top_k]
            
            # Convert to router probabilities (normalize)
            router_probs = torch.full_like(logits, float('-inf'))
            router_probs.scatter_(-1, top_k_indices, top_k_logits)
            router_probs = F.softmax(router_probs, dim=-1)  # [num_tokens, n_exp]

            # Compute auxiliary load balancing loss
            if self.use_aux_loss and self.training:
                # Calculate mean of one-hot encodings
                one_hot = F.one_hot(top_k_indices, num_classes=self.n_exp).float()  # [num_tokens, top_k, n_exp]
                one_hot = one_hot.sum(dim=1)  # [num_tokens, n_exp]
                tokens_per_expert = one_hot.mean(dim=0)  # [n_exp]
                
                # Calculate mean of router probabilities
                prob_per_expert = router_probs.mean(dim=0)  # [n_exp]
                
                # Compute auxiliary loss
                aux_loss = self.n_exp * torch.sum(prob_per_expert * tokens_per_expert)
                MANAGER.add_aux_loss(aux_loss)

            # Calculate expert capacity - how many tokens each expert can process
            capacity = self.get_capacity(num_tokens)
            
            # Create expert assignment and capacity tracking
            # This is a more memory-efficient implementation
            token_expert_indices = []
            expert_capacity_usage = torch.zeros(self.n_exp, device=x.device, dtype=torch.int32)
            
            # Create dispatch map (which token goes to which expert at which position)
            # Instead of creating a large tensor, build a list of assignments
            expert_assignments = []
            for token_idx in range(num_tokens):
                for expert_idx in top_k_indices[token_idx]:
                    # Check if expert has capacity
                    if expert_capacity_usage[expert_idx] < capacity:
                        # Record token → expert assignment
                        expert_assignments.append((token_idx, expert_idx.item(), expert_capacity_usage[expert_idx].item()))
                        expert_capacity_usage[expert_idx] += 1
            
            # Create routing matrices more efficiently
            # Instead of large tensor operations, we'll build from the assignments
            used_capacity = expert_capacity_usage.clone()
            
            # Create dispatch tensors
            dispatched_input = torch.zeros(self.n_exp, capacity, x.shape[1], device=x.device, dtype=x.dtype)
            
            # Create weight matrix for combining outputs
            cb_weight = torch.zeros(num_tokens, self.n_exp, capacity, device=x.device, dtype=x.dtype)
            
            # Create binary mask
            sec_mask = torch.zeros(num_tokens, self.n_exp, capacity, device=x.device, dtype=torch.bool)
            
            # Fill the tensors based on assignments
            for token_idx, expert_idx, slot_idx in expert_assignments:
                # Get router probability for this token-expert pair
                weight = router_probs[token_idx, expert_idx]
                
                # Fill dispatch tensors
                dispatched_input[expert_idx, slot_idx] = x[token_idx]
                cb_weight[token_idx, expert_idx, slot_idx] = weight
                sec_mask[token_idx, expert_idx, slot_idx] = True
            
            return used_capacity, cb_weight, sec_mask
    
    def get_capacity(self, tokens_per_batch):
        """Calculate expert capacity based on batch size"""
        capacity_factor = self.train_capacity if self.training else self.eval_capacity
        capacity = math.floor(self.top_k * capacity_factor * tokens_per_batch / self.n_exp)
        capacity += capacity % 2  # make sure capacity is an even number
        capacity = max(capacity, self.min_capacity)
        return int(capacity)