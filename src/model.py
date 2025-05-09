import torch
import math
import transformers.activations
from transformers import GPT2Config

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

