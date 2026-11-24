"""Six architectures over one fused EIC token stream, one interface:
    encode(code, mask) -> (B, L, d)
    forward_mlm(code, mask) -> (B, L, V)     pretraining head
    forward_clf(code, mask) -> (B,)          finetuning head

`causal` decides the objective and the pooling: bidirectional -> MLM, mean-pool;
causal -> next-token, last real token.

CONTEXT is model-specific and follows each architecture's published geometry.
Batch size is set so tokens-per-batch is constant (32,768) across models.
"""
import math, torch, torch.nn as nn

PAD = 0

# Published / native context per architecture. Every optimizer step sees the
# same 32,768 tokens (BATCH * ACCUM * CONTEXT); attention is O(L^2) so the
# longer contexts need smaller micro-batches with gradient accumulation.
CONTEXT = {"bilstm": 512, "bert": 512, "gpt2": 1024, "gpt2s": 1024, "gpt2l": 1024,
           "mistral": 2048, "mistrals": 2048, "mistrall": 2048, "bigbird": 4096,
           "mamba": 2048, "mambas": 2048, "mambal": 2048,
           "bilstms": 512, "bilstml": 512, "berts": 512, "bertl": 512}
BATCH   = {"bilstm": 64,  "bert": 64,  "gpt2": 8,    "gpt2s": 4,    "gpt2l": 4,
           "mistral": 8,     "mistrals": 16,   "mistrall": 4,    "bigbird": 2,
           "mamba": 4,       "mambas": 16,     "mambal": 2,
           "bilstms": 64, "bilstml": 64, "berts": 64, "bertl": 32}   # 4090 micro-batches; gpt2 = medium geometry (18.8 GB at 8x1024)
ACCUM   = {a: 32768 // (BATCH[a] * CONTEXT[a]) for a in CONTEXT}
assert all(BATCH[a] * ACCUM[a] * CONTEXT[a] == 32768 for a in CONTEXT)


class _PosEnc(nn.Module):
    def __init__(self, d, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, : x.size(1)]


def _safe_pad_mask(mask):
    """src_key_padding_mask with at least one attendable position per row. A
    fully padded row gives the softmax nothing to normalise over -> NaN."""
    pad = (mask == 0)
    empty = pad.all(dim=1)
    if empty.any():
        pad = pad.clone(); pad[empty, 0] = False
    return pad


def _guard_attention_mask(mask):
    """HF convention (1 = attend), same empty-row guard."""
    empty = mask.sum(1) == 0
    if empty.any():
        mask = mask.clone(); mask[empty, 0] = 1
    return mask


class _Base(nn.Module):
    causal = False
    def _heads(self, vocab, d, dropout):
        self.drop = nn.Dropout(dropout)
        self.mlm = nn.Linear(d, vocab)
        self.clf = nn.Linear(d, 1)
    def forward_mlm(self, code, mask):
        return self.mlm(self.encode(code, mask))
    def forward_clf(self, code, mask):
        h = self.encode(code, mask)
        if self.causal:
            idx = mask.sum(1).clamp(min=1) - 1
            pooled = h[torch.arange(h.size(0), device=h.device), idx]
        else:
            m = mask.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.clf(self.drop(pooled)).squeeze(-1)


def _add_residual_dropout(branches, p):
    """HF Mistral and Mamba expose no residual dropout (large-data designs).
    Hook the *sublayer branches* (attention / MLP / SSM mixer), whose outputs the
    HF layer adds onto the residual stream, so each layer computes
    x + dropout(f(norm(x))) -- GPT-2's resid_pdrop -- like the torch encoders.
    (Hooking the whole layer would zero the residual stream itself.)"""
    drop = nn.Dropout(p)
    def hook(module, args, output):
        if isinstance(output, tuple): return (drop(output[0]),) + tuple(output[1:])
        return drop(output)
    for br in branches: br.register_forward_hook(hook)
    return drop


class BiLSTM(_Base):
    causal = False
    def __init__(self, vocab, d=768, layers=4, dropout=0.1, **kw):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=PAD)
        self.lstm = nn.LSTM(d, d // 2, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self._heads(vocab, d, dropout); self.d = d
    def encode(self, code, mask):
        h, _ = self.lstm(self.drop(self.emb(code))); return h


class BERT(_Base):
    """Bidirectional encoder, MLM, mean pooling. Context 512 (BERT-base)."""
    causal = False
    def __init__(self, vocab, d=768, layers=12, heads=12, dropout=0.1, **kw):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=PAD); self.pos = _PosEnc(d)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self._heads(vocab, d, dropout); self.d = d
    def encode(self, code, mask):
        return self.enc(self.pos(self.drop(self.emb(code))), src_key_padding_mask=_safe_pad_mask(mask))


class GPT2(_Base):
    """Causal, next-token, last-token pooling. Context 1024 (GPT-2)."""
    causal = True
    def __init__(self, vocab, d=768, layers=12, heads=12, dropout=0.1, **kw):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=PAD); self.pos = _PosEnc(d)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self._heads(vocab, d, dropout); self.d = d
    def encode(self, code, mask):
        L = code.size(1)
        causal = torch.triu(torch.ones(L, L, device=code.device, dtype=torch.bool), 1)
        return self.enc(self.pos(self.drop(self.emb(code))), mask=causal,
                        src_key_padding_mask=_safe_pad_mask(mask))


class Mistral(_Base):
    """HF MistralModel: RoPE, SwiGLU, RMSNorm, GQA. Causal. Context 2048 (ETHOS scale)."""
    causal = True
    def __init__(self, vocab, d=768, layers=12, heads=12, dropout=0.1, **kw):
        super().__init__()
        from transformers import MistralConfig, MistralModel
        cfg = MistralConfig(vocab_size=vocab, hidden_size=d, num_hidden_layers=layers,
                            num_attention_heads=heads, num_key_value_heads=max(heads // 4, 1),
                            intermediate_size=d * 4, max_position_embeddings=4096,
                            pad_token_id=PAD, attention_dropout=dropout)
        self.backbone = MistralModel(cfg)
        self._layer_drop = _add_residual_dropout(
            [l.self_attn for l in self.backbone.layers] + [l.mlp for l in self.backbone.layers], dropout)
        self._heads(vocab, d, dropout); self.d = d
    def encode(self, code, mask):
        return self.backbone(input_ids=code, attention_mask=_guard_attention_mask(mask)).last_hidden_state


class BigBird(_Base):
    """HF BigBirdModel, block-sparse attention. Bidirectional, MLM. Context 4096 (published)."""
    causal = False
    def __init__(self, vocab, d=768, layers=12, heads=12, dropout=0.1, **kw):
        super().__init__()
        from transformers import BigBirdConfig, BigBirdModel
        cfg = BigBirdConfig(vocab_size=vocab, hidden_size=d, num_hidden_layers=layers,
                            num_attention_heads=heads, intermediate_size=d * 4,
                            max_position_embeddings=4096, attention_type="block_sparse",
                            block_size=64, num_random_blocks=3, use_bias=True,
                            hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
                            pad_token_id=PAD)
        self.backbone = BigBirdModel(cfg)
        self._heads(vocab, d, dropout); self.d = d
    def encode(self, code, mask):
        return self.backbone(input_ids=code, attention_mask=_guard_attention_mask(mask)).last_hidden_state


class Mamba(_Base):
    """HF MambaModel (selective SSM, mamba_ssm kernels). Causal by construction,
    no attention mask -- right-padding is harmless to a causal scan. Context 2048.
    Trained at mamba-130m depth (24 blocks at d=768): one SSM block is about half a
    transformer block, so 24 of them give parameter parity with the 12-layer models."""
    causal = True
    def __init__(self, vocab, d=768, layers=12, dropout=0.1, **kw):
        super().__init__()
        from transformers import MambaConfig, MambaModel
        cfg = MambaConfig(vocab_size=vocab, hidden_size=d, num_hidden_layers=layers,
                          state_size=16, expand=2, conv_kernel=4, pad_token_id=PAD)
        self.backbone = MambaModel(cfg)
        self._layer_drop = _add_residual_dropout([b.mixer for b in self.backbone.layers], dropout)
        self._heads(vocab, d, dropout); self.d = d
    def encode(self, code, mask):
        return self.backbone(input_ids=code).last_hidden_state


# Size ladders share a class; the launcher sets the geometry: gpt2s/gpt2/gpt2l = GPT-2 small/medium/large,
# mistrals/mistral/mistrall = Mistral architecture at ~34M / 109M / 373M (x3.4 steps, like GPT-2's),
# mambas/mamba/mambal = Mamba at 31M / mamba-130m geometry (97M) / mamba-370m geometry (328M, 48 x 1024).
ARCHS = {"bilstm": BiLSTM, "bert": BERT, "gpt2": GPT2, "gpt2s": GPT2, "gpt2l": GPT2,
         "mistral": Mistral, "mistrals": Mistral, "mistrall": Mistral, "bigbird": BigBird,
         "mamba": Mamba, "mambas": Mamba, "mambal": Mamba,
         "bilstms": BiLSTM, "bilstml": BiLSTM, "berts": BERT, "bertl": BERT}


def build(name, vocab, d=768, **kw):
    if name not in ARCHS:
        raise ValueError(f"unknown architecture {name!r}; have {sorted(ARCHS)}")
    return ARCHS[name](vocab, d=d, **kw)
