import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import flex_attention

    _HAS_FLEX_ATTENTION = True
except Exception:
    flex_attention = None
    _HAS_FLEX_ATTENTION = False


class MLP(nn.Module):
    def __init__(self, in_dim, layer_dims, dropout=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for h in layer_dims:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        emb_dim,
        num_heads,
        ffn_dim,
        dropout=0.1,
        use_flex_attention=False,
        recency_bias=0.0,
        causal=False,
    ):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError("emb_dim must be divisible by num_heads")
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.dropout = dropout
        self.use_flex_attention = use_flex_attention
        self.recency_bias = float(recency_bias)
        self.causal = causal

        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.o_proj = nn.Linear(emb_dim, emb_dim)

        self.norm1 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, emb_dim),
        )
        self.norm2 = nn.LayerNorm(emb_dim)
        self.drop = nn.Dropout(dropout)

    def _reshape_heads(self, x):
        # [B, L, D] -> [B, H, L, Dh]
        bsz, seqlen, _ = x.shape
        return x.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

    def _recency_bias_matrix(self, seqlen, dtype, device):
        if self.recency_bias == 0.0:
            return None
        pos = torch.arange(seqlen, device=device)
        dist = torch.abs(pos[:, None] - pos[None, :]).to(dtype)
        return -self.recency_bias * dist

    def _flex_attention(self, q, k, v):
        def score_mod(score, _b, _h, q_idx, kv_idx):
            if self.recency_bias != 0.0:
                dist = torch.abs(q_idx - kv_idx).to(score.dtype)
                score = score - self.recency_bias * dist
            if self.causal:
                score = torch.where(kv_idx > q_idx, torch.full_like(score, -float("inf")), score)
            return score

        return flex_attention(q, k, v, score_mod=score_mod)

    def _sdpa_attention(self, q, k, v, key_padding_mask=None):
        # q,k,v: [B, H, L, Dh]
        bsz, _heads, seqlen, _dh = q.shape
        attn_mask = self._recency_bias_matrix(seqlen, q.dtype, q.device)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1,1,L,L]
        if self.causal:
            causal_mask = torch.triu(
                torch.full((seqlen, seqlen), -float("inf"), dtype=q.dtype, device=q.device),
                diagonal=1,
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            attn_mask = causal_mask if attn_mask is None else attn_mask + causal_mask
        if key_padding_mask is not None:
            kpm = key_padding_mask.unsqueeze(1).unsqueeze(1)  # [B,1,1,L]
            pad_bias = torch.zeros((bsz, 1, 1, seqlen), dtype=q.dtype, device=q.device)
            pad_bias = pad_bias.masked_fill(kpm, -float("inf"))
            attn_mask = pad_bias if attn_mask is None else attn_mask + pad_bias
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # causal handled via mask above
        )

    def forward(self, x, key_padding_mask=None):
        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))

        use_flex_now = self.use_flex_attention and not (
            key_padding_mask is not None and bool(key_padding_mask.any().item())
        )
        if use_flex_now:
            attn = self._flex_attention(q, k, v)
        else:
            attn = self._sdpa_attention(q, k, v, key_padding_mask=key_padding_mask)

        # [B, H, L, Dh] -> [B, L, D]
        attn_out = attn.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.emb_dim)
        attn_out = self.o_proj(attn_out)
        x = self.norm1(x + self.drop(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop(ffn_out))
        return x


class SeqCTRModel(nn.Module):
    """
    Sequence CTR model for fixed-length token sequences:
    - One shared token embedding table
    - Learnable positional embedding
    - Stacked self-attention blocks
    - Masked mean pooling -> MLP -> logit
    """

    def __init__(
        self,
        vocab_size=131072,
        seq_len=128,
        emb_dim=64,
        num_layers=2,
        num_heads=4,
        ffn_dim=256,
        dropout=0.1,
        top_mlp=(256, 128),
        use_flex_attention=False,
        recency_bias=0.0,
        causal=False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.use_flex_attention = bool(use_flex_attention)
        if self.use_flex_attention and not _HAS_FLEX_ATTENTION:
            raise RuntimeError(
                "use_flex_attention=True but torch flex_attention is unavailable in this environment."
            )

        self.token_emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(seq_len, emb_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    use_flex_attention=self.use_flex_attention,
                    recency_bias=recency_bias,
                    causal=causal,
                )
                for _ in range(num_layers)
            ]
        )
        self.top = MLP(emb_dim, top_mlp, dropout=dropout)
        self.out = nn.Linear(top_mlp[-1], 1)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.01)

    def forward(self, seq_ids: torch.Tensor) -> torch.Tensor:
        """
        seq_ids: [B, seq_len] int64, with 0 reserved as padding/missing.
        """
        if seq_ids.dim() != 2:
            raise ValueError("seq_ids must be [B, seq_len]")
        if seq_ids.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_ids.size(1)}")

        bsz = seq_ids.size(0)
        pos_idx = torch.arange(self.seq_len, device=seq_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.token_emb(seq_ids) + self.pos_emb(pos_idx)

        pad_mask = seq_ids.eq(0)  # [B, L], True means pad
        for block in self.blocks:
            x = block(x, key_padding_mask=pad_mask)

        # Masked mean pool so padding tokens do not contribute.
        valid = (~pad_mask).unsqueeze(-1).float()  # [B, L, 1]
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        x = self.top(pooled)
        logit = self.out(x).squeeze(1)  # [B]
        return logit
