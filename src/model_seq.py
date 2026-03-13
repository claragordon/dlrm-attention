import torch
import torch.nn as nn


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
    def __init__(self, emb_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError("emb_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, emb_dim),
        )
        self.norm2 = nn.LayerNorm(emb_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
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
    ):
        super().__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim

        self.token_emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(seq_len, emb_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
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
