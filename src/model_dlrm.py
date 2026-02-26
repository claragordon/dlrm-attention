import math
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

class DLRM(nn.Module):
    """
    Classic DLRM-ish:
    - 26 categorical fields -> 26 embedding tables (hashed IDs already)
    - 13 dense -> bottom MLP -> d
    - Interaction: pairwise dot products over tokens (dense + each embedding)
    - Top MLP -> logit
    """
    def __init__(
        self,
        num_sparse_fields=26,
        num_dense=13,
        emb_dim=64,
        hash_size=131072,          # must match preprocessing
        bottom_mlp=(128, 64),      # ends at emb_dim
        top_mlp=(512, 256),
        dropout=0.0,
        use_attention=False,
        attention_heads=4,
    ):
        super().__init__()
        self.num_sparse_fields = num_sparse_fields
        self.emb_dim = emb_dim
        self.hash_size = hash_size
        self.use_attention = use_attention

        # One embedding table per field
        self.embs = nn.ModuleList([
            nn.Embedding(hash_size + 1, emb_dim, padding_idx=0)
            for _ in range(num_sparse_fields)
        ])

        # Dense -> emb_dim
        assert bottom_mlp[-1] == emb_dim, "bottom_mlp must end at emb_dim"
        self.bottom = MLP(num_dense, bottom_mlp, dropout=dropout)

        if self.use_attention:
            if emb_dim % attention_heads != 0:
                raise ValueError("emb_dim must be divisible by attention_heads")
            self.attn = nn.MultiheadAttention(
                embed_dim=emb_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(emb_dim)

        # Interaction dim = emb_dim + T*(T-1)/2 where T = 1 + num_sparse_fields
        T = 1 + num_sparse_fields
        interact_dim = emb_dim + (T * (T - 1)) // 2

        self.top = MLP(interact_dim, top_mlp, dropout=dropout)
        self.out = nn.Linear(top_mlp[-1], 1)

        self._init_embeddings()

    def _init_embeddings(self):
        # Reasonable init; keep simple
        for e in self.embs:
            nn.init.normal_(e.weight, mean=0.0, std=0.01)

    def interaction(self, x_tokens: torch.Tensor) -> torch.Tensor:
        """
        x_tokens: [B, T, D] where T = 1 + num_sparse_fields
        Returns: [B, D + T*(T-1)/2]
        """
        B, T, D = x_tokens.shape
        # Pairwise dot products: for i<j, dot(x_i, x_j)
        # Compute all dot products via batch matmul: [B, T, T]
        gram = torch.bmm(x_tokens, x_tokens.transpose(1, 2))
        # Extract upper triangle (excluding diagonal)
        iu = torch.triu_indices(T, T, offset=1, device=x_tokens.device)
        dots = gram[:, iu[0], iu[1]]  # [B, T*(T-1)/2]
        # Concatenate dense token (token 0) with dot features
        dense_tok = x_tokens[:, 0, :]  # [B, D]
        return torch.cat([dense_tok, dots], dim=1)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        """
        dense: [B, 13] float32
        sparse: [B, 26] int64
        """
        # bottom MLP gives dense token
        dense_tok = self.bottom(dense)  # [B, D]

        # embeddings: list of [B, D]
        emb_toks = []
        for j, emb in enumerate(self.embs):
            emb_toks.append(emb(sparse[:, j]))

        # stack tokens: [B, T, D]
        x_tokens = torch.stack([dense_tok] + emb_toks, dim=1)
        if self.use_attention:
            attn_out, _ = self.attn(x_tokens, x_tokens, x_tokens, need_weights=False)
            # Residual + layer norm stabilizes attention training.
            x_tokens = self.attn_norm(x_tokens + attn_out)

        x = self.interaction(x_tokens)
        x = self.top(x)
        logit = self.out(x).squeeze(1)  # [B]
        return logit