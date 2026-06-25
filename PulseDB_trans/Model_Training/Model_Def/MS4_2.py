import torch
import torch.nn as nn

from .s42 import S4 as S42

class S4Block(nn.Module):
    def __init__(self, d_model, d_state=64, l_max=2048, dropout=0.2, bidirectional=True, use_layernorm=True):
        super().__init__()
        self.use_ln = use_layernorm
        self.norm1 = nn.LayerNorm(d_model) if use_layernorm else nn.BatchNorm1d(d_model)
        self.norm2 = nn.LayerNorm(d_model) if use_layernorm else nn.BatchNorm1d(d_model)

        self.s4 = S42(
            d_state=d_state,
            l_max=l_max,
            d_model=d_model,
            bidirectional=bidirectional,
            postact='glu',
            dropout=dropout,
            transposed=True,
        )
        self.do = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Conv1d(d_model, 4*d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(4*d_model, d_model, kernel_size=1),
        )

    def _maybe_norm(self, x, norm):
        if self.use_ln:
            return norm(x.transpose(-1, -2)).transpose(-1, -2)  # (B,L,C) <-> LN
        else:
            return norm(x)  # BN: (B,C,L)

    def forward(self, x, rate=1.0):  # x: (B,C,L)
        y = self._maybe_norm(x, self.norm1)
        y, _ = self.s4(y, rate=rate)      # keep (B,C,L)
        x = x + self.do(y)
        y = self._maybe_norm(x, self.norm2)
        y = self.ffn(y)
        x = x + self.do(y)
        return x


class AttnPool1D(nn.Module):
    def __init__(self, d_model, hidden=128):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):  # x: (B, L, C)
        w = self.scorer(x).squeeze(-1)     # (B, L)
        w = torch.softmax(w, dim=1)
        return torch.bmm(w.unsqueeze(1), x).squeeze(1)  # (B, C)


class FiLM(nn.Module):
    def __init__(self, static_hidden, d_model):
        super().__init__()
        self.proj = nn.Linear(static_hidden, 2*d_model)

    def forward(self, x, s):  # x: (B,L,C), s: (B,static_hidden)
        gamma, beta = self.proj(s).chunk(2, dim=-1)  # (B,C), (B,C)
        return x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class MS4_1D(nn.Module):
    def __init__(
        self,
        num_static_features=3,
        num_BP=1,
        static_hidden=32,
        fusion_hidden=64,
        s4_d_input=1,
        s4_d_model=256,
        s4_n_layers=4,
        s4_l_max=2048,
        dropout=0.2,
        bidirectional=True
    ):
        super().__init__()

        # Convolutional stem before S4 (denoise + local patterns)
        self.stem = nn.Sequential(
            nn.Conv1d(s4_d_input, s4_d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(s4_d_model, s4_d_model, kernel_size=3, padding=1),
            nn.GELU()
        )

        # Stack of S4 blocks
        self.blocks = nn.ModuleList([
            S4Block(
                d_model=s4_d_model,
                d_state=64,
                l_max=s4_l_max,
                dropout=dropout,
                bidirectional=bidirectional,
                use_layernorm=True
            ) for _ in range(s4_n_layers)
        ])

        # Static MLP (embed + nonlinearity)
        self.static_mlp = nn.Sequential(
            nn.Linear(num_static_features, static_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(static_hidden, static_hidden),
            nn.ReLU(inplace=True)
        )

        # FiLM to modulate sequence features with demographics
        self.film = FiLM(static_hidden, s4_d_model)

        # Attention pooling over time
        self.pool = AttnPool1D(d_model=s4_d_model, hidden=128)

        # Fusion head (concat pooled seq + static)
        self.head = nn.Sequential(
            nn.Linear(s4_d_model + static_hidden, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_BP)
        )

    def forward(self, ppg, static_feat):
        """
        ppg: (B, 1, L)
        static_feat: (B, num_static_features)
        """
        x = self.stem(ppg)  # (B, C, L)

        for blk in self.blocks:
            x = blk(x)      # (B, C, L)

        # (B, C, L) -> (B, L, C) for FiLM + pooling
        x = x.transpose(-1, -2)

        s = self.static_mlp(static_feat)         # (B, static_hidden)
        x = self.film(x, s)                      # (B, L, C) conditioned by demographics

        x = self.pool(x)                         # (B, C)
        fused = torch.cat([x, s], dim=-1)        # (B, C+static_hidden)
        out = self.head(fused)                   # (B, num_BP)
        return out.squeeze(-1)
