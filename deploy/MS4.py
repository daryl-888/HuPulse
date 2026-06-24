"""
MS4.py — drop-in replacement for Model_Def/MS4.py on Carya.

scp to:
    /project/rhu/PulseBP/Pulse/PulseDB_multi_full/Model_Training/Model_Def/MS4.py

Architecture (based on MS4_2.py which was never run through the bootstrap):
  - S4Model: VERBATIM COPY from original MS4.py — untouched.
  - S4Block: pre-norm residual block with S4 + pointwise FFN (like a Transformer block).
  - AttnPool1D: attention-weighted pooling over time (learned, vs. dumb mean pool).
  - FiLM: applied to the FULL SEQUENCE (B, L, C) before pooling — modulates all
    timesteps, not just the pooled vector. This is strictly better than post-pool FiLM.

Variants (all share the same S4/S4Block/stem — only fusion differs):
  MS4_1D        — conv stem + S4Block + sequence FiLM + attn pool  [bootstrap default]
  MS4_1D_Gate   — same but sigmoid gate on sequence instead of FiLM
  MS4_1D_FiLMGate — FiLM then gate, both on sequence
"""
import torch
import torch.nn as nn

from .s42 import S4 as S42


# =============================================================================
# S4Model — VERBATIM COPY from original MS4.py — DO NOT MODIFY
# =============================================================================

class S4Model(nn.Module):
    def __init__(
        self,
        d_input,
        d_output,
        d_state=64,
        d_model=512,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        l_max=1024,
        transposed_input=True,
        bidirectional=True,
        layer_norm=True,
        pooling=True,
    ):
        super().__init__()
        self.prenorm = prenorm
        self.transposed_input = transposed_input
        self.layer_norm = layer_norm

        if d_input is None:
            self.encoder = nn.Identity()
        else:
            self.encoder = nn.Conv1d(d_input, d_model, 1) if transposed_input else nn.Linear(d_input, d_model)

        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S42(
                    d_state=d_state,
                    l_max=l_max,
                    d_model=d_model,
                    bidirectional=bidirectional,
                    postact='glu',
                    dropout=dropout,
                    transposed=True,
                )
            )
            if layer_norm:
                self.norms.append(nn.LayerNorm(d_model))
            else:
                self.norms.append(nn.BatchNorm1d(d_model))
            self.dropouts.append(nn.Dropout2d(dropout))

        self.pooling = pooling
        if d_output is None:
            self.decoder = None
        else:
            self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x, rate=1.0):
        x = self.encoder(x)
        if not self.transposed_input:
            x = x.transpose(-1, -2)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2) if self.layer_norm else norm(z)
            z, _ = layer(z, rate=rate)
            z = dropout(z)
            x = z + x
            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2) if self.layer_norm else norm(z)
        x = x.transpose(-1, -2)
        if self.pooling:
            x = x.mean(dim=1)
        if self.decoder is not None:
            x = self.decoder(x)
        if not self.pooling and self.transposed_input:
            x = x.transpose(-1, -2)
        return x


# =============================================================================
# Building blocks (from MS4_2.py)
# =============================================================================

class S4Block(nn.Module):
    """Pre-norm residual block: S4 + pointwise FFN (like a Transformer block)."""
    def __init__(self, d_model, d_state=64, l_max=2048, dropout=0.2, bidirectional=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.s4 = S42(d_state=d_state, l_max=l_max, d_model=d_model,
                      bidirectional=bidirectional, postact='glu',
                      dropout=dropout, transposed=True)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Conv1d(d_model, 4 * d_model, kernel_size=1), nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(4 * d_model, d_model, kernel_size=1),
        )

    def forward(self, x, rate=1.0):  # x: (B, C, L)
        y = self.norm1(x.transpose(-1, -2)).transpose(-1, -2)
        y, _ = self.s4(y, rate=rate)
        x = x + self.drop(y)
        y = self.norm2(x.transpose(-1, -2)).transpose(-1, -2)
        x = x + self.drop(self.ffn(y))
        return x


class AttnPool1D(nn.Module):
    """Learned attention pooling over time — smarter than mean pool."""
    def __init__(self, d_model, hidden=128):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(d_model, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):  # x: (B, L, C)
        w = torch.softmax(self.scorer(x).squeeze(-1), dim=1)  # (B, L)
        return torch.bmm(w.unsqueeze(1), x).squeeze(1)         # (B, C)


class _FiLM(nn.Module):
    """FiLM on sequence: modulates (B, L, C) given static embedding (B, D)."""
    def __init__(self, static_dim, d_model):
        super().__init__()
        self.proj = nn.Linear(static_dim, 2 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, s):  # x: (B, L, C),  s: (B, D)
        gamma, beta = self.proj(s).chunk(2, dim=-1)  # (B, C) each
        return x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class _Gate(nn.Module):
    """Sigmoid gate on sequence: demographics suppress/amplify each channel."""
    def __init__(self, static_dim, d_model):
        super().__init__()
        self.proj = nn.Linear(static_dim, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 1.0)

    def forward(self, x, s):  # x: (B, L, C),  s: (B, D)
        g = torch.sigmoid(self.proj(s)).unsqueeze(1)  # (B, 1, C)
        return x * g


# =============================================================================
# MS4_1D — FiLM on sequence  [bootstrap default: MODEL_NAME = "ms4"]
# =============================================================================

class MS4_1D(nn.Module):
    """
    Conv stem → S4Blocks → sequence FiLM → attention pool → head.

    Improvements over baseline MS4.py:
      1. Conv stem (k=5, k=3) extracts local PPG morphology before S4.
      2. S4Block adds pointwise FFN after each S4 layer (Transformer-style).
      3. FiLM on the sequence (B,L,C) — demographics modulate ALL timesteps,
         not just the pooled summary. This is the key fix for feature dominance.
      4. AttnPool1D — learned weighted pooling vs. mean pool.
    """

    def __init__(self, num_static_features=3, num_BP=1,
                 static_hidden=32, fusion_hidden=64,
                 s4_d_input=1, s4_d_model=256, s4_n_layers=4,
                 s4_pooling=True, s4_l_max=2048, dropout=0.2):
        super().__init__()
        D = s4_d_model

        self.stem = nn.Sequential(
            nn.Conv1d(s4_d_input, D, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(D, D, kernel_size=3, padding=1), nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            S4Block(D, d_state=64, l_max=s4_l_max, dropout=dropout)
            for _ in range(s4_n_layers)
        ])
        self.static_mlp = nn.Sequential(
            nn.Linear(num_static_features, static_hidden), nn.ReLU(inplace=True),
            nn.Linear(static_hidden, static_hidden), nn.ReLU(inplace=True),
        )
        self.film = _FiLM(static_hidden, D)
        self.pool = AttnPool1D(D, hidden=128)
        self.head = nn.Sequential(
            nn.Linear(D + static_hidden, fusion_hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_BP),
        )

    def forward(self, ppg, static_feat):
        x = self.stem(ppg)                        # (B, C, L)
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(-1, -2)                   # (B, L, C)
        s = self.static_mlp(static_feat)           # (B, static_hidden)
        x = self.film(x, s)                        # (B, L, C) — FiLM on sequence
        x = self.pool(x)                           # (B, C)
        return self.head(torch.cat([x, s], dim=-1)).squeeze(-1)


# =============================================================================
# MS4_1D_Gate — gate on sequence instead of FiLM
# =============================================================================

class MS4_1D_Gate(nn.Module):
    """Same backbone as MS4_1D but uses sigmoid gate instead of FiLM."""

    def __init__(self, num_static_features=3, num_BP=1,
                 static_hidden=32, fusion_hidden=64,
                 s4_d_input=1, s4_d_model=256, s4_n_layers=4,
                 s4_l_max=2048, dropout=0.2):
        super().__init__()
        D = s4_d_model
        self.stem = nn.Sequential(
            nn.Conv1d(s4_d_input, D, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(D, D, kernel_size=3, padding=1), nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            S4Block(D, d_state=64, l_max=s4_l_max, dropout=dropout)
            for _ in range(s4_n_layers)
        ])
        self.static_mlp = nn.Sequential(
            nn.Linear(num_static_features, static_hidden), nn.ReLU(inplace=True),
            nn.Linear(static_hidden, static_hidden), nn.ReLU(inplace=True),
        )
        self.gate = _Gate(static_hidden, D)
        self.pool = AttnPool1D(D, hidden=128)
        self.head = nn.Sequential(
            nn.Linear(D + static_hidden, fusion_hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_BP),
        )

    def forward(self, ppg, static_feat):
        x = self.stem(ppg)
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(-1, -2)
        s = self.static_mlp(static_feat)
        x = self.gate(x, s)
        x = self.pool(x)
        return self.head(torch.cat([x, s], dim=-1)).squeeze(-1)


# =============================================================================
# MS4_1D_FiLMGate — FiLM then gate, both on sequence
# =============================================================================

class MS4_1D_FiLMGate(nn.Module):
    """FiLM scale+shift, then sigmoid gate — both applied to sequence."""

    def __init__(self, num_static_features=3, num_BP=1,
                 static_hidden=32, fusion_hidden=64,
                 s4_d_input=1, s4_d_model=256, s4_n_layers=4,
                 s4_l_max=2048, dropout=0.2):
        super().__init__()
        D = s4_d_model
        self.stem = nn.Sequential(
            nn.Conv1d(s4_d_input, D, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(D, D, kernel_size=3, padding=1), nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            S4Block(D, d_state=64, l_max=s4_l_max, dropout=dropout)
            for _ in range(s4_n_layers)
        ])
        self.static_mlp = nn.Sequential(
            nn.Linear(num_static_features, static_hidden), nn.ReLU(inplace=True),
            nn.Linear(static_hidden, static_hidden), nn.ReLU(inplace=True),
        )
        self.film = _FiLM(static_hidden, D)
        self.gate = _Gate(static_hidden, D)
        self.pool = AttnPool1D(D, hidden=128)
        self.head = nn.Sequential(
            nn.Linear(D + static_hidden, fusion_hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_BP),
        )

    def forward(self, ppg, static_feat):
        x = self.stem(ppg)
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(-1, -2)
        s = self.static_mlp(static_feat)
        x = self.film(x, s)
        x = self.gate(x, s)
        x = self.pool(x)
        return self.head(torch.cat([x, s], dim=-1)).squeeze(-1)
