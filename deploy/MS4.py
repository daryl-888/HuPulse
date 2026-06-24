"""
MS4.py — drop-in replacement for Model_Def/MS4.py on Carya.

scp to:
    /project/rhu/PulseBP/Pulse/PulseDB_multi_full/Model_Training/Model_Def/MS4.py

What changed vs baseline:
  - S4Model: VERBATIM COPY — untouched, same s42 import, same parameters.
  - MS4_1D: fusion replaced with FiLM conditioning.
      * Demographics (3-dim) predict per-channel scale+shift of 512-dim S4 output.
      * Zero-initialized: model starts identical to original, then learns to modulate.
      * Directly addresses paper Section 5: "512-dim output dominates; demographic
        contribution becomes marginal via simple concatenation."
      * Paper's own future-work suggestion: "cross-modal conditioning."
  - MS4_1D_Gate: sigmoid gate — demographics suppress/amplify S4 channels.
      * Paper's own future-work suggestion: "attention/gating."
  - MS4_1D_FiLMGate: FiLM then gate combined.

Bootstrap usage (MODEL_NAME = "ms4" unchanged):
    MS4_1D (FiLM) runs automatically — no other file needs to change.

Variant runs use scripts/train_ms4plus_carya.py --variant gate|filmgate.
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
        x = x.transpose(-1, -2)  # (B, L, d_model)
        if self.pooling:
            x = x.mean(dim=1)
        if self.decoder is not None:
            x = self.decoder(x)
        if not self.pooling and self.transposed_input:
            x = x.transpose(-1, -2)
        return x


# =============================================================================
# MS4_1D — FiLM fusion (replaces original concat fusion)
# =============================================================================

class MS4_1D(nn.Module):
    """
    Drop-in for MODEL_NAME = "ms4" in Model_training_bootstrap.py.

    S4 backbone: identical to original (d_model=512, n_layers=4, d_state=64, l_max=2048).
    Fusion: FiLM — demographics predict per-channel scale+shift of S4 output.
    Zero-init on FiLM layer so the model starts as the original and learns to modulate.
    """

    def __init__(self, num_static_features=3, num_BP=1,
                 s4_d_model=512, s4_n_layers=4, s4_d_state=64, s4_l_max=2048,
                 demo_hidden=64, dropout=0.1,
                 # legacy args kept so existing call-sites don't break
                 static_hidden=16, fusion_hidden=32,
                 s4_d_input=1, s4_pooling=True):
        super().__init__()
        self.s4model = S4Model(
            d_input=s4_d_input, d_output=None, d_state=s4_d_state,
            d_model=s4_d_model, n_layers=s4_n_layers,
            pooling=s4_pooling, l_max=s4_l_max,
        )
        D = s4_d_model  # 512

        self.demo_mlp = nn.Sequential(
            nn.Linear(num_static_features, demo_hidden), nn.GELU(),
            nn.Linear(demo_hidden, demo_hidden), nn.GELU(),
        )
        # FiLM: predict gamma and beta — zero-init for identity at step 0
        self.film = nn.Linear(demo_hidden, D * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        self.head = nn.Sequential(
            nn.Linear(D, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 32), nn.ReLU(inplace=True),
            nn.Linear(32, num_BP),
        )

    def forward(self, ppg, static_feat):
        """ppg: (B, 1, L)   static_feat: (B, num_static_features)"""
        s4_out = self.s4model(ppg)              # (B, 512)
        demo   = self.demo_mlp(static_feat)     # (B, 64)
        params = self.film(demo)                # (B, 1024)
        gamma, beta = params[:, :512], params[:, 512:]
        s4_out = (1.0 + gamma) * s4_out + beta  # FiLM
        return self.head(s4_out).squeeze(-1)    # (B,)


# =============================================================================
# MS4_1D_Gate — sigmoid gating variant
# =============================================================================

class MS4_1D_Gate(nn.Module):
    """
    Demographics predict a sigmoid gate over 512-dim S4 output.
    Gate init bias=1 so sigmoid(1)≈0.73 — softly open at start.
    """

    def __init__(self, num_static_features=3, num_BP=1,
                 s4_d_model=512, s4_n_layers=4, s4_d_state=64, s4_l_max=2048,
                 demo_hidden=64, dropout=0.1):
        super().__init__()
        self.s4model = S4Model(
            d_input=1, d_output=None, d_state=s4_d_state,
            d_model=s4_d_model, n_layers=s4_n_layers,
            pooling=True, l_max=s4_l_max,
        )
        D = s4_d_model
        self.demo_mlp = nn.Sequential(
            nn.Linear(num_static_features, demo_hidden), nn.GELU(),
            nn.Linear(demo_hidden, demo_hidden), nn.GELU(),
        )
        self.gate_linear = nn.Linear(demo_hidden, D)
        nn.init.zeros_(self.gate_linear.weight)
        nn.init.constant_(self.gate_linear.bias, 1.0)
        self.head = nn.Sequential(
            nn.Linear(D, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 32), nn.ReLU(inplace=True),
            nn.Linear(32, num_BP),
        )

    def forward(self, ppg, static_feat):
        s4_out = self.s4model(ppg)
        demo   = self.demo_mlp(static_feat)
        gate   = torch.sigmoid(self.gate_linear(demo))
        s4_out = s4_out * gate
        return self.head(s4_out).squeeze(-1)


# =============================================================================
# MS4_1D_FiLMGate — FiLM then gate combined
# =============================================================================

class MS4_1D_FiLMGate(nn.Module):
    """FiLM scale+shift, then sigmoid gate."""

    def __init__(self, num_static_features=3, num_BP=1,
                 s4_d_model=512, s4_n_layers=4, s4_d_state=64, s4_l_max=2048,
                 demo_hidden=64, dropout=0.1):
        super().__init__()
        self.s4model = S4Model(
            d_input=1, d_output=None, d_state=s4_d_state,
            d_model=s4_d_model, n_layers=s4_n_layers,
            pooling=True, l_max=s4_l_max,
        )
        D = s4_d_model
        self.demo_mlp = nn.Sequential(
            nn.Linear(num_static_features, demo_hidden), nn.GELU(),
            nn.Linear(demo_hidden, demo_hidden), nn.GELU(),
        )
        self.film = nn.Linear(demo_hidden, D * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.gate_linear = nn.Linear(demo_hidden, D)
        nn.init.zeros_(self.gate_linear.weight)
        nn.init.constant_(self.gate_linear.bias, 1.0)
        self.head = nn.Sequential(
            nn.Linear(D, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 32), nn.ReLU(inplace=True),
            nn.Linear(32, num_BP),
        )

    def forward(self, ppg, static_feat):
        s4_out = self.s4model(ppg)
        demo   = self.demo_mlp(static_feat)
        params = self.film(demo)
        gamma, beta = params[:, :512], params[:, 512:]
        s4_out = (1.0 + gamma) * s4_out + beta
        gate   = torch.sigmoid(self.gate_linear(demo))
        s4_out = s4_out * gate
        return self.head(s4_out).squeeze(-1)
