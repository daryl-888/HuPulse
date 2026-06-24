"""
MS4Plus: Improved S4-based model for cuffless blood pressure estimation.

Key improvements over the baseline MS4 from the paper:
1. FiLM conditioning: demographics predict scale/shift of PPG features
   instead of simple concatenation (fixes the dominance problem where
   512-dim S4 features drown out 3-dim demographics).
2. Bidirectional S4D: forward + backward processing, features concatenated.
3. S4D diagonal (more stable training than full S4).
4. Huber loss (called from train.py): less sensitive to BP outliers than MSE.
5. Stem convolution: channels expanded before S4D layers.
6. One model per BP target (SBP or DBP), set via `target` argument.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .s4_layer import S4DBlock


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.
    Demographics (3-dim) predict per-channel scale (gamma) and shift (beta)
    applied to PPG features. Residual formulation: output = (1+gamma)*feat + beta
    keeps training stable (gamma starts near zero -> identity transform).
    """

    def __init__(self, demo_dim: int, feature_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(demo_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, feature_dim * 2),
        )
        # Zero-init so model starts as identity at step 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, demo: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # demo: (B, demo_dim)
        # feat: (B, feature_dim)
        params = self.net(demo)  # (B, 2*feature_dim)
        D = feat.shape[-1]
        gamma, beta = params[:, :D], params[:, D:]  # (B, D) each
        return (1.0 + gamma) * feat + beta


class MS4Plus(nn.Module):
    """
    Improved MS4 model for blood pressure estimation from PPG + demographics.

    Architecture:
        PPG (B, L) -> normalize -> stem -> S4D forward path
                                        -> S4D backward path
                                  -> concat(fwd, bwd) -> mean pool -> (B, 2*d_model)
                                  -> FiLM(demo) -> head -> scalar BP prediction

    Args:
        target:   'SBP' or 'DBP' — which blood pressure component to predict.
        d_model:  S4D channel width (bidirectional doubles this at the pool step).
        n_layers: Number of S4DBlock layers per direction.
        d_state:  S4D state dimension (must be even; N/2 complex pairs used).
        dropout:  Dropout probability in S4D blocks and prediction head.
        demo_dim: Demographic input dimension (default 3: age, sex, BMI normalized).
    """

    def __init__(
        self,
        target: str = 'SBP',
        d_model: int = 128,
        n_layers: int = 6,
        d_state: int = 32,
        dropout: float = 0.1,
        demo_dim: int = 3,
    ):
        super().__init__()
        assert target in ('SBP', 'DBP'), "target must be 'SBP' or 'DBP'"
        self.target = target

        # --- Stem: project 1 input channel to d_model ---
        self.stem = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
        )

        # --- Bidirectional S4D backbone ---
        self.s4_forward = nn.ModuleList([
            S4DBlock(d_model, d_state=d_state, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.s4_backward = nn.ModuleList([
            S4DBlock(d_model, d_state=d_state, dropout=dropout)
            for _ in range(n_layers)
        ])

        pool_dim = 2 * d_model  # forward + backward concatenated

        # --- FiLM: demographics modulate pooled PPG features ---
        self.film = FiLM(demo_dim, pool_dim, hidden=64)

        # --- Prediction head ---
        self.head = nn.Sequential(
            nn.LayerNorm(pool_dim),
            nn.Linear(pool_dim, pool_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pool_dim // 2, 1),
        )

    @staticmethod
    def _normalize_ppg(ppg: torch.Tensor) -> torch.Tensor:
        """Per-sample z-score normalization."""
        mu = ppg.mean(dim=-1, keepdim=True)
        sigma = ppg.std(dim=-1, keepdim=True) + 1e-8
        return (ppg - mu) / sigma

    def forward(self, ppg: torch.Tensor, demo: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ppg:  (B, L)  raw PPG segment
            demo: (B, 3)  [age/100, sex (0/1), BMI/40]
        Returns:
            pred: (B,)   predicted SBP or DBP in mmHg
        """
        # Normalize PPG per sample
        ppg = self._normalize_ppg(ppg)

        # Stem: (B, L) -> (B, L, d_model)
        x = ppg.unsqueeze(-1)   # (B, L, 1)
        x = self.stem(x)        # (B, L, d_model)

        # Forward pass
        xf = x
        for blk in self.s4_forward:
            xf = blk(xf)

        # Backward pass (flip sequence, process, flip back)
        xb = x.flip(1)
        for blk in self.s4_backward:
            xb = blk(xb)
        xb = xb.flip(1)

        # Concatenate and pool: (B, L, 2*d_model) -> (B, 2*d_model)
        feat = torch.cat([xf, xb], dim=-1).mean(dim=1)

        # FiLM conditioning: demographics modulate PPG features
        feat = self.film(demo, feat)

        # Predict
        out = self.head(feat).squeeze(-1)  # (B,)
        return out
