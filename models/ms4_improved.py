"""
MS4Plus: Improved S4-based model for cuffless blood pressure estimation.

Drop-in replacement for MS4_1D in the existing PulseDB codebase.
Interface is identical to the existing models:
    forward(ppg, static_feat)
    ppg         : (B, 1, L)  — PPG signal with channel dim (as loaded by Build_Dataset)
    static_feat : (B, 3)     — raw [Age (years), BMI, Gender (0/1)]

Key improvements over baseline MS4_1D:
1. FiLM conditioning: demographics predict per-channel scale+shift of S4
   features instead of simple concatenation.  This is the core fix — the
   baseline's 512-dim S4 output drowns out 3-dim demographics via concat,
   which is why adding demographics made MS4 *worse* in the paper (7.10 vs
   6.83 MAE).  FiLM lets demographics modulate the entire feature space.
2. Bidirectional S4D: forward + backward pass, features concatenated before
   pooling.  Captures both causal and anti-causal PPG patterns.
3. S4D diagonal (no CUDA extension needed, more stable than full S4).
4. Per-sample PPG normalisation inside the model.
5. Internal demographic normalisation (age/100, bmi/40) so raw values from
   the existing data loader work without any external changes.

To use with the existing model_training_bootstrap.py, add to get_model_instance:
    if n in ["ms4plus", "ms4_plus", "ms4improved"]:
        from Model_Def.MS4Plus import MS4Plus_1D
        return MS4Plus_1D(num_static_features=num_static_features, num_BP=num_BP)
"""
import torch
import torch.nn as nn

from .s4_layer import S4DBlock


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation for demographic conditioning.

    Demographics -> gamma, beta -> (1 + gamma) * ppg_feat + beta

    Residual formulation (1 + gamma) ensures the model starts as an identity
    transform (gamma=0, beta=0 at init) for stable early training.
    """

    def __init__(self, demo_dim: int, feature_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(demo_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, feature_dim * 2),   # gamma + beta
        )
        # Zero-init output so model starts as identity at step 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, demo: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        params = self.net(demo)          # (B, 2*D)
        D = feat.shape[-1]
        gamma, beta = params[:, :D], params[:, D:]
        return (1.0 + gamma) * feat + beta


class MS4Plus_1D(nn.Module):
    """
    Improved MS4 with FiLM conditioning and bidirectional S4D.

    Args:
        num_static_features: number of demographic features (default 3).
        num_BP:              output size — 1 for single-target (SBP or DBP).
        d_model:             S4D channel width; bidirectional -> 2*d_model at pool.
        n_layers:            S4DBlock layers per direction.
        d_state:             S4D state dimension (must be even).
        dropout:             dropout in S4D blocks and head.
    """

    def __init__(
        self,
        num_static_features: int = 3,
        num_BP: int = 1,
        d_model: int = 128,
        n_layers: int = 6,
        d_state: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Stem: Conv1d matches the transposed (B, 1, L) input format used by
        # the existing S4Model encoder
        self.stem = nn.Conv1d(1, d_model, kernel_size=1, bias=False)

        # Bidirectional S4D backbone
        # S4DBlock expects (B, L, d_model); we transpose before/after
        self.s4_forward = nn.ModuleList([
            S4DBlock(d_model, d_state=d_state, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.s4_backward = nn.ModuleList([
            S4DBlock(d_model, d_state=d_state, dropout=dropout)
            for _ in range(n_layers)
        ])

        pool_dim = 2 * d_model   # fwd + bwd concatenated

        # Demographics MLP for FiLM
        # Internally normalises raw demographics (age/100, bmi/40) so the
        # existing data loader does not need to be changed.
        self.film = FiLM(num_static_features, pool_dim, hidden=64)

        # Prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(pool_dim),
            nn.Linear(pool_dim, pool_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pool_dim // 2, num_BP),
        )

    @staticmethod
    def _norm_ppg(x: torch.Tensor) -> torch.Tensor:
        """Per-sample z-score: (B, 1, L) -> (B, 1, L)."""
        mu    = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1,  keepdim=True) + 1e-8
        return (x - mu) / sigma

    @staticmethod
    def _norm_demo(static_feat: torch.Tensor) -> torch.Tensor:
        """
        Normalise raw demographics so all features are ~[0, 1].
        Assumes column order: [Age (years), BMI (kg/m²), Gender (0/1)]
        matching the existing Build_Dataset / Dataset.__getitem__.
        """
        age    = static_feat[:, 0:1] / 100.0
        bmi    = static_feat[:, 1:2] / 40.0
        gender = static_feat[:, 2:3]          # already 0/1
        return torch.cat([age, bmi, gender], dim=1)

    def forward(self, ppg: torch.Tensor, static_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ppg         : (B, 1, L)  raw PPG (channel dim included)
            static_feat : (B, 3)     raw [Age, BMI, Gender]
        Returns:
            (B,) or (B, num_BP) predicted BP in mmHg
        """
        # 1. Normalise inputs
        ppg         = self._norm_ppg(ppg)                    # (B, 1, L)
        static_feat = self._norm_demo(static_feat)           # (B, 3)

        # 2. Stem: (B, 1, L) -> (B, d_model, L) -> (B, L, d_model)
        x = self.stem(ppg).transpose(1, 2)                   # (B, L, d_model)

        # 3. Forward S4D pass
        xf = x
        for blk in self.s4_forward:
            xf = blk(xf)                                     # (B, L, d_model)

        # 4. Backward S4D pass (flip time, process, flip back)
        xb = x.flip(1)
        for blk in self.s4_backward:
            xb = blk(xb)
        xb = xb.flip(1)                                      # (B, L, d_model)

        # 5. Concatenate directions and global average pool
        feat = torch.cat([xf, xb], dim=-1).mean(dim=1)      # (B, 2*d_model)

        # 6. FiLM: demographics modulate PPG features
        feat = self.film(static_feat, feat)                  # (B, 2*d_model)

        # 7. Predict
        out = self.head(feat)                                 # (B, num_BP)
        return out.squeeze(-1)                               # (B,) when num_BP=1
