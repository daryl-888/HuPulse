"""
S4D: Diagonal State Space Layer.

Simplified S4 using diagonal A matrix (S4D-Lin initialization).
Reference: "On the Parameterization and Initialization of Diagonal State
Space Models" (Gu et al., 2022).

The key benefit over full S4: no CUDA extension needed, trains stably,
same O(L log L) inference via FFT convolution.

Compatible with PyTorch >= 1.7 (tested on 1.10.1+cu113).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class S4DKernel(nn.Module):
    """
    Computes the S4D convolution kernel K of length L.

    State-space system: h'(t) = Ah(t) + Bu(t), y(t) = Ch(t) + Du(t)
    With A diagonal (complex): A_n = -1/2 + j*pi*n  (S4D-Lin)
    """

    def __init__(self, d_model: int, d_state: int = 32,
                 dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        assert d_state % 2 == 0, "d_state must be even (complex conjugate pairs)"
        H = d_model
        N = d_state // 2  # number of complex conjugate pairs

        # Log timescale, one per channel, initialized uniformly in [dt_min, dt_max]
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # S4D-Lin: A_n = -1/2 + j*pi*n, shared across channels (columns fixed)
        # Real parts are learned; imaginary parts start as pi*n and are frozen
        # (following the paper's recommendation for stable training)
        A_re = -0.5 * torch.ones(H, N)
        A_im = math.pi * torch.arange(N).float().unsqueeze(0).expand(H, -1)
        self.A_re = nn.Parameter(A_re)
        self.A_im = nn.Parameter(A_im)

        # B: complex input projection, (H, N) stored as real (H, N, 2)
        self.B = nn.Parameter(torch.randn(H, N, 2) / math.sqrt(N))

        # C: complex output projection, (H, N) stored as real (H, N, 2)
        self.C = nn.Parameter(torch.randn(H, N, 2) / math.sqrt(N))

    def forward(self, L: int) -> torch.Tensor:
        """
        Return real convolution kernel K of shape (H, L).
        Called once per forward pass of S4DLayer.
        """
        dt = torch.exp(self.log_dt)  # (H,)
        A = torch.complex(self.A_re, self.A_im)  # (H, N)
        B = torch.view_as_complex(self.B.contiguous())  # (H, N)
        C = torch.view_as_complex(self.C.contiguous())  # (H, N)

        # ZOH discretization
        dtA = dt.unsqueeze(-1) * A  # (H, N)
        A_bar = torch.exp(dtA)  # (H, N) complex, |A_bar| < 1 for stable Re(A) < 0

        # B_bar = (A_bar - 1) / A * B  (Euler step; safe when Re(A) != 0)
        B_bar = (A_bar - 1.0) / A * B  # (H, N) complex

        # Kernel K[l] = 2 * Re(sum_n C_n * B_bar_n * A_bar_n^l) for l=0..L-1
        # Compute via log to avoid overflow: A_bar_n^l = exp(l * log(A_bar_n))
        log_A_bar = torch.log(A_bar)  # (H, N) complex
        CB = C * B_bar  # (H, N)

        # l: (L,); broadcast with (H, N, 1) * (1, 1, L) -> (H, N, L)
        l = torch.arange(L, device=log_A_bar.device, dtype=torch.float32)
        exponent = log_A_bar.unsqueeze(-1) * l.view(1, 1, -1)  # (H, N, L)
        K = 2.0 * (CB.unsqueeze(-1) * torch.exp(exponent)).real.sum(dim=1)  # (H, L)

        return K  # (H, L) real


class S4DLayer(nn.Module):
    """
    Full S4D layer: computes output y = K * u + D*u via FFT convolution.

    Input/output: (B, L, H)  [batch, sequence, channels]
    """

    def __init__(self, d_model: int, d_state: int = 32,
                 dropout: float = 0.0,
                 dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.kernel = S4DKernel(d_model, d_state, dt_min, dt_max)
        self.D = nn.Parameter(torch.ones(d_model))  # skip connection
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Output mix: channels are independent in S4D, so add a pointwise mix
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (B, L, H)
        B, L, H = u.shape
        u_t = u.transpose(-1, -2)  # (B, H, L)

        # Convolution kernel
        K = self.kernel(L)  # (H, L)

        # FFT convolution (causal via zero-padding to 2L)
        fft_len = 2 * L
        K_f = torch.fft.rfft(K, n=fft_len)           # (H, fft_len//2+1)
        u_f = torch.fft.rfft(u_t, n=fft_len)          # (B, H, fft_len//2+1)
        y_t = torch.fft.irfft(K_f * u_f, n=fft_len)[..., :L]  # (B, H, L)

        # Skip connection
        y_t = y_t + self.D.view(1, H, 1) * u_t  # (B, H, L)
        y = y_t.transpose(-1, -2)  # (B, L, H)

        y = self.drop(y)
        return self.out_proj(y)


class S4DBlock(nn.Module):
    """
    Pre-norm residual block: LayerNorm -> S4DLayer -> LayerNorm -> FFN.
    Standard in sequence model literature (e.g., S4 repo's default).
    """

    def __init__(self, d_model: int, d_state: int = 32,
                 d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 2 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.s4d = S4DLayer(d_model, d_state=d_state, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.s4d(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
