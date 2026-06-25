import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Common ----------
class PatchEmbedding1D(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, emb_dim: int, seq_len: int):
        super().__init__()
        assert seq_len % patch_size == 0, "seq_len must be divisible by patch_size."
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.proj = nn.Conv1d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):  # x: (B, C=1, L)
        # -> (B, emb_dim, n_patches) -> (B, n_patches, emb_dim)
        return self.proj(x).transpose(1, 2)


class DemographicEncoder(nn.Module):
    """Encode (age, gender, BMI) -> emb_dim vector"""
    def __init__(self, in_dim: int, emb_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, static_feat):  # (B, 3)
        return self.net(static_feat)  # (B, emb_dim)


# ---------- FiLM Version ----------
class TransformerEncoder1D_FiLM(nn.Module):
    """Pre-LN Transformer block with FiLM on both Attn and MLP residual paths."""
    def __init__(self, emb_dim: int, nhead: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=nhead, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(emb_dim)
        hidden = int(emb_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, film1_gamma, film1_beta, film2_gamma, film2_beta):
        # x: (B, T, C) ; film*_gamma/beta: (B, 1, C)
        x1 = self.norm1(x)
        x1 = x1 * (1.0 + film1_gamma) + film1_beta
        attn_out, _ = self.attn(x1, x1, x1, need_weights=False)
        x = x + attn_out

        x2 = self.norm2(x)
        x2 = x2 * (1.0 + film2_gamma) + film2_beta
        x = x + self.mlp(x2)
        return x


class FiLMGenerator(nn.Module):
    """Generate per-layer FiLM params from demographic vector."""
    def __init__(self, demog_emb_dim: int, emb_dim: int, depth: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.depth = depth
        out_dim = depth * 4 * emb_dim  # (gamma1, beta1, gamma2, beta2) for each layer
        self.net = nn.Sequential(
            nn.Linear(demog_emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, demog_vec):  # (B, demog_emb_dim)
        B = demog_vec.size(0)
        film = self.net(demog_vec)  # (B, depth*4*emb_dim)
        return film.view(B, self.depth, 4, -1)  # (B, depth, 4, emb_dim)


class ViT1D_FiLM(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 seq_len: int = 1250,
                 patch_size: int = 10,
                 emb_dim: int = 128,
                 depth: int = 6,
                 nhead: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 demog_dim: int = 3,
                 num_BP: int = 1,
                 num_shape_classes: int = 2):
        super().__init__()
        # patches
        self.patch_embed = PatchEmbedding1D(in_channels, patch_size, emb_dim, seq_len)
        self.n_patches = seq_len // patch_size

        # positional embedding (learned)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, emb_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # demographic pathway
        self.demog_enc = DemographicEncoder(demog_dim, emb_dim, hidden=128, dropout=dropout)
        self.film_gen = FiLMGenerator(emb_dim, emb_dim, depth, hidden=128, dropout=dropout)

        # transformer encoder
        self.encoder = nn.ModuleList([
            TransformerEncoder1D_FiLM(emb_dim, nhead, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)

        # heads
        self.bp_head = nn.Linear(emb_dim, num_BP)
        self.shape_head = nn.Linear(emb_dim, num_shape_classes)

    def forward(self, ppg, static_feat):
        """
        ppg: (B, 1, L) ; static_feat: (B, 3)
        """
        B = ppg.size(0)
        tokens = self.patch_embed(ppg)  # (B, T, C)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]

        demog_vec = self.demog_enc(static_feat)        # (B, C)
        film_all = self.film_gen(demog_vec)            # (B, depth, 4, C)

        for i, blk in enumerate(self.encoder):
            # (B, 1, C) for broadcasting along time dimension
            gamma1 = film_all[:, i, 0].unsqueeze(1)
            beta1  = film_all[:, i, 1].unsqueeze(1)
            gamma2 = film_all[:, i, 2].unsqueeze(1)
            beta2  = film_all[:, i, 3].unsqueeze(1)
            tokens = blk(tokens, gamma1, beta1, gamma2, beta2)

        tokens = self.norm(tokens)
        global_feat = tokens.mean(dim=1)  # (B, C)

        bp_pred = self.bp_head(global_feat).squeeze(-1)   # (B,)
        shape_pred = self.shape_head(global_feat)         # (B, 2)
        return bp_pred, shape_pred