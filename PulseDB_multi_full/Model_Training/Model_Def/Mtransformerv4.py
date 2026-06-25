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

    def forward(self, x):  # (B,1,L)
        return self.proj(x).transpose(1, 2)  # (B,T,C)


class DemographicEncoder(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, static_feat):
        return self.net(static_feat)  # (B, emb_dim)


# ---------- Conditional Attention ----------
class ConditionalSelfAttention(nn.Module):
    """
    Demographic-conditioned self-attention.
    - Learn a relative position bias (RPE) per head: (H, Lmax, Lmax)
    - Demographic vector -> per-head scaling alpha: (B, H)
    - Attention logits += alpha[...,None,None] * RPE[:, :L, :L]
    """
    def __init__(self, emb_dim: int, nhead: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        assert emb_dim % nhead == 0
        self.emb_dim = emb_dim
        self.nhead = nhead
        self.head_dim = emb_dim // nhead
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Relative position bias (shared across batch), sliced to current L
        self.max_len = max_len
        self.rel_pos_bias = nn.Parameter(torch.zeros(nhead, max_len, max_len))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

        # demographic -> per-head alpha
        self.alpha_gen = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.GELU(),
            nn.Linear(emb_dim // 2, nhead),
        )

    def forward(self, x, demog_vec):
        # x: (B,T,C), demog_vec: (B,C)
        B, T, C = x.shape
        H, D = self.nhead, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).permute(0, 2, 1, 3)  # (B,H,T,D)
        k = self.k_proj(x).view(B, T, H, D).permute(0, 2, 1, 3)  # (B,H,T,D)
        v = self.v_proj(x).view(B, T, H, D).permute(0, 2, 1, 3)  # (B,H,T,D)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,T,T)

        # conditional RPE
        alpha = self.alpha_gen(demog_vec).unsqueeze(-1).unsqueeze(-1)  # (B,H,1,1)
        rpb = self.rel_pos_bias[:, :T, :T].unsqueeze(0)               # (1,H,T,T)
        attn_logits = attn_logits + alpha * rpb

        attn = attn_logits.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B,H,T,D)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)  # (B,T,C)
        out = self.proj_drop(self.out_proj(out))  # (B,T,C)
        return out


class TransformerEncoder1D_Cond(nn.Module):
    """Pre-LN Transformer block; attention logits are conditioned by demographics."""
    def __init__(self, emb_dim: int, nhead: int, max_len: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = ConditionalSelfAttention(emb_dim, nhead, max_len, dropout)
        self.norm2 = nn.LayerNorm(emb_dim)
        hidden = int(emb_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, demog_vec):
        x = x + self.attn(self.norm1(x), demog_vec)
        x = x + self.mlp(self.norm2(x))
        return x


class ViT1D_ConditionalAttn(nn.Module):
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
        self.patch_embed = PatchEmbedding1D(in_channels, patch_size, emb_dim, seq_len)
        self.n_patches = seq_len // patch_size

        # learned positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, emb_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # demographic encoder produces conditioning vector
        self.demog_enc = DemographicEncoder(demog_dim, emb_dim, hidden=128, dropout=dropout)

        self.encoder = nn.ModuleList([
            TransformerEncoder1D_Cond(emb_dim, nhead, max_len=self.n_patches, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)

        self.bp_head = nn.Linear(emb_dim, num_BP)
        self.shape_head = nn.Linear(emb_dim, num_shape_classes)

    def forward(self, ppg, static_feat):
        """
        ppg: (B,1,L) ; static_feat: (B,3)
        """
        tokens = self.patch_embed(ppg)  # (B,T,C)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]

        demog_vec = self.demog_enc(static_feat)  # (B,C)

        for blk in self.encoder:
            tokens = blk(tokens, demog_vec)

        tokens = self.norm(tokens)
        global_feat = tokens.mean(dim=1)

        bp_pred = self.bp_head(global_feat).squeeze(-1)
        shape_pred = self.shape_head(global_feat)
        return bp_pred, shape_pred
