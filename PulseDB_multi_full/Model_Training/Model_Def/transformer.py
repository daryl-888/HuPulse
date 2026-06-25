import torch
import torch.nn as nn

class PatchEmbedding1D(nn.Module):
    def __init__(self, in_channels, patch_size, emb_dim, seq_len):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.proj = nn.Conv1d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: [B, C, L]
        x = self.proj(x)  # [B, emb_dim, n_patches]
        x = x.transpose(1, 2)  # [B, n_patches, emb_dim]
        return x

class TransformerEncoder1D(nn.Module):
    def __init__(self, emb_dim, nhead, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, int(emb_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(emb_dim * mlp_ratio), emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ViT1D(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 seq_len=1250, 
                 patch_size=10, 
                 emb_dim=128, 
                 depth=6, 
                 nhead=8, 
                 mlp_ratio=4.0,
                 num_BP=1,
                 use_cls_token=False,
                 dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding1D(in_channels, patch_size, emb_dim, seq_len)
        n_patches = seq_len // patch_size

        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, emb_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, emb_dim))

        self.pos_drop = nn.Dropout(p=dropout)
        self.encoder = nn.ModuleList([
            TransformerEncoder1D(emb_dim, nhead, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)

        self.head = nn.Linear(emb_dim, num_BP)

        # ???
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if use_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: [B, C, L]
        x = self.patch_embed(x)  # [B, n_patches, emb_dim]

        if self.use_cls_token:
            B = x.shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, emb_dim]
            x = torch.cat((cls_tokens, x), dim=1)  # [B, n_patches+1, emb_dim]
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.pos_drop(x)

        for blk in self.encoder:
            x = blk(x)
        x = self.norm(x)

        if self.use_cls_token:
            out = x[:, 0]  # [B, emb_dim]
        else:
            out = x.mean(dim=1)  # mean pooling

        out = self.head(out)  # [B, num_BP]
        return out

def ViT_1D(**kwargs):
    return ViT1D(**kwargs)
