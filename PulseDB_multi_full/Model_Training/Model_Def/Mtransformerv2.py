import torch
import torch.nn as nn

class PatchEmbedding1D(nn.Module):
    def __init__(self, in_channels, patch_size, emb_dim, seq_len):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.proj = nn.Conv1d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x).transpose(1, 2)
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

class ViT1D_Fusion(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 seq_len=1250, 
                 patch_size=10, 
                 emb_dim=128, 
                 depth=6, 
                 nhead=8, 
                 mlp_ratio=4.0,
                 num_BP=1,
                 dropout=0.1,
                 num_static_features=3,
                 num_shape_classes=2):  # ?? or ??
        super().__init__()
        self.patch_embed = PatchEmbedding1D(in_channels, patch_size, emb_dim, seq_len)
        n_patches = seq_len // patch_size

        self.static_embed_layers = nn.ModuleList([
            nn.Linear(1, emb_dim) for _ in range(num_static_features)
        ])

        self.num_static_features = num_static_features
        n_tokens = n_patches + num_static_features

        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, emb_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=dropout)
        self.encoder = nn.ModuleList([
            TransformerEncoder1D(emb_dim, nhead, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)

        # ???:????
        self.bp_head = nn.Linear(emb_dim, num_BP)

        # ????:??????(0:??,1:??)
        self.shape_head = nn.Linear(emb_dim, num_shape_classes)

    def forward(self, ppg, static_feat):
        patch_tokens = self.patch_embed(ppg)

        static_tokens = torch.stack([
            self.static_embed_layers[i](static_feat[:, i:i+1])
            for i in range(self.num_static_features)
        ], dim=1)

        tokens = torch.cat([static_tokens, patch_tokens], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]
        tokens = self.pos_drop(tokens)

        for blk in self.encoder:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        global_feat = tokens.mean(dim=1)

        # ?????
        bp_pred = self.bp_head(global_feat).squeeze()

        # ??????(????)
        shape_pred = self.shape_head(global_feat)

        return bp_pred, shape_pred