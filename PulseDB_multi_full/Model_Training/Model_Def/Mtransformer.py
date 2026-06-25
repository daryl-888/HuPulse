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
                 num_static_features=3):
        super().__init__()
        self.patch_embed = PatchEmbedding1D(in_channels, patch_size, emb_dim, seq_len)
        n_patches = seq_len // patch_size

        # ???? embedding,??????Linear
        self.static_embed_layers = nn.ModuleList([
            nn.Linear(1, emb_dim) for _ in range(num_static_features)
        ])
        self.num_static_features = num_static_features

        n_tokens = n_patches + num_static_features
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, emb_dim))

        self.pos_drop = nn.Dropout(p=dropout)
        self.encoder = nn.ModuleList([
            TransformerEncoder1D(emb_dim, nhead, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_BP)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, ppg, static_feat):
        # ppg: [B, 1, L], static_feat: [B, 3]
        patch_tokens = self.patch_embed(ppg)   # [B, n_patches, emb_dim]

        # ??????embedding???token
        static_tokens = []
        for i in range(self.num_static_features):
            feat = static_feat[:, i].unsqueeze(-1)     # [B, 1]
            static_tokens.append(self.static_embed_layers[i](feat))  # [B, emb_dim]
        static_tokens = torch.stack(static_tokens, dim=1)   # [B, n_static, emb_dim]

        # ????:????token???
        tokens = torch.cat([static_tokens, patch_tokens], dim=1)   # [B, n_static+n_patches, emb_dim]

        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]
        tokens = self.pos_drop(tokens)

        for blk in self.encoder:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        # ?mean pooling
        out = tokens.mean(dim=1)     # [B, emb_dim]
        out = self.head(out)         # [B, num_BP]
        print(f'out.shape:{out.shape}')
        return out.squeeze()

def ViT1D_Fusion_Model(**kwargs):
    return ViT1D_Fusion(**kwargs)