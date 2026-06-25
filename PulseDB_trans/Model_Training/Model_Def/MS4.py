import torch
import torch.nn as nn

from .s42 import S4 as S42

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

        # Encoder
        if d_input is None:
            self.encoder = nn.Identity()
        else:
            self.encoder = nn.Conv1d(d_input, d_model, 1) if transposed_input else nn.Linear(d_input, d_model)

        # S4 layers
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

# --- Multi-modal S4???? ---
class MS4_1D(nn.Module):
    def __init__(self, 
                 num_static_features=3, 
                 num_BP=1, 
                 static_hidden=16, 
                 fusion_hidden=32, 
                 s4_d_input=1, 
                 s4_d_model=512, 
                 s4_n_layers=4, 
                 s4_pooling=True, 
                 s4_l_max=2048):
        super(MS4_1D, self).__init__()
        # S4??
        self.s4model = S4Model(
            d_input=s4_d_input,
            d_output=None,      # ??feature,?????
            d_state=64,
            d_model=s4_d_model,
            n_layers=s4_n_layers,
            pooling=s4_pooling,
            l_max=s4_l_max,
        )
        # ??MLP
        self.static_mlp = nn.Sequential(
            nn.Linear(num_static_features, static_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(static_hidden, static_hidden),
            nn.ReLU(inplace=True)
        )
        # ?????
        self.ppg_feat_mlp= nn.Sequential(
            nn.Linear(s4_d_model, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        
        self.fusion_fc = nn.Sequential(
            nn.Linear(128 + static_hidden, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden, num_BP)
        )

    def forward(self, ppg, static_feat):
        """
        ppg: (batch, 1, seq_len)
        static_feat: (batch, num_static_features)
        """
        ppg_feat = self.s4model(ppg)           # (batch, s4_d_model)
        static_feat = self.static_mlp(static_feat)  # (batch, static_hidden)
        processed_ppg_feat = self.ppg_feat_mlp(ppg_feat)
        fusion = torch.cat([processed_ppg_feat, static_feat], dim=1)  # (batch, s4_d_model+static_hidden)
        out = self.fusion_fc(fusion)
        return out.squeeze()

