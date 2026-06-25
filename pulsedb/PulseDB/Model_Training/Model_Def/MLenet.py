import torch
import torch.nn as nn
from enum import Enum

NormType = Enum('NormType', 'Batch BatchZero')

def BatchNorm(nf, norm_type=NormType.Batch, **kwargs):
    return _get_norm('BatchNorm', nf, zero=norm_type==NormType.BatchZero, **kwargs)

def _get_norm(prefix, nf, zero=False, **kwargs):
    bn = getattr(nn, f"{prefix}1d")(nf, **kwargs)
    if bn.affine:
        bn.bias.data.fill_(1e-3)
        bn.weight.data.fill_(0. if zero else 1.)
    return bn

def init_default(m, func=nn.init.kaiming_normal_):
    if func and hasattr(m, 'weight'): func(m.weight)
    with torch.no_grad():
        if getattr(m, 'bias', None) is not None: m.bias.fill_(0.)
    return m

class LeNet1d(nn.Module):
    def __init__(self, input_channels, num_classes, bn=True, ps=False, hidden=False, 
                 c1=12, c2=32, c3=64, c4=128, k=5, s=2, E=128, flatten_dim=4736):
        super().__init__()
        self.bn = bn
        self.ps = ps
        self.hidden = hidden
        modules = []
        modules.append(self.conv_block(input_channels, c2, k, s, bn, ps))
        modules.append(self.conv_block(c2, c3, k, s, bn, ps))
        modules.append(self.conv_block(c3, c4, k, s, bn, ps, final=True))
        self.conv_layers = nn.Sequential(*modules)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flatten_dim, E)
        self.fc2 = nn.Linear(E, num_classes)
        if hidden:
            self.hidden_layers = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(E, E),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
        else:
            self.hidden_layers = None

    def conv_block(self, in_channels, out_channels, kernel_size, stride, bn, ps, final=False):
        layers = [init_default(nn.Conv1d(in_channels, out_channels, kernel_size, stride))]
        if bn:
            layers.append(BatchNorm(out_channels, norm_type=NormType.Batch))
        layers.append(nn.ReLU())
        if not final:
            layers.append(nn.MaxPool1d(2))
        if ps:
            layers.append(nn.Dropout(0.1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        if self.hidden_layers:
            x = self.hidden_layers(x)
        x = self.fc2(x)
        return x

    def freeze_backbone(self):
        for param in self.conv_layers.parameters():
            param.requires_grad = False
        for param in self.fc1.parameters():
            param.requires_grad = True
        for param in self.fc2.parameters():
            param.requires_grad = True

# --- Multi-modal LeNet1d???? ---
class MLeNet1d(nn.Module):
    def __init__(self, 
                 num_static_features=3, 
                 num_BP=1, 
                 static_hidden=16, 
                 fusion_hidden=32,
                 ppg_input_channels=1,
                 lenet_flatten_dim=4736,   # ??:????LeNet1d flatten????
                 lenet_hidden_dim=128,
                 lenet_kwargs=None,
    ):
        super(MLeNet1d, self).__init__()
        if lenet_kwargs is None: lenet_kwargs = {}
        self.lenet1d = LeNet1d(
            input_channels=ppg_input_channels, 
            num_classes=lenet_hidden_dim, 
            flatten_dim=lenet_flatten_dim,
            **lenet_kwargs
        )
        self.static_mlp = nn.Sequential(
            nn.Linear(num_static_features, static_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(static_hidden, static_hidden),
            nn.ReLU(inplace=True)
        )
        self.fusion_fc = nn.Sequential(
            nn.Linear(lenet_hidden_dim + static_hidden, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden, num_BP)
        )

    def forward(self, ppg, static_feat):
        # ppg: (batch, 1, seq_len)
        ppg_feat = self.lenet1d(ppg)            # (batch, lenet_hidden_dim)
        static_feat = self.static_mlp(static_feat)  # (batch, static_hidden)
        fusion = torch.cat([ppg_feat, static_feat], dim=1)  # (batch, lenet_hidden_dim+static_hidden)
        out = self.fusion_fc(fusion)
        return out.squeeze()
