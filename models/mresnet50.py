"""
MResNet50: 1D XResNet-50 with late demographic fusion.

Matches the paper's MResNet50 architecture:
  - 1D XResNet-50 backbone (stem expanded to 64 channels)
  - Demographics: 2-layer MLP (16 hidden units)
  - Late fusion: concat -> 32-dim -> 1 output
  - One model per BP target (SBP or DBP)

Paper cal-based results: MAE 5.35/3.24 mmHg (SBP/DBP)
"""
import torch
import torch.nn as nn


# ── Bottleneck block (1D) ─────────────────────────────────────────────────────

class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_ch: int, mid_ch: int, stride: int = 1,
                 downsample=None):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv1d(in_ch, mid_ch, 1, bias=False)
        self.bn1   = nn.BatchNorm1d(mid_ch)
        self.conv2 = nn.Conv1d(mid_ch, mid_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(mid_ch)
        self.conv3 = nn.Conv1d(mid_ch, out_ch, 1, bias=False)
        self.bn3   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


# ── XResNet-50 1D backbone ────────────────────────────────────────────────────

class XResNet1D50(nn.Module):
    """
    1D XResNet-50. XResNet replaces the 7×7 stem conv with three 3-wide convs.
    Initial conv expanded to 64 output channels as per the paper.
    """

    def __init__(self):
        super().__init__()
        # XResNet stem: three 1D convs instead of one 7×7
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        # Residual stages: [3, 4, 6, 3] blocks (same as ResNet-50)
        # mid_ch × 4 = out_ch: 64→256, 128→512, 256→1024, 512→2048
        self.layer1 = self._make_layer(64,  64,  3, stride=1)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.out_dim = 2048

    def _make_layer(self, in_ch: int, mid_ch: int,
                    n_blocks: int, stride: int) -> nn.Sequential:
        out_ch = mid_ch * Bottleneck1D.expansion
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        layers = [Bottleneck1D(in_ch, mid_ch, stride=stride, downsample=downsample)]
        for _ in range(1, n_blocks):
            layers.append(Bottleneck1D(out_ch, mid_ch))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)   # (B, 2048)
        return x


# ── Full MResNet50 with late fusion ───────────────────────────────────────────

class MResNet50(nn.Module):
    """
    MResNet50: XResNet-50 backbone + demographic late fusion.

    Matches paper exactly:
      PPG -> XResNet50 -> 2048-dim
      Demo (3) -> Linear(16) -> ReLU -> Linear(16) -> ReLU -> 16-dim
      Concat(2064) -> Linear(32) -> ReLU -> Linear(1)
    """

    def __init__(self, target: str = 'SBP', demo_dim: int = 3):
        super().__init__()
        assert target in ('SBP', 'DBP')
        self.target = target

        self.backbone = XResNet1D50()
        ppg_dim = self.backbone.out_dim  # 2048

        # Demographics MLP (paper: 2 layers, 16 hidden)
        self.demo_mlp = nn.Sequential(
            nn.Linear(demo_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True),
        )

        # Fusion layer (paper: 32 hidden neurons)
        fused_dim = ppg_dim + 16
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, ppg: torch.Tensor, demo: torch.Tensor) -> torch.Tensor:
        # ppg:  (B, L)
        # demo: (B, 3)  [age/100, sex, bmi/40]
        ppg_feat  = self.backbone(ppg.unsqueeze(1))   # (B, 2048)
        demo_feat = self.demo_mlp(demo)               # (B, 16)
        fused     = torch.cat([ppg_feat, demo_feat], dim=-1)
        return self.fusion(fused).squeeze(-1)          # (B,)
