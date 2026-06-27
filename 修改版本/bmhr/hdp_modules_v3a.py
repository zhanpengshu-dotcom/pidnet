import torch
import torch.nn as nn
import torch.nn.functional as F


BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, act=True):
        super(ConvBNAct, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      dilation=dilation, groups=groups, bias=False),
            BatchNorm2d(out_channels, momentum=bn_mom),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class FixedLaplacian(nn.Module):
    def __init__(self, channels):
        super(FixedLaplacian, self).__init__()
        kernel = torch.tensor(
            [[0.0, -1.0, 0.0],
             [-1.0, 4.0, -1.0],
             [0.0, -1.0, 0.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer('weight', kernel.repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=1, groups=self.channels)


class BMHR(nn.Module):
    """Boundary-guided multi-scale high-frequency refinement.

    The module keeps the input/output shape unchanged and starts as an identity
    mapping when zero_init=True.
    """

    def __init__(self, channels, reduction=4, use_boundary_gate=True,
                 zero_init=True):
        super(BMHR, self).__init__()
        mid = max(channels // reduction, 16)
        self.use_boundary_gate = use_boundary_gate

        self.reduce = ConvBNAct(channels, mid, kernel_size=1, act=True)
        self.local = ConvBNAct(mid, mid, kernel_size=3, padding=1,
                               groups=mid, act=True)
        self.medium = ConvBNAct(mid, mid, kernel_size=3, padding=2,
                                dilation=2, groups=mid, act=True)
        self.large = ConvBNAct(mid, mid, kernel_size=3, padding=3,
                               dilation=3, groups=mid, act=True)
        self.high_pass = FixedLaplacian(mid)
        self.high_freq = ConvBNAct(mid, mid, kernel_size=3, padding=1,
                                   groups=mid, act=True)

        self.gate_plain = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            BatchNorm2d(mid, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 4, kernel_size=1, bias=True),
        )
        if use_boundary_gate:
            self.gate_boundary = nn.Sequential(
                nn.Conv2d(channels * 2, mid, kernel_size=1, bias=False),
                BatchNorm2d(mid, momentum=bn_mom),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, 4, kernel_size=1, bias=True),
            )
        else:
            self.gate_boundary = None

        self.expand = ConvBNAct(mid, channels, kernel_size=1, act=False)

        if zero_init:
            nn.init.constant_(self.expand.block[1].weight, 0)
            nn.init.constant_(self.expand.block[1].bias, 0)

    def _gate(self, p, d):
        if self.use_boundary_gate and d is not None:
            gate_input = torch.cat([p, d], dim=1)
            gate_logits = self.gate_boundary(gate_input)
        else:
            gate_input = p
            gate_logits = self.gate_plain(gate_input)
        return torch.softmax(gate_logits, dim=1)

    def forward(self, p, d=None):
        res = self.reduce(p)

        b1 = self.local(res)
        b2 = self.medium(res)
        b3 = self.large(res)
        b4 = self.high_freq(self.high_pass(res))

        weights = self._gate(p, d)
        fused = (
            weights[:, 0:1] * b1
            + weights[:, 1:2] * b2
            + weights[:, 2:3] * b3
            + weights[:, 3:4] * b4
        )

        return p + self.expand(fused)
