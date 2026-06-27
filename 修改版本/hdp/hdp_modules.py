# ------------------------------------------------------------------------------
# HDP-PIDNet lightweight modules.
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, groups=1, act=True):
        super(ConvBNAct, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=groups, bias=False),
            BatchNorm2d(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=True) if act else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class DWConvBNAct(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, padding=0, act=True):
        super(DWConvBNAct, self).__init__()
        self.block = ConvBNAct(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=channels,
            act=act,
        )

    def forward(self, x):
        return self.block(x)


class DGA_PAPPM(nn.Module):
    """Directional gated aggregation variant of PAPPM.

    Keeps the PAPPM-style parallel pyramid pooling interface while adding a
    lightweight directional branch and branch-level gating.
    """

    def __init__(self, inplanes, branch_planes, outplanes,
                 dir_kernel=7, use_gate=True):
        super(DGA_PAPPM, self).__init__()
        if dir_kernel % 2 == 0:
            raise ValueError('dir_kernel must be odd to preserve spatial size')

        self.use_gate = use_gate
        self.num_branches = 6
        dir_pad = dir_kernel // 2

        self.scale0 = ConvBNAct(inplanes, branch_planes, kernel_size=1)

        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            ConvBNAct(inplanes, branch_planes, kernel_size=1),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            ConvBNAct(inplanes, branch_planes, kernel_size=1),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            ConvBNAct(inplanes, branch_planes, kernel_size=1),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBNAct(inplanes, branch_planes, kernel_size=1),
        )

        self.direction = nn.Sequential(
            ConvBNAct(inplanes, branch_planes, kernel_size=1),
            DWConvBNAct(branch_planes, kernel_size=(1, dir_kernel),
                        padding=(0, dir_pad)),
            DWConvBNAct(branch_planes, kernel_size=(dir_kernel, 1),
                        padding=(dir_pad, 0)),
        )

        if use_gate:
            gate_mid = max(inplanes // 16, 16)
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(inplanes, gate_mid, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(gate_mid, self.num_branches, kernel_size=1, bias=True),
            )
        else:
            self.gate = None

        self.fusion = ConvBNAct(branch_planes, outplanes, kernel_size=1, act=False)
        self.shortcut = ConvBNAct(inplanes, outplanes, kernel_size=1, act=False)

    def forward(self, x):
        height = x.shape[-2]
        width = x.shape[-1]

        local = self.scale0(x)
        scale1 = F.interpolate(self.scale1(x), size=[height, width],
                               mode='bilinear', align_corners=algc) + local
        scale2 = F.interpolate(self.scale2(x), size=[height, width],
                               mode='bilinear', align_corners=algc) + local
        scale3 = F.interpolate(self.scale3(x), size=[height, width],
                               mode='bilinear', align_corners=algc) + local
        global_context = F.interpolate(self.scale4(x), size=[height, width],
                                       mode='bilinear', align_corners=algc) + local
        directional = self.direction(x)

        branches = torch.stack(
            [local, scale1, scale2, scale3, global_context, directional],
            dim=1,
        )

        if self.gate is not None:
            weights = torch.softmax(self.gate(x), dim=1).unsqueeze(2)
            fused = (branches * weights).sum(dim=1)
        else:
            fused = branches.mean(dim=1)

        return self.fusion(fused) + self.shortcut(x)
