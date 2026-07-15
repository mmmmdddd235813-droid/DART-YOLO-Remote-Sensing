"""CSD: Channel-Split Downsampling module used by DART-YOLO.

The module combines a learnable depthwise-separable downsampling branch with
a max-pooling saliency branch, then fuses both branches with a 1x1 convolution.
It is named CSD in the paper and in `DART-YOLO.yaml`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["CSD", "SDZ", "SPD"]


class _DepthwiseContextBranch(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CSD(nn.Module):
    """Channel-Split Downsampling.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        *unused: Extra positional arguments are accepted for compatibility
            with Ultralytics YAML parser variants that pass kernel/stride fields.
    """

    def __init__(self, in_channels: int, out_channels: int, *unused):
        super().__init__()
        if in_channels < 2:
            raise ValueError("CSD requires at least two input channels for channel splitting.")

        branch1_channels = in_channels // 2
        branch2_channels = in_channels - branch1_channels
        out1_channels = out_channels // 2
        out2_channels = out_channels - out1_channels

        self.context_branch = _DepthwiseContextBranch(branch1_channels, out1_channels)
        self.saliency_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(branch2_channels, out2_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out2_channels),
            nn.ReLU6(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, [x.shape[1] // 2, x.shape[1] - x.shape[1] // 2], dim=1)
        return self.fuse(torch.cat((self.context_branch(x1), self.saliency_branch(x2)), dim=1))


# Backward-compatible aliases for older code and model YAML files.
SDZ = CSD
SPD = CSD
