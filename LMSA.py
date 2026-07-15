"""Local Multi-Scale Attention module for DART-YOLO."""
import torch
from torch import nn, Tensor
from typing import List, Optional

try:
    from mmengine.model import BaseModule, constant_init
except ModuleNotFoundError:
    class BaseModule(nn.Module):
        def init_weights(self):
            pass

    def constant_init(module, val, bias=0):
        nn.init.constant_(module.weight, val)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias, bias)

from ultralytics.nn.modules.block import Bottleneck, C2f, C3k

try:
    from mmcv.cnn import ConvModule
except ModuleNotFoundError:
    class ConvModule(nn.Sequential):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm_cfg=None, act_cfg=None):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=norm_cfg is None)]
            if norm_cfg is not None:
                layers.append(nn.BatchNorm2d(out_channels))
            if act_cfg is not None:
                act_type = act_cfg.get("type", "ReLU") if isinstance(act_cfg, dict) else "ReLU"
                layers.append(getattr(nn, act_type, nn.ReLU)())
            super().__init__(*layers)
from ultralytics.nn.modules.conv import Conv

__all__ = ['LMSA']

################################新颖的多尺度卷积注意力（MSCA）########################################################################################
# 文件名: cgpa_module.py

import torch
from torch import nn


# =====================================================================================
# 1. 基础构建块 (Building Blocks)
# =====================================================================================

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))


# =====================================================================================
# 2. 核心轻量化模块 (Core Lightweight Modules)
# =====================================================================================

class GhostConv(nn.Module):
    """
    Ghost Convolution from GhostNet.
    This module reduces parameters by generating "ghost" feature maps from "intrinsic" ones
    using cheap linear operations (depthwise convolutions).
    """

    def __init__(self, c1, c2, k=1, s=1, d=1, ratio=2):
        super().__init__()
        self.c_out = c2
        # Calculate the number of intrinsic feature maps
        c_intrinsic = c2 // ratio
        c_ghost = c2 - c_intrinsic

        # Convolution to generate intrinsic feature maps
        self.conv_intrinsic = Conv(c1, c_intrinsic, k=k, s=s, d=d)

        # Cheap depthwise convolution to generate ghost feature maps
        # A 3x3 kernel is commonly used here to add non-linearity, even if the primary conv is 1x1
        self.conv_ghost = Conv(c_intrinsic, c_ghost, k=3, s=1, g=c_intrinsic) if c_ghost > 0 else nn.Identity()

    def forward(self, x):
        x_intrinsic = self.conv_intrinsic(x)
        if hasattr(self, 'conv_ghost') and isinstance(self.conv_ghost, Conv):
            x_ghost = self.conv_ghost(x_intrinsic)
            # Concatenate intrinsic and ghost features
            return torch.cat((x_intrinsic, x_ghost), dim=1)
        return x_intrinsic


class DWR_Ghost(nn.Module):
    """
    Lightweight version of the DWR module using GhostConv.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        bottleneck_dim = dim // 2

        # --- Stage 1: Bottleneck ---
        # Depthwise part of the initial separable convolution
        self.dw_conv_3x3 = Conv(dim, dim, k=3, g=dim)
        # Pointwise part is replaced by GhostConv
        self.ghost_pointwise_3x3 = GhostConv(dim, bottleneck_dim, k=1)

        # --- Stage 2: Parallel Dilated Branches ---
        # Branch 1 (d=1)
        self.dw_conv_d1 = Conv(bottleneck_dim, bottleneck_dim, k=3, g=bottleneck_dim, d=1)
        self.ghost_pointwise_d1 = GhostConv(bottleneck_dim, dim, k=1)

        # Branch 2 (d=3)
        self.dw_conv_d3 = Conv(bottleneck_dim, bottleneck_dim, k=3, g=bottleneck_dim, d=3)
        self.ghost_pointwise_d3 = GhostConv(bottleneck_dim, dim // 2, k=1)

        # Branch 3 (d=5)
        self.dw_conv_d5 = Conv(bottleneck_dim, bottleneck_dim, k=3, g=bottleneck_dim, d=5)
        self.ghost_pointwise_d5 = GhostConv(bottleneck_dim, dim // 2, k=1)

        # --- Stage 3: Fusion ---
        # Final 1x1 fusion layer, also optimized with GhostConv
        self.ghost_conv_1x1_fusion = GhostConv(dim * 2, dim, k=1)

    def forward(self, x):
        # Apply bottleneck
        x_dw = self.dw_conv_3x3(x)
        x_bottleneck = self.ghost_pointwise_3x3(x_dw)

        # Process through parallel branches
        x1 = self.ghost_pointwise_d1(self.dw_conv_d1(x_bottleneck))
        x2 = self.ghost_pointwise_d3(self.dw_conv_d3(x_bottleneck))
        x3 = self.ghost_pointwise_d5(self.dw_conv_d5(x_bottleneck))

        # Concatenate and fuse
        x_out = torch.cat([x1, x2, x3], dim=1)
        x_out = self.ghost_conv_1x1_fusion(x_out) + x
        return x_out


# =====================================================================================
# 3. 高效注意力模块 (Efficient Attention Module)
# =====================================================================================

class MSCAAttention(nn.Module):
    # SegNext NeurIPS 2022
    def __init__(self, dim):
        super().__init__()
        self.gelu = nn.GELU()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.gelu(attn)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u


# =====================================================================================
# 4. 最终组合模块 (Final Assembled Module)
# =====================================================================================

class LMSA(nn.Module):
    """
    Cascaded Ghost Pyramid Attention (CGPA) Module.
    """

    def __init__(self, dim: int, a=1, b=1):
        super().__init__()
        self.feature_extractor = DWR_Ghost(dim=dim)
        self.attention_refiner = MSCAAttention(dim=dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.attention_refiner(x)
        return x


class MSCAAttention(nn.Module):
    # SegNext NeurIPS 2022
    # https://github.com/Visual-Attention-Network/SegNeXt/tree/main
    def __init__(self, dim):
        super().__init__()
        self.gelu = nn.GELU()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x=self.conv3(x)
        x=self.gelu(x)
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u

########################################################################################################################
##################上下文锚点注意力（CAA）######################################################################################################

def autopad(kernel_size: int, padding: int = None, dilation: int = 1):
    assert kernel_size % 2 == 1, 'if use autopad, kernel size must be odd'
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1
    if padding is None:
        padding = kernel_size // 2
    return padding


def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """Make divisible function.
    This function rounds the channel number to the nearest value that can be
    divisible by the divisor. It is taken from the original tf repo. It ensures
    that all layers have a channel number that is divisible by divisor. It can
    be seen here: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py  # noqa
    Args:
        value (int, float): The original channel number.
        divisor (int): The divisor to fully divide the channel number.
        min_value (int): The minimum value of the output channel.
            Default: None, means that the minimum value equal to the divisor.
        min_ratio (float): The minimum ratio of the rounded channel number to
            the original channel number. Default: 0.9.
    Returns:
        int: The modified output channel number.
    """

    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than (1-min_ratio).
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


class BCHW2BHWC(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x.permute([0, 2, 3, 1])


class BHWC2BCHW(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x.permute([0, 3, 1, 2])


class GSiLU(BaseModule):
    """Global Sigmoid-Gated Linear Unit, reproduced from paper <SIMPLE CNN FOR VISION>"""

    def __init__(self):
        super().__init__()
        self.adpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return x * torch.sigmoid(self.adpool(x))


class CAA(BaseModule):
    """Context Anchor Attention"""

    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor


class C3k2_CAA(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else CAA(self.c) for _ in range(n)
        )

########################################################################################################################

class DWR(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv_3x3 = Conv(dim, dim // 2, 3)

        self.conv_3x3_d1 = Conv(dim // 2, dim, 3, d=1)
        self.conv_3x3_d3 = Conv(dim // 2, dim // 2, 3, d=3)
        self.conv_3x3_d5 = Conv(dim // 2, dim // 2, 3, d=5)

        self.conv_1x1 = Conv(dim * 2, dim, k=1)

    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)
        x1, x2, x3 = self.conv_3x3_d1(conv_3x3), self.conv_3x3_d3(conv_3x3), self.conv_3x3_d5(conv_3x3)
        x_out = torch.cat([x1, x2, x3], dim=1)
        x_out = self.conv_1x1(x_out) + x
        return x_out


class C3k2_DWR(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else DWR(self.c) for _ in range(n)
        )

########################################################################################################################
# class Conv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1, bias=False):
#         super().__init__()
#         # 如果没有指定 padding，则自动计算以保持输出尺寸不变 (适用于奇数 kernel_size)
#         if padding is None:
#             padding = kernel_size // 2 if isinstance(kernel_size, int) else tuple(k // 2 for k in kernel_size)
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         return self.relu(self.bn(self.conv(x)))
# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, bias=False):
#         super().__init__()
#         if padding is None:
#             padding = kernel_size // 2 if isinstance(kernel_size, int) else tuple(k // 2 for k in kernel_size)
#
#         # 逐深度卷积：对每个输入通道独立应用一个卷积核
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
#         self.bn1 = nn.BatchNorm2d(in_channels)
#         self.relu1 = nn.ReLU(inplace=True)
#
#         # 逐点卷积：1x1 卷积，用于组合逐深度卷积的输出并混合通道信息
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.relu2 = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.relu1(self.bn1(self.depthwise(x)))
#         x = self.relu2(self.bn2(self.pointwise(x)))
#         return x
# class DWR(nn.Module):
#     def __init__(self, dim) -> None:
#         super().__init__()
#
#         # 将 Conv(dim, dim // 2, 3) 替换为 DepthwiseSeparableConv
#         self.conv_3x3 = DepthwiseSeparableConv(dim, dim // 2, 3)
#
#         # 扩张卷积也替换为 DepthwiseSeparableConv
#         self.conv_3x3_d1 = DepthwiseSeparableConv(dim // 2, dim, 3, dilation=1)
#         self.conv_3x3_d3 = DepthwiseSeparableConv(dim // 2, dim // 2, 3, dilation=3)
#         self.conv_3x3_d5 = DepthwiseSeparableConv(dim // 2, dim // 2, 3, dilation=5)
#
#         # 1x1 卷积 (pointwise convolution) 不需要替换为 DepthwiseSeparableConv，
#         # 因为它本身就是逐点卷积，可以使用常规的 Conv 模块。
#         self.conv_1x1 = Conv(dim * 2, dim, kernel_size=1)
#
#     def forward(self, x):
#         conv_3x3 = self.conv_3x3(x)
#         x1, x2, x3 = self.conv_3x3_d1(conv_3x3), self.conv_3x3_d3(conv_3x3), self.conv_3x3_d5(conv_3x3)
#         x_out = torch.cat([x1, x2, x3], dim=1)
#         x_out = self.conv_1x1(x_out) + x
#         return x_out

####################################################
def create_dwr_then_msca_sequential_block(dim: int) -> nn.Sequential:
    """
    创建一个 nn.Sequential 模块，将 DWR 和 MSCAAttention 模块按顺序串联起来。

    Args:
        dim (int): DWR 和 MSCAAttention 模块的通道维度。
                   确保此维度是这两个模块的共同输入/输出通道数。

    Returns:
        nn.Sequential: 包含 DWR 和 MSCAAttention 的 nn.Sequential 实例。
    """
    return nn.Sequential(
        DWR(dim=dim),
        MSCAAttention(dim=dim)
    )
# --- 串联方式 2: 自定义 nn.Module 来封装 ---
# class chuan_DWRAndMSCA(nn.Module):
#     def __init__(self, dim: int,a=1,b=1):
#         super().__init__()
#         self.dwr = DWR(dim=dim)
#         self.msca_attention = MSCAAttention(dim=dim)
#
#     def forward(self, x):
#         x = self.dwr(x)
#         x = self.msca_attention(x)
#         return x

