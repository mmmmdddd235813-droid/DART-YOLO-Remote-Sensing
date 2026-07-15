"""Dynamic Global Attention Module for DART-YOLO."""

import torch
import torch.nn as nn

try:
    from mmengine.model import caffe2_xavier_init, constant_init
except ModuleNotFoundError:
    def caffe2_xavier_init(module):
        nn.init.xavier_uniform_(module.weight)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias, 0)

    def constant_init(module, val, bias=0):
        nn.init.constant_(module.weight, val)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias, bias)
from ultralytics.nn.modules.conv import Conv

__all__ = ['DGAM']
class ConvBNAct(nn.Module):
    """
    一个简化的 Conv-BN-Activation 模块，用于替代 ConvModule，使代码自洽。
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, act_cfg=nn.GELU, norm_cfg=nn.BatchNorm2d,
                 bias=None):
        super().__init__()
        # 如果有BN层，通常不使用偏置
        use_bias = bias if bias is not None else (norm_cfg is None)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, bias=use_bias)
        self.norm = norm_cfg(out_channels) if norm_cfg is not None else nn.Identity()
        self.act = act_cfg() if act_cfg is not None else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class GhostConv(nn.Module):
    """
    Ghost Convolution Module to reduce parameters and FLOPs.
    """

    def __init__(self, c1, c2, k=1, s=1, ratio=2):
        super().__init__()
        c_intrinsic = c2 // ratio
        c_ghost = c2 - c_intrinsic

        # 使用常规 Conv 生成内生特征图 (Pointwise Conv)
        self.conv_intrinsic = ConvBNAct(c1, c_intrinsic, kernel_size=k, stride=s, act_cfg=nn.GELU)

        # 使用廉价的深度卷积生成鬼影特征图 (Depthwise Conv)
        self.conv_ghost = ConvBNAct(c_intrinsic, c_ghost, kernel_size=3, stride=1,
                                    act_cfg=nn.GELU) if c_ghost > 0 else nn.Identity()

    def forward(self, x):
        x_intrinsic = self.conv_intrinsic(x)
        if hasattr(self, 'conv_ghost') and isinstance(self.conv_ghost, ConvBNAct):
            x_ghost = self.conv_ghost(x_intrinsic)
            return torch.cat((x_intrinsic, x_ghost), dim=1)
        return x_intrinsic


# =====================================================================================
# 2. 最终的轻量化 ContextAggregation 模块
# =====================================================================================

class DGAM(nn.Module):
    """
    Lightweight version of ContextAggregation using GhostConv to optimize 1x1 convolutions.
    """

    def __init__(self, in_channels, reduction=1, a=1):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = max(in_channels // reduction, 1)

        # --- Part 1: Self-Attention like mechanism (Optimized) ---
        self.a = nn.Sequential(
            GhostConv(in_channels, in_channels // 2),
            nn.GELU(),
            # The final layer to 1 channel is cheap, no need to optimize
            nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        )
        self.k = nn.Sequential(
            GhostConv(in_channels, in_channels // 2),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        )
        # V and M branches have fewer parameters, can be kept for performance
        self.v = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.m = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
        nn.init.constant_(self.m.weight, 0)
        nn.init.constant_(self.m.bias, 0)

        # --- Part 2: Coordinate Attention like mechanism (Optimized) ---
        mid_channels = max(8, in_channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = ConvBNAct(in_channels, mid_channels, kernel_size=1, act_cfg=nn.Hardswish)  # Keep this one simple

        # Optimize the two most expensive convolutions here
        self.conv_h = GhostConv(mid_channels, in_channels, k=1)
        self.conv_w = GhostConv(mid_channels, in_channels, k=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Part 1: Self-Attention like mechanism
        n = x.size(0)
        c_inter = self.inter_channels

        a = self.a(x).sigmoid()
        k = self.k(x).view(n, 1, -1, 1).softmax(2)
        v = self.v(x).view(n, 1, c_inter, -1)

        y = torch.matmul(v, k).view(n, c_inter, 1, 1)
        y = self.m(y) * a
        out = x + y

        # Part 2: Coordinate Attention like mechanism
        identity = out
        _, _, h, w = out.size()

        pooled_h = self.pool_h(out)
        pooled_w = self.pool_w(out).permute(0, 1, 3, 2)

        combined_features = torch.cat([pooled_h, pooled_w], dim=2)
        transformed_features = self.conv1(combined_features)

        x_h_attn, x_w_attn = torch.split(transformed_features, [h, w], dim=2)

        attention_h = self.sigmoid(self.conv_h(x_h_attn))
        attention_w = self.sigmoid(self.conv_w(x_w_attn)).permute(0, 1, 3, 2)

        return identity * attention_h * attention_w
