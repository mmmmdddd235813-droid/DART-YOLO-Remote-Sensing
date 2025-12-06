import torch
from torch import nn
import torch.nn.functional as F

import torch
import torch.nn as nn
__all__=['SPD']

class LightweightContextBranch(nn.Module):
    """
    SPD模块的组件1: 使用深度可分离卷积的高效上下文分支
    - 负责通过学习的方式进行下采样，并捕捉局部上下文信息。
    """

    def __init__(self, in_channels, out_channels):
        super(LightweightContextBranch, self).__init__()
        self.depthwise_separable_conv = nn.Sequential(
            # 3x3 深度卷积, stride=2, groups=in_channels
            # 负责空间下采样和特征提取，参数量极低
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            # 1x1 逐点卷积
            # 负责通道变换和信息融合
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.depthwise_separable_conv(x)


class SPD(nn.Module):
    """
    SPD (Channel-Split Downsampling) 模块

    一个轻量化且高效的下采样模块。它融合了CSPNet的通道分割思想和
    MobileNet的深度可分离卷积思想，通过并行分支处理上下文和显著性特征。

    工作流程:
    1. 通道分割 (Channel-Split): 输入通道一分为二。
    2. 并行处理 (Parallel):
        - 分支1 (上下文): 使用深度可分离卷积进行可学习的下采样。
        - 分支2 (显著性): 使用最大池化进行特征筛选，再用1x1卷积调整通道。
    3. 特征融合 (Fusion): 将两个分支的结果拼接，并通过1x1卷积进行信息融合。
    """

    def __init__(self, in_channels, out_channels, A=1,B=1, C=1):
        super(SDZ, self).__init__()
        # 按照CSP思想，将输入通道一分为二
        branch_in_channels = in_channels // 2

        # 两个分支的输出通道数设为最终输出通道数的一半
        # 这样拼接后总通道数就是 out_channels
        branch_out_channels = out_channels // 2

        # 分支1: 高效上下文分支
        self.branch1 = LightweightContextBranch(branch_in_channels, branch_out_channels)

        # 分支2: 高效显著性分支 (MaxPool + 1x1 Conv)
        self.branch2_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.branch2_conv = nn.Sequential(
            nn.Conv2d(branch_in_channels, branch_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_out_channels),
            nn.ReLU6(inplace=True)
        )

        # 两个分支拼接后，使用一个1x1卷积进行最终的信息融合
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        # 1. 通道分割
        x1, x2 = torch.chunk(x, 2, dim=1)

        # 2. 并行分支处理
        out1 = self.branch1(x1)

        pooled_x2 = self.branch2_pool(x2)
        out2 = self.branch2_conv(pooled_x2)

        # 3. 特征拼接 (Concat)
        out = torch.cat([out1, out2], dim=1)

        # 4. 最终信息融合
        out = self.final_conv(out)

        return out


# --- 使用示例 ---
if __name__ == '__main__':
    # 定义超参数
    # 假设输入特征图的批次大小为4，通道数为64，尺寸为56x56
    # 目标下采样后通道数变为128，尺寸变为28x28
    in_c = 64
    out_c = 128

    # 创建一个模拟的输入张量
    input_tensor = torch.randn(4, in_c, 56, 56)

    # 初始化 SPD 模块
    cspd_downsampler = SPD(in_channels=in_c, out_channels=out_c)

    # 执行前向传播
    output_tensor = cspd_downsampler(input_tensor)

    # 打印结果，验证模块是否正常工作
    print("--- SPD 模块运行验证 ---")
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output_tensor.shape}")

    # 检查输出形状是否符合预期 (空间尺寸减半，通道数改变)
    expected_shape = (4, out_c, 28, 28)
    assert output_tensor.shape == expected_shape, f"形状不匹配! 期望: {expected_shape}, 得到: {output_tensor.shape}"
    print("\n模块运行成功，输出形状符合预期！")