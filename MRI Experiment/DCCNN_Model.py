import torch
import torch.nn as nn
# 严格引入你保存的中心化 FFT 工具
from fft_utils import fft2c, ifft2c


class DataConsistencyLayer(nn.Module):
    def forward(self, x_rec_img, k0, mask):
        # x_rec_img: [B, 2, H, W] -> 实部和虚部双通道
        # 将双通道实数转换成 PyTorch 原生复数 [B, H, W]
        x_complex = torch.view_as_complex(x_rec_img.permute(0, 2, 3, 1).contiguous())

        # 调用中心化 FFT，将图像转到 K 空间
        k_rec = fft2c(x_complex)

        # 数据融合 (确保 mask 是浮点数)
        mask = mask.float()
        k_out = k_rec * (1 - mask) + k0 * mask

        # 调用中心化 IFFT，从 K 空间转回图像域复数
        x_out_c = ifft2c(k_out)

        # 重新转回双通道实数 [B, 2, H, W] 返回给网络
        return torch.view_as_real(x_out_c).permute(0, 3, 1, 2).contiguous()


class ConvBlock(nn.Module):
    def __init__(self, inner_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, inner_channels, 3, padding=1),
            nn.InstanceNorm2d(inner_channels, affine=True),  # 第一处 IN
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1),
            nn.InstanceNorm2d(inner_channels, affine=True),  # 💡 第二处也改成了 IN，彻底解决小 Batch 问题
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, 2, 3, padding=1)
        )

    def forward(self, x):
        return x + self.net(x)


class DCCNN(nn.Module):
    """DC-CNN 级联网络，接口与训练循环完美匹配"""

    def __init__(self, n=5):
        super().__init__()
        self.cascades = nn.ModuleList([
            nn.ModuleDict({'c': ConvBlock(), 'd': DataConsistencyLayer()}) for _ in range(n)
        ])

    def forward(self, x, k0, m):
        """
        x: 初始零填充图像，双通道 [B,2,H,W]
        k0: 欠采样K空间，复数 [B,H,W]
        m: 采样掩模，浮点 [B,H,W]
        """
        for stage in self.cascades:
            x = stage['c'](x)
            x = stage['d'](x, k0, m)
        return x