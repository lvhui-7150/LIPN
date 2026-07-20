import torch
import torch.nn as nn
from fft_utils import ifft2c   # 完美统一调用

class ComplexUNet(nn.Module):
    """
    复数域 U-Net 变体，已统一 FFT 约定，并移除了小 Batch 隐患。
    输入 (x, k0, mask)，输出 [B,2,H,W] 双通道重建图像
    """
    def __init__(self):
        super(ComplexUNet, self).__init__()
        # 编码器
        self.enc1 = self.conv_block(2, 32)
        self.enc2 = self.conv_block(32, 64)
        # 解码器
        self.dec2 = self.conv_block(64, 32)
        self.dec1 = nn.Conv2d(32, 2, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),  # 💡 核心修正：换成 IN 确保 8GB 显卡小 Batch 训练绝不崩塌
            nn.ReLU(inplace=True),
        )

    def forward(self, x, k0, mask):
        """
        参数:
            x   : 零填充双通道图像 [B,2,H,W] （本模型未使用）
            k0  : 欠采样 K 空间 [B,H,W] (复数)
            mask: 采样掩模 [B,H,W] （未使用）
        返回:
            out : 重建双通道图像 [B,2,H,W]
        """
        # 使用统一的 ifft2c 从 k0 生成零填充复数图像
        x_und = ifft2c(k0)                     # 复数 [B,H,W]
        # 转为双通道实数 [B,2,H,W]
        x_real = torch.view_as_real(x_und).permute(0, 3, 1, 2).contiguous()

        # U-Net 前向拓扑结构
        e1 = self.enc1(x_real)                        # [B,32,H,W]
        e2 = self.enc2(nn.functional.max_pool2d(e1, 2))  # [B,64,H/2,W/2]
        d2 = self.dec2(nn.functional.interpolate(e2, scale_factor=2, mode='bilinear', align_corners=False))
        out = self.dec1(d2 + e1)                      # [B,2,H,W]
        return out