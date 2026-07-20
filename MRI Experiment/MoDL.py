import torch
import torch.nn as nn
# 💡 严格引入你保存的中心化 FFT 工具
from fft_utils import fft2c, ifft2c


class DataConsistencyLayer(nn.Module):
    def forward(self, x_rec_img, k0, mask_f):
        # 1. 将网络输入的实虚双通道图 [B, 2, H, W] 转换为原生复数图像 [B, H, W]
        x_complex = torch.view_as_complex(x_rec_img.permute(0, 2, 3, 1).contiguous())

        # 2. 直接调用标准 fft2c 变换到 k 空间
        k_rec = fft2c(x_complex)

        # 3. 数据一致性替代（此时 mask_f 已经是 float 保证性能）
        k_out = k_rec * (1 - mask_f) + k0 * mask_f

        # 4. 直接调用标准 ifft2c 变换回图像域复数
        x_out_c = ifft2c(k_out)

        # 5. 再转换回实虚双通道 [B, 2, H, W] 返回给 CNN
        return torch.view_as_real(x_out_c).permute(0, 3, 1, 2).contiguous()


class MoDL(nn.Module):
    """MoDL: 共享去噪器 + 数据一致性，迭代 K 次"""

    def __init__(self, K=5, denoiser_feats=64):
        super().__init__()
        self.K = K
        self.denoiser = nn.Sequential(
            nn.Conv2d(2, denoiser_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(denoiser_feats, denoiser_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(denoiser_feats, 2, 3, padding=1)
        )
        self.dc = DataConsistencyLayer()

    def forward(self, x, k0, mask):
        """
        x:    初始零填充图像，双通道 [B, 2, H, W]
        k0:   欠采样 K 空间，复数 [B, H, W]
        mask: 采样掩模 [B, H, W]
        """
        z = x
        mask_f = mask.float()  # 💡 优化：在迭代外只转换一次，提升显存效率

        for _ in range(self.K):
            z = self.dc(z, k0, mask_f)
            z = z + self.denoiser(z)

        return z.contiguous()  # 💡 优化：确保输出给 Loss 计算的张量内存绝对连续