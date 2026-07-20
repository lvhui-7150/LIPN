import torch
import torch.nn as nn
# 💡 严格引入你保存的中心化 FFT 工具
from fft_utils import fft2c, ifft2c


class DataConsistencyLayer(nn.Module):
    def forward(self, x_rec_img, k0, mask_f):
        # 1. 将网络输入的实虚双通道图 (B, 2, H, W) 转换为复数图像 (B, H, W)
        x_complex = torch.view_as_complex(x_rec_img.permute(0, 2, 3, 1).contiguous())

        # 2. 直接调用标准 fft2c 变换到 k 空间
        k_rec = fft2c(x_complex)

        # 3. 数据一致性替代（采用提前转好的 mask_f）
        k_out = k_rec * (1 - mask_f) + k0 * mask_f

        # 4. 直接调用标准 ifft2c 变换回图像域
        x_out_c = ifft2c(k_out)

        # 5. 再转换回实虚双通道 (B, 2, H, W) 返回给 CNN
        return torch.view_as_real(x_out_c).permute(0, 3, 1, 2).contiguous()


class VarNet(nn.Module):
    """Variational Network: 独立级联复数卷积 + 数据一致性，含可学习步长"""

    def __init__(self, K=5, inner_feats=64):
        super().__init__()
        self.K = K
        # 每一个迭代步使用独立的 CNN 去噪器（参数不共享）
        self.denoisers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, inner_feats, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(inner_feats, inner_feats, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(inner_feats, 2, 3, padding=1)
            ) for _ in range(K)
        ])
        self.dc_layers = nn.ModuleList([DataConsistencyLayer() for _ in range(K)])
        # 可学习的解耦步长 eta
        self.etas = nn.Parameter(torch.ones(K) * 0.5)

    def forward(self, x, k0, mask):
        """
        x:    初始零填充图像，双通道 [B, 2, H, W]
        k0:   欠采样 K 空间，复数 [B, H, W]
        mask: 采样掩模 [B, H, W]
        """
        z = x
        mask_f = mask.float()  # 💡 优化：在迭代外转换一次，避免 K 次前向时反复开辟新显存

        for i in range(self.K):
            z = self.dc_layers[i](z, k0, mask_f)
            delta = self.denoisers[i](z)
            z = z + self.etas[i] * delta

        return z.contiguous()  # 💡 优化：锁死输出内存连续性，保护下游评测指标