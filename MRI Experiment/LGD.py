import torch
import torch.nn as nn
# 严格引入你保存的中心化 FFT 工具
from fft_utils import fft2c, ifft2c


class LGD_Net(nn.Module):
    def __init__(self, num_iters):
        super().__init__()
        self.K = num_iters
        # 每一步的梯度修正项
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 2, 3, padding=1)
            ) for _ in range(num_iters)
        ])

    def forward(self, x, k0, mask):
        """
        统一接口:
            x: 零填充双通道图像 [B,2,H,W] (本模型未使用)
            k0: 欠采样 K 空间 [B,H,W] (复数)
            mask: 采样掩模 [B,H,W] (浮点, 1表示采样)
        返回:
            out: 重建双通道图像 [B,2,H,W]
        """
        y = k0
        mask_f = mask.float()

        # 使用统一的 ifft2c 从 k0 生成初始复数图像 [B,H,W]
        z_k = ifft2c(y)

        for i in range(self.K):
            # 1. 物理梯度步: grad = IFFT( FFT(z) * mask - y )
            grad = ifft2c(fft2c(z_k) * mask_f - y)

            # 2. 变换为网络输入: [B,H,W] 复数 -> [B,2,H,W] 实数双通道 (零拷贝高性能 view)
            inp = torch.view_as_real(z_k).permute(0, 3, 1, 2).contiguous()

            # 3. 网络预测网络修正项
            corr = self.layers[i](inp)

            # 4. 变换回复数域: [B,2,H,W] -> [B,H,W] 复数
            corr_cmplx = torch.view_as_complex(corr.permute(0, 2, 3, 1).contiguous())

            # 5. LGD 状态更新
            z_k = z_k - grad + corr_cmplx

        # 输出统一的双通道实数图像 [B,2,H,W]
        out = torch.view_as_real(z_k).permute(0, 3, 1, 2).contiguous()
        return out