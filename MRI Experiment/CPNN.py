import torch
import torch.nn as nn
from fft_utils import fft2c, ifft2c


class CPNN(nn.Module):
    """
    Complex Proximal Neural Network (噪声鲁棒版，仍无可学习参数)

    求解带噪声的 CMVI：
        z^{k+1} = prox_{ηλ||·||_1}( z^k - η A_soft^*(A_soft z^k - y) )
    其中 A_soft 是“软数据一致性”算子，根据噪声方差自适应地融合测量与当前估计。
    """

    def __init__(self, K=8, eta=0.5, lam=0.1, noise_std=0.0):
        super().__init__()
        self.K = K
        self.eta = eta
        self.lam = lam
        self.noise_std = noise_std          # 噪声标准差（归一化后，0 表示无噪声）

    def proximal_mapping(self, v):
        """复数软阈值，阈值为 eta * lam"""
        thresh = self.eta * self.lam
        return torch.sgn(v) * torch.relu(v.abs() - thresh)

    def forward(self, x, k0, mask):
        """
        x   : 零填充双通道图像 [B,2,H,W] （仅保留统一接口，内部不使用）
        k0  : 欠采样的 K 空间数据 [B,H,W] 复数
        mask: 采样掩模 [B,H,W] 浮点
        返回:
            out : 重建图像，双通道实数 [B,2,H,W]
        """
        y = k0
        z = ifft2c(y)                       # 零填充初始化
        noise_var = self.noise_std ** 2

        for _ in range(self.K):
            # --- 噪声自适应的软数据一致性 ---
            k_est = fft2c(z)
            if noise_var > 0:
                # 软投影：在采样点根据噪声方差调和当前估计与测量值
                k_new = (noise_var * k_est + mask * y) / (mask + noise_var)
            else:
                # 无噪声时退化为标准硬投影
                k_new = k_est * (1 - mask) + y * mask

            # 梯度 = 当前估计 - 软投影结果
            z = z - self.eta * ifft2c(k_est - k_new)

            # --- 近端映射 (L1 软阈值) ---
            z = self.proximal_mapping(z)

        out = torch.view_as_real(z).permute(0, 3, 1, 2).contiguous()
        return out