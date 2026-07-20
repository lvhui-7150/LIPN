import torch
import torch.nn as nn
import torch.nn.functional as F
# 💡 核心修正：严格引入你保存的中心化 FFT 工具
from fft_utils import fft2c, ifft2c


class ISTANet(nn.Module):
    """
    标准的学术级 ISTA-Net (已完全统一 FFT 约定，修复残差与阈值约束)
    """

    def __init__(self, K=5, latent_dim=32):
        super().__init__()
        self.K = K

        # 1. 可学习的梯度步长 \eta
        self.eta = nn.Parameter(torch.ones(K) * 0.5)

        # 2. 每一级迭代专用的对称稀疏变换对 (Encoder-Decoder)
        self.enc = nn.ModuleList([nn.Conv2d(2, latent_dim, 3, padding=1) for _ in range(K)])
        self.dec = nn.ModuleList([nn.Conv2d(latent_dim, 2, 3, padding=1) for _ in range(K)])

        # 3. 软阈值参数（初始化为 0.1）
        self.thresh = nn.Parameter(torch.ones(K) * 0.1)

    def forward(self, x, k0, mask):
        """
        x   : 初始图像 (双通道 [B, 2, H, W])
        k0  : 原始采样 k 空间数据 (复数 [B, H, W])
        mask: 采样掩码 (浮点 [B, H, W])
        """
        z = x
        mask_f = mask.float()

        for i in range(self.K):
            # --- 【步骤 1】真正的物理梯度步 (Data Fidelity Gradient Step) ---
            # 1. 先把当前双通道图像转到原生复数 [B, H, W]
            z_complex = torch.view_as_complex(z.permute(0, 2, 3, 1).contiguous())

            # 2. 直接调用中心化 FFT，将图像转到 K 空间
            k_rec = fft2c(z_complex)

            # 3. 计算数据保真项的梯度: A^T(Ax - y)
            grad_k = (k_rec * mask_f - k0) * mask_f

            # 4. 直接调用中心化 IFFT，转回图像域复数
            grad_img_c = ifft2c(grad_k)

            # 5. 重新拆回双通道实数 [B, 2, H, W]
            grad_img = torch.view_as_real(grad_img_c).permute(0, 3, 1, 2).contiguous()

            # 6. 梯度更新 (得到中间变量 r)
            r = z - self.eta[i] * grad_img

            # --- 【步骤 2】近端映射步 (Proximal Step / Sparse Nonlinear Transform) ---
            feat = self.enc[i](r)

            # 强制将阈值约束为正数，维持收敛性
            th = F.relu(self.thresh[i])
            feat_thresh = torch.sgn(feat) * F.relu(torch.abs(feat) - th)

            # 严格的残差连接：dec 出来的结构加回给 r，保证大结构不丢失
            z = r + self.dec[i](feat_thresh)

        return z