import torch
import torch.nn as nn
from fft_utils import fft2c, ifft2c   # 直接用数据集的变换！！

class ADMM_TV(nn.Module):
    def __init__(self, iter_num=50, rho=0.1, lam=0.01):
        super().__init__()
        self.iter_num = iter_num
        self.rho = rho          # 惩罚参数 ρ
        self.lam = lam          # L1 正则化系数 λ

    def forward(self, x, k0, mask):
        """
        参数:
            x    : 零填充图像 (双通道 [B,2,H,W]) —— 未使用，仅为接口统一
            k0   : 欠采样 K 空间 (复数 [B,H,W])
            mask : 采样掩模 (浮点 [B,H,W])
        返回:
            out  : 重建图像 (双通道 [B,2,H,W])
        """
        device = k0.device
        y = k0                     # 观测值（已采样点有数据，未采样点为0）
        mask_f = mask.float()

        # ---------- 初始化 ----------
        z = ifft2c(y)              # 零填充重建作为初始解
        d = torch.zeros_like(z)    # 辅助变量 (图像域)
        u = torch.zeros_like(z)    # 对偶变量 (图像域)

        threshold = self.lam / self.rho

        for _ in range(self.iter_num):
            # ---------- 1. z 更新：数据一致性 + 近端项 ----------
            # 解 min_z 1/2||M F z - y||^2 + (ρ/2)||z - (d - u)||^2
            F_du = fft2c(d - u)                        # F(d-u)
            numerator = mask_f * y + self.rho * F_du   # M^H y + ρ F(d-u)
            denominator = mask_f + self.rho            # M + ρ
            Z_new = numerator / denominator
            z_new = ifft2c(Z_new)

            # ---------- 2. d 更新：L1 软阈值 ----------
            v = z_new + u
            d = torch.sgn(v) * torch.relu(torch.abs(v) - threshold)

            # ---------- 3. u 更新（对偶上升）----------
            u = u + (z_new - d)

            # 进入下一轮
            z = z_new

        # 输出：双通道实数图像 [B,2,H,W]
        out = torch.view_as_real(z).permute(0, 3, 1, 2).contiguous()
        return out