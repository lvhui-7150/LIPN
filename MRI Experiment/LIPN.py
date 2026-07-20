import torch
import torch.nn as nn
from fft_utils import fft2c, ifft2c


# ------------------------------------------------------------
class Denoiser(nn.Module):
    def __init__(self, ch=2, inner_feats=64, num_blocks=4):
        super().__init__()
        # 所有卷积均用谱归一化保证 Lipschitz ≤ 1
        layers = [
            nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(ch, inner_feats, 3, padding=1)),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_blocks):
            layers.append(self._block(inner_feats))
        layers.append(
            nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(inner_feats, ch, 3, padding=1))
        )
        self.net = nn.Sequential(*layers)

    def _block(self, feats):
        return nn.Sequential(
            nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(feats, feats, 3, padding=1)),
            nn.ReLU(inplace=True),
            nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(feats, feats, 3, padding=1)),
            nn.ReLU(inplace=True)
        )

    def forward(self, z):
        # z : 复数 [B, H, W]
        z_stack = torch.stack([z.real, z.imag], dim=1)   # [B,2,H,W]
        out = self.net(z_stack)

        res = torch.complex(out[:, 0], out[:, 1])
        return 0.5 * (z + res)          # ⇒ G_θ 是 firmly nonexpansive


# ------------------------------------------------------------
class LIPN(nn.Module):
    def __init__(self, K=5, eta_init=0.5):
        super().__init__()
        self.K = K
        # 可学习的步长（保留你原有的设计）
        self.eta = nn.Parameter(torch.ones(K) * eta_init)
        # 使用受约束的去噪器
        self.denoiser = Denoiser()

    # 数据一致性投影（投影到仿射子空间，firmly nonexpansive）
    def dc(self, z, y, mask):
        k = fft2c(z)
        k = k * (1 - mask) + y * mask
        return ifft2c(k)

    def forward(self, x, k0, mask):
        z = torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous())
        for i in range(self.K):
            # --- 步骤1：梯度步（对应 f(z) 的 forward‑backward）---
            grad = ifft2c(fft2c(z) * mask - k0)      # f(z)
            z = z - self.eta[i] * grad               # z - η f(z)

            # --- 步骤2：学习近端映射（代替 prox_{ηΦ}）---
            z = self.denoiser(z)                     # G_θ(·)

            # --- 步骤3：硬数据一致性投影（可选，见理论解释）---
            z = self.dc(z, k0, mask)
        return torch.view_as_real(z).permute(0, 3, 1, 2)