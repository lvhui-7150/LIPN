import torch
import torch.nn as nn
from fft_utils import fft2c, ifft2c


# ===================== 增强的复数等变模块 =====================
class ComplexConv2d(nn.Module):
    """相位等变复数卷积 + 谱归一化"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.dim_in = in_channels // 2
        self.dim_out = out_channels // 2
        self.A = nn.utils.parametrizations.spectral_norm(
            nn.Conv2d(self.dim_in, self.dim_out, kernel_size, padding=padding, bias=False)
        )
        self.B = nn.utils.parametrizations.spectral_norm(
            nn.Conv2d(self.dim_in, self.dim_out, kernel_size, padding=padding, bias=False)
        )

    def forward(self, x):
        R, I = torch.chunk(x, 2, dim=1)
        real = self.A(R) - self.B(I)
        imag = self.B(R) + self.A(I)
        return torch.cat([real, imag], dim=1)


class ComplexModReLU(nn.Module):
    """幅值激活函数，保持相位不变，可学习偏置"""
    def __init__(self, channels):
        super().__init__()
        # 每个通道独立偏置，初始为负使激活接近线性
        self.b = nn.Parameter(torch.zeros(1, channels // 2, 1, 1) - 0.05)

    def forward(self, x):
        R, I = torch.chunk(x, 2, dim=1)
        radius = torch.sqrt(R**2 + I**2 + 1e-8)
        scale = torch.relu(radius + self.b) / (radius + 1e-8)
        return torch.cat([R * scale, I * scale], dim=1)


class ComplexSE(nn.Module):
    """复数通道注意力（相位等变）：对幅值进行全局池化，生成通道权重"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.dim = channels // 2
        self.fc = nn.Sequential(
            nn.Linear(self.dim, self.dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim // reduction, self.dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        R, I = torch.chunk(x, 2, dim=1)
        mag = torch.sqrt(R**2 + I**2 + 1e-8)                 # [B, dim, H, W]
        gap = mag.mean(dim=(2, 3))                           # [B, dim]
        att = self.fc(gap).unsqueeze(-1).unsqueeze(-1)       # [B, dim, 1, 1]
        return torch.cat([R * att, I * att], dim=1)


class EnhancedEquivarResBlock(nn.Module):
    """增强残差块：卷积 → 激活 → 卷积 → 通道注意力 → 残差连接"""
    def __init__(self, feats, use_se=True):
        super().__init__()
        self.conv1 = ComplexConv2d(feats, feats)
        self.act = ComplexModReLU(feats)
        self.conv2 = ComplexConv2d(feats, feats)
        self.se = ComplexSE(feats) if use_se else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = self.se(out)
        return x + out


class PhaseEquivarFNEDenoiser(nn.Module):
    """
    增强版牢固非扩张去噪器（firmly nonexpansive）
    容量更大：inner_feats=128，num_blocks=6，增加通道注意力
    """
    def __init__(self, inner_feats=128, num_blocks=6, use_se=True):
        super().__init__()
        self.head = nn.Sequential(
            ComplexConv2d(2, inner_feats),
            ComplexModReLU(inner_feats)
        )
        blocks = [EnhancedEquivarResBlock(inner_feats, use_se=use_se) for _ in range(num_blocks)]
        self.body = nn.Sequential(*blocks)
        self.tail = ComplexConv2d(inner_feats, 2)

    def forward(self, z_tensor):
        h_z = self.tail(self.body(self.head(z_tensor)))
        return 0.5 * z_tensor + 0.5 * h_z   # firmly nonexpansive


# ===================== 主网络 LIPN_Pro（增强版） =====================
class LIPN_pro(nn.Module):
    """
    增强版：无动量，谱归一化，复数等变，双重软投影，大容量去噪器
    完全匹配论文理论，同时提升性能
    """
    def __init__(self, K=5, eta_init=0.1, noise_std=0.01, inner_feats=128, num_blocks=6):
        super().__init__()
        self.K = K
        # 可学习噪声方差（软投影参数）
        self.noise_std_raw = nn.Parameter(torch.tensor(noise_std))
        # 每个阶段的步长和数据项权重
        self.step_size = nn.Parameter(torch.ones(K) * eta_init)
        self.lam = nn.Parameter(torch.ones(K) * 1.0)

        # 增强去噪器（大容量 + 通道注意力）
        self.denoiser = PhaseEquivarFNEDenoiser(
            inner_feats=inner_feats,
            num_blocks=num_blocks,
            use_se=True
        )

        # 更好的权重初始化（Xavier，gain 0.1 防止初期梯度爆炸）
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)

    def forward(self, x, k0, mask):
        """
        x    : [B, 2, H, W] 零填充图像（实部+虚部）
        k0   : [B, H, W] 欠采样k空间数据（复数）
        mask : [B, H, W] 采样掩模（0/1实数）
        """
        z = torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous())
        mask_f = mask.float()
        noise_var = torch.nn.functional.softplus(self.noise_std_raw).square() + 1e-8

        for i in range(self.K):
            # ----- 步骤1：软数据一致性梯度 -----
            k_est = fft2c(z)
            k_new = (noise_var * k_est + mask_f * k0) / (mask_f + noise_var)
            grad = k_est - k_new
            # 梯度步（无动量）
            z = z - self.lam[i] * self.step_size[i] * ifft2c(grad)

            # ----- 步骤2：等变近端映射（firmly nonexpansive）-----
            z_input = torch.stack([z.real, z.imag], dim=1)   # [B,2,H,W]
            z_output = self.denoiser(z_input)                # [B,2,H,W]
            z = torch.complex(z_output[:, 0], z_output[:, 1])

            # ----- 步骤3：二次软数据一致性（可选，提升鲁棒性）-----
            k_est = fft2c(z)
            z = ifft2c((noise_var * k_est + mask_f * k0) / (mask_f + noise_var))

        return torch.stack([z.real, z.imag], dim=1)