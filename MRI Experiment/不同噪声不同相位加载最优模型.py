import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from LIPN_pro import LIPN_pro          # 确保路径正确
from fft_utils import fft2c, ifft2c

# --------------------------- 参数配置 ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = r"C:\Users\Administrator\Desktop\论文\变分不等式第二篇\数值实验\checkpoints\LIPN_pro_best.pth"
H, W = 64, 64                     # 图像尺寸（与训练一致）
num_test_samples = 20             # 测试样本数（随机生成复数图像）
noise_levels = [0.00, 0.05, 0.10, 0.20, 0.40, 0.80]   # 噪声标准差 ε
angles_deg = [0, 45, 90, 135, 180, 225, 270, 315, 360]  # 旋转角度（度）
angles_rad = [np.deg2rad(a) for a in angles_deg]

# --------------------------- 加载模型 ---------------------------
model = LIPN_pro(K=5).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# --------------------------- 生成固定测试样本（复数图像） ---------------------------
torch.manual_seed(42)   # 固定种子，保证可复现
test_images = []
for _ in range(num_test_samples):
    # 随机复数图像，幅度和相位均随机，保证覆盖各种情况
    img_real = torch.randn(H, W, dtype=torch.float32)
    img_imag = torch.randn(H, W, dtype=torch.float32)
    img_c = torch.complex(img_real, img_imag)
    # 可选：归一化到 [0,1] 范围（幅度归一化）
    img_c = img_c / (torch.abs(img_c).max() + 1e-8)
    test_images.append(img_c)

# 使用全采样掩码（简化测试，避免采样影响）
mask = torch.ones(H, W, dtype=torch.float32).to(device)

# --------------------------- 相位等变性误差计算函数 ---------------------------
def phase_equivariance_error(model, img_orig, angle_rad, noise_std, mask, device):
    """计算单个样本在给定旋转角度和噪声水平下的相对误差"""
    # 旋转原始图像
    phase_factor = torch.exp(1j * torch.tensor(angle_rad, device=device))
    img_rot = img_orig * phase_factor

    # 生成 k-space 数据（无噪声时直接使用；有噪声时添加复高斯噪声）
    k_orig = fft2c(img_orig.unsqueeze(0))           # [1, H, W]
    k_rot = fft2c(img_rot.unsqueeze(0))

    if noise_std > 0:
        noise_orig = torch.randn_like(k_orig.real) * noise_std + 1j * torch.randn_like(k_orig.imag) * noise_std
        noise_rot = torch.randn_like(k_rot.real) * noise_std + 1j * torch.randn_like(k_rot.imag) * noise_std
        k_orig_noisy = k_orig + noise_orig
        k_rot_noisy = k_rot + noise_rot
    else:
        k_orig_noisy = k_orig
        k_rot_noisy = k_rot

    # 构造模型输入（零填充图像 + k0 + mask）
    x_orig = ifft2c(k_orig_noisy * mask)
    x_rot = ifft2c(k_rot_noisy * mask)
    # 转换为 [B,2,H,W] 格式
    x_orig = torch.view_as_real(x_orig).permute(0, 3, 1, 2).contiguous()
    x_rot = torch.view_as_real(x_rot).permute(0, 3, 1, 2).contiguous()

    # 模型前向
    out_orig = model(x_orig, k_orig_noisy, mask.unsqueeze(0))
    out_rot = model(x_rot, k_rot_noisy, mask.unsqueeze(0))

    # 转换为复数图像
    out_orig_c = torch.view_as_complex(out_orig.permute(0, 2, 3, 1).contiguous())
    out_rot_c = torch.view_as_complex(out_rot.permute(0, 2, 3, 1).contiguous())

    # 理想输出应为旋转后的原始输出
    target = out_orig_c * phase_factor
    error = torch.norm(out_rot_c - target) / (torch.norm(target) + 1e-8)
    return error.item()

# --------------------------- 主测试循环 ---------------------------
results = np.zeros((len(noise_levels), len(angles_deg)))  # 行：噪声，列：角度

for i, noise_std in enumerate(noise_levels):
    for j, angle_rad in enumerate(angles_rad):
        errors = []
        for img in test_images:
            img = img.to(device)
            err = phase_equivariance_error(model, img, angle_rad, noise_std, mask, device)
            errors.append(err)
        results[i, j] = np.mean(errors)
        print(f"噪声 ε={noise_std:.2f}, 角度={angles_deg[j]}°, 平均误差={results[i,j]:.2e}")

# --------------------------- 绘制高级热力图 ---------------------------
sns.set_theme(style='whitegrid', font_scale=1.2)
plt.figure(figsize=(10, 6))
ax = sns.heatmap(results,
                 xticklabels=[f"{a}°" for a in angles_deg],
                 yticklabels=[f"ε={nl:.2f}" for nl in noise_levels],
                 annot=True, fmt='.2e', cmap='coolwarm',
                 cbar_kws={'label': 'Relative Phase Rotation Error'},
                 square=True, linewidths=0.5, linecolor='white')
ax.set_xlabel('Rotation Angle', fontsize=12)
ax.set_ylabel('Noise Level ε', fontsize=12)
ax.set_title('LIPN-pro Phase Equivariance Error under Various Noise Levels', fontsize=14)
plt.tight_layout()
plt.savefig('phase_equivariance_heatmap.png', dpi=600, bbox_inches='tight')
plt.show()

# --------------------------- 可选：曲线图（不同噪声下误差随角度的变化） ---------------------------
plt.figure(figsize=(9, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(noise_levels)))
for i, noise_std in enumerate(noise_levels):
    plt.plot(angles_deg, results[i, :], marker='o', linestyle='-', color=colors[i],
             label=f'ε={noise_std:.2f}', linewidth=2)
plt.xlabel('Rotation Angle (degree)')
plt.ylabel('Relative Phase Rotation Error')
plt.title('Phase Equivariance Error vs Rotation Angle')
plt.yscale('log')   # 误差通常很小，对数坐标更清晰
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('phase_equivariance_curves.png', dpi=600)
plt.show()

print("\n测试完成，已保存热力图和曲线图。")