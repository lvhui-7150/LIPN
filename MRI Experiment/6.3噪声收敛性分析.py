import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from torch.utils.data import Dataset, DataLoader, random_split
import random
import csv

# 导入自定义模块（根据实际位置调整）
from LIPN_pro import LIPN_pro                # 请确保 LIPN_pro.py 在 sys.path 中
from fft_utils import fft2c, ifft2c

# ---------- 1. 数据集定义（与训练时保持一致）----------
class MRIDataset(Dataset):
    def __init__(self, data_dir, sampling_rate=0.3, mask_type='cartesian', normalize_to_01=True):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.sampling_rate = sampling_rate
        self.mask_type = mask_type
        self.normalize_to_01 = normalize_to_01

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.file_list[idx])
        data = torch.load(path)
        if torch.is_complex(data):
            img_c = data
        else:
            img = data.float()
            if self.normalize_to_01:
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img_c = img.to(torch.complex64)

        H, W = img_c.shape
        mask = self._generate_mask(H, W)
        return img_c, mask

    def _generate_mask(self, H, W):
        mask = torch.zeros((H, W))
        center_ratio = 0.1
        center_lines = int(H * center_ratio)
        start = H // 2 - center_lines // 2
        end = H // 2 + center_lines // 2
        mask[start:end, :] = 1.0
        remaining = int(H * self.sampling_rate) - center_lines
        if remaining > 0:
            step = max(1, (H - center_lines) // remaining)
            indices = list(range(0, start)) + list(range(end, H))
            sampled = indices[::step][:remaining]
            for i in sampled:
                mask[i, :] = 1.0
        return mask

# ---------- 2. 指标计算 ----------
def rel_error(gt, rec):
    """相对 L2 误差：||gt - rec|| / ||gt||"""
    gt_abs = torch.abs(gt)
    rec_abs = torch.abs(rec)
    return torch.norm(gt_abs - rec_abs) / (torch.norm(gt_abs) + 1e-9)

# ---------- 3. 加载模型 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LIPN_pro(K=5)                     # K 需与训练时一致
checkpoint_path = r"C:\Users\Administrator\Desktop\论文\变分不等式第二篇\数值实验\checkpoints\LIPN_pro_best.pth"
# checkpoint = torch.load(checkpoint_path, map_location=device)
# checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
# model.load_state_dict(checkpoint)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])  # 关键修改：提取模型权重
model.to(device)
model.eval()

# ---------- 4. 准备验证集 ----------
data_dir = "IXI-dataset-master/size64"
full_dataset = MRIDataset(data_dir, sampling_rate=0.3, mask_type='cartesian', normalize_to_01=True)
# 使用与训练时相同的划分（固定种子）
torch.manual_seed(42)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
_, val_dataset = random_split(full_dataset, [train_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 固定验证样本（取出所有样本，避免每次运行随机）
val_samples = []
for img_c, mask in val_loader:
    val_samples.append((img_c.squeeze(0), mask.squeeze(0)))   # 去掉 batch 维

# ---------- 5. 噪声水平与随机种子 ----------
noise_levels = [0.00, 0.05, 0.10, 0.20, 0.40, 0.80]
num_seeds = 5
seeds = [2026 + i for i in range(num_seeds)]

# 存储结果：errors[noise_idx][seed_idx] = 该噪声下所有样本的平均误差
errors = {nl: [] for nl in noise_levels}

with torch.no_grad():
    for nl in noise_levels:
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            total_rel_err = 0.0
            count = 0
            for img_c, mask in val_samples:
                img_c = img_c.unsqueeze(0).to(device)    # [1, H, W]
                mask = mask.unsqueeze(0).to(device)      # [1, H, W]
                # 生成带噪 k-space
                k_full = fft2c(img_c)
                if nl > 0:
                    noise_real = torch.randn_like(k_full.real) * nl
                    noise_imag = torch.randn_like(k_full.imag) * nl
                    k0 = k_full + torch.complex(noise_real, noise_imag)
                else:
                    k0 = k_full
                # 零填充重建作为输入 x
                x_und = ifft2c(k0 * mask)                # 复数图像
                x_input = torch.view_as_real(x_und).permute(0, 3, 1, 2).contiguous()  # [B,2,H,W]
                # 重建
                out = model(x_input, k0, mask)
                rec = torch.view_as_complex(out.permute(0, 2, 3, 1).contiguous())
                err = rel_error(img_c, rec).item()
                total_rel_err += err
                count += 1
            avg_err = total_rel_err / count
            errors[nl].append(avg_err)

# ---------- 6. 计算均值和标准差 ----------
mean_err = []
std_err = []
for nl in noise_levels:
    vals = errors[nl]
    mean_err.append(np.mean(vals))
    std_err.append(np.std(vals))

# ---------- 7. 线性拟合 ----------
X = np.array(noise_levels).reshape(-1, 1)
y = np.array(mean_err)
reg = LinearRegression().fit(X, y)
slope = reg.coef_[0]
intercept = reg.intercept_
r2 = reg.score(X, y)
print(f"线性拟合: 斜率 = {slope:.4f}, 截距 = {intercept:.4f}, R² = {r2:.4f}")

# 理论收缩因子（从拟合斜率反推）
gamma_fit = 1 - 1/slope
print(f"根据拟合斜率推算的收缩因子 γ' = {gamma_fit:.4f}")

# ---------- 8. 高级绘图（替换原绘图部分）----------
# ---------- 8. 高级绘图（白色背景，无标题，无网格）----------
plt.style.use('seaborn-v0_8-whitegrid')   # 白色背景 + 浅灰网格（网格稍后关闭）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

fig, ax = plt.subplots(figsize=(8, 5))

# 实验数据点（带误差棒）
ax.errorbar(noise_levels, mean_err, yerr=std_err, fmt='o',
            color='#1f77b4', capsize=4, capthick=1.5,
            markersize=8, elinewidth=1.5, markeredgecolor='w',
            label='Experimental (mean ± std)')

# 填充标准差区域
ax.fill_between(noise_levels,
                np.array(mean_err) - np.array(std_err),
                np.array(mean_err) + np.array(std_err),
                color='#1f77b4', alpha=0.2, label='±1 std')

# 拟合直线
x_fit = np.linspace(0, max(noise_levels), 100)
y_fit = slope * x_fit + intercept
ax.plot(x_fit, y_fit, 'r--', linewidth=2, label=f'Linear fit (slope={slope:.2f}, R²={r2:.3f})')

# 理论界
theoretical = np.array(noise_levels) / (1 - gamma_fit)
ax.plot(noise_levels, theoretical, 'g-.', linewidth=2, label=f'Theory ε/(1-γ) with γ={gamma_fit:.3f}')

# 拟合公式文本框
ax.text(0.6, 0.85, f'y = {slope:.2f}·ε + {intercept:.3f}\nR² = {r2:.4f}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 轴标签
ax.set_xlabel('Noise standard deviation ε', fontsize=12)
ax.set_ylabel('Relative Reconstruction Error', fontsize=12)
# ❌ 标题已删除

# 坐标轴范围
ax.set_xlim(-0.05, max(noise_levels)+0.05)
ax.set_ylim(bottom=0)

# 图例（保留，方便阅读）
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)

# 关闭网格
ax.grid(False)

plt.tight_layout()
plt.savefig('6.3噪声收敛性分析.png', dpi=600, facecolor='white')  # 显式指定白色背景
plt.show()