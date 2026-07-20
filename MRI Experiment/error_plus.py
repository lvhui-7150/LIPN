import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from LIPN_pro import LIPN_pro
from fft_utils import fft2c, ifft2c

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===============================
# 1. 数据集（保持不变）
# ===============================
class MRIDataset(Dataset):
    def __init__(self, data_dir):
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.data_dir = data_dir

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.data_dir, self.file_list[idx]))
        if not torch.is_complex(data):
            data = data.float()
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            data = data.to(torch.complex64)

        H, W = data.shape
        mask = self._generate_mask(H, W)
        k_full = fft2c(data)
        k0 = k_full * mask
        x_und = ifft2c(k0)

        return {
            'input': torch.view_as_real(x_und).permute(2, 0, 1),
            'target': torch.view_as_real(data).permute(2, 0, 1),
            'k0': k0,
            'mask': mask
        }

    def _generate_mask(self, H, W):
        mask = torch.zeros((H, W))
        mask[:, W // 2 - 4:W // 2 + 4] = 1
        rand = (torch.rand((H, W)) < 0.25).float()
        return torch.maximum(mask, rand)


# ===============================
# 2. 去噪器：严格 FNE
# ===============================
class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(nn.Conv2d(2, 64, 3, padding=1)),
            nn.ReLU(),
            nn.utils.parametrizations.spectral_norm(nn.Conv2d(64, 64, 3, padding=1)),
            nn.ReLU(),
            nn.utils.parametrizations.spectral_norm(nn.Conv2d(64, 2, 3, padding=1))
        )

    def forward(self, z):
        z_stack = torch.stack([z.real, z.imag], dim=1)
        out = self.net(z_stack)
        res = torch.complex(out[:, 0], out[:, 1])
        return 0.5 * (z + res)


def compute_error_to_gt(model, loader, device, noise_std, seed):
    """计算指定种子下的重建误差（L2范数）"""
    model.eval()
    errs = []
    with torch.no_grad():
        for b in loader:
            x, y, k0, m = b['input'].to(device), b['target'].to(device), \
                          b['k0'].to(device), b['mask'].to(device)
            out = model(x, k0, m, noise_std=noise_std, seed=seed)
            err = torch.norm((out - y).reshape(out.shape[0], -1), dim=1)
            errs.extend(err.cpu().numpy())
    return np.mean(errs)


def train_model(model, loader, epochs, device):
    opt = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        for b in loader:
            x, y, k0, m = b['input'].to(device), b['target'].to(device), \
                          b['k0'].to(device), b['mask'].to(device)
            out = model(x, k0, m, noise_std=0.0)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch {ep + 1}/{epochs} finished.")
    return model


# ===============================
# 5. 主程序（增强分析）
# ===============================
if __name__ == "__main__":
    DATA_DIR = "IXI-dataset-master/size64"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MRIDataset(DATA_DIR)
    train_size = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=4, shuffle=False)

    print(">>> Step 1: Training base model ...")
    model = LIPN_pro(K=5).to(device)
    model = train_model(model, train_loader, epochs=15, device=device)

    # ---------- 稳定性实验（多次随机重复）----------
    print(">>> Step 2: Stability analysis with multiple seeds ...")
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
    n_seeds = 5  # 每个噪声水平重复实验次数
    seeds = range(2024, 2024 + n_seeds)  # 固定种子序列

    mean_errors = []
    std_errors = []
    for ns in noise_levels:
        errors = []
        for s in seeds:
            err = compute_error_to_gt(model, val_loader, device, noise_std=ns, seed=s)
            errors.append(err)
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        mean_errors.append(mean_err)
        std_errors.append(std_err)
        print(f"Noise std: {ns:.3f} | Mean error: {mean_err:.4f} | Std: {std_err:.4f}")

    # ---------- 理论界计算 ----------
    # 这里我们假设一个合理的收缩因子 γ，例如通过实验估计或取典型值
    # 为展示方法，假定 γ ≈ 0.92 （可根据实际情况调整）
    gamma_est = 0.92
    theoretical_bound = [ns / (1 - gamma_est) for ns in noise_levels]  # ε/(1-γ)

    # ---------- 线性拟合 ----------
    fit_coeff = np.polyfit(noise_levels, mean_errors, 1)
    fit_line = np.polyval(fit_coeff, noise_levels)
    slope, intercept = fit_coeff
    # 计算 R²
    residuals = np.array(mean_errors) - fit_line
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((np.array(mean_errors) - np.mean(mean_errors))**2)
    r_squared = 1 - ss_res / ss_tot

    # ---------- 绘图 ----------
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 7))

    # 实验点 + 误差棒
    plt.errorbar(noise_levels, mean_errors, yerr=std_errors,
                 fmt='o', color='#1f77b4', capsize=5, markersize=8,
                 label='Empirical steady-state error (mean ± std)')

    # 理论界曲线
    plt.plot(noise_levels, theoretical_bound, 'g--', linewidth=2,
             label=f'Theoretical bound ($\\gamma$={gamma_est})')

    # 线性拟合曲线
    plt.plot(noise_levels, fit_line, 'r-', linewidth=2,
             label=f'Linear fit (slope={slope:.2f}, $R^2$={r_squared:.3f})')

    # 可选：基于实验斜率反推出的有效 γ 曲线（斜率为 1/(1-γ')）
    if slope > 0:
        gamma_emp = 1 - 1/slope
        # 确保在 (0,1) 内才绘制
        if 0 < gamma_emp < 1:
            bound_emp = [ns / (1 - gamma_emp) for ns in noise_levels]
            plt.plot(noise_levels, bound_emp, 'm:', linewidth=2,
                     label=f'Fitted bound ($\\gamma\'={gamma_emp:.3f}$)')

    plt.xlabel('Perturbation magnitude ε (noise std)', fontsize=14)
    plt.ylabel('Reconstruction error ||z^K - z*||', fontsize=14)
    plt.title('Steady-state stability verification of SPLP-VI', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('stability_theorem_verification.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 输出拟合结果
    print(f"\nLinear regression: error = {intercept:.4f} + {slope:.4f} * ε")
    print(f"R² = {r_squared:.4f}")
    if slope > 0 and 0 < 1 - 1/slope < 1:
        print(f"Estimated contraction factor γ' = {1 - 1/slope:.4f}")
    else:
        print("Note: Estimated γ' is out of (0,1), using assumed γ = 0.92 for bound.")