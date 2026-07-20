import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')          # 非交互式后端，避免弹窗
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim
import random

# 导入你的模型（请根据实际路径调整）
from LIPN_pro import LIPN_pro
from MoDL import MoDL
from VarNet import VarNet
# from ADMM_TV import ADMM_TV     # 如果不用可注释掉
# from CPNN import CPNN
from fft_utils import fft2c, ifft2c

# -------------------- 随机种子 --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------- 数据集 --------------------
class MRIDataset(Dataset):
    def __init__(self, data_dir, sampling_rate=0.3, mask_type='cartesian',
                 noise_std=0.0, normalize_to_01=True):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.sampling_rate = sampling_rate
        self.mask_type = mask_type
        self.noise_std = noise_std
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
        k_full = fft2c(img_c)
        k0 = k_full * mask

        if self.noise_std > 0:
            noise_real = torch.randn_like(k0.real) * self.noise_std
            noise_imag = torch.randn_like(k0.imag) * self.noise_std
            k0 = k0 + torch.complex(noise_real, noise_imag)

        x_und = ifft2c(k0)
        x_real = torch.view_as_real(x_und).permute(2, 0, 1).contiguous()
        y_real = torch.view_as_real(img_c).permute(2, 0, 1).contiguous()
        return {
            'input': x_real,
            'target': y_real,
            'k0': k0,
            'mask': mask
        }

    def _generate_mask(self, H, W):
        if self.mask_type == 'cartesian':
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
        else:
            raise ValueError("当前只实现了 cartesian 掩膜")

# -------------------- 指标 --------------------
def calculate_metrics(gt, rec):
    gt_abs = np.abs(gt)
    rec_abs = np.abs(rec)

    mse = np.mean((gt_abs - rec_abs) ** 2)
    nmse = mse / (np.mean(gt_abs ** 2) + 1e-9)
    mae = np.mean(np.abs(gt_abs - rec_abs))
    data_range = 1.0   # 归一化到 [0,1]
    psnr = 20 * np.log10(data_range / (np.sqrt(mse) + 1e-9))
    ssim_val = ssim(gt_abs, rec_abs, data_range=data_range,
                    gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    rel_error = np.linalg.norm(gt_abs - rec_abs) / (np.linalg.norm(gt_abs) + 1e-9)
    return {'mse': mse, 'nmse': nmse, 'mae': mae, 'psnr': psnr, 'ssim': ssim_val, 'rel_error': rel_error}

# -------------------- 评估函数 --------------------
def evaluate_models_on_loader(models_dict, loader, device):
    all_metrics = {name: {m: [] for m in ['mse','nmse','mae','psnr','ssim','rel_error']}
                   for name in models_dict}
    with torch.no_grad():
        for batch in loader:
            x = batch['input'].to(device)
            y = batch['target'].to(device)
            k0 = batch['k0'].to(device)
            m = batch['mask'].to(device)
            for name, model in models_dict.items():
                out = model(x, k0, m)
                rec = torch.view_as_complex(out.permute(0,2,3,1).contiguous()).cpu().numpy()
                gt = torch.view_as_complex(y.permute(0,2,3,1).contiguous()).cpu().numpy()
                for i in range(len(rec)):
                    met = calculate_metrics(gt[i], rec[i])
                    for key in all_metrics[name]:
                        all_metrics[name][key].append(met[key])
    avg = {}
    for name in all_metrics:
        avg[name] = {k: np.mean(all_metrics[name][k]) for k in all_metrics[name]}
    return avg

# -------------------- 绘图 --------------------
def plot_metric_vs_noise(results, noise_levels, metric='psnr', title_suffix=''):
    plt.figure(figsize=(10, 6))
    model_names = list(results[noise_levels[0]].keys())
    for model in model_names:
        values = [results[ns][model][metric] for ns in noise_levels]
        plt.plot(noise_levels, values, marker='o', linewidth=2, label=model)
    plt.xlabel('Noise Standard Deviation (k-space)')
    plt.ylabel(metric.upper())
    plt.title(f'{metric.upper()} vs Noise Level {title_suffix}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_path = f'noise_{metric}{title_suffix.replace(" ", "_")}.png'
    plt.savefig(save_path, dpi=150)
    plt.show()

# -------------------- 模型实例化 --------------------
def build_model_from_name(name, noise_std=None):
    if name == 'LIPN_pro':
        return LIPN_pro(K=7, noise_std=noise_std if noise_std else 15/255)
    elif name == 'MoDL':
        return MoDL(K=5)
    elif name == 'VarNet':
        return VarNet(K=5)
    # elif name == 'ADMM_TV':
    #     return ADMM_TV(iter_num=50, rho=0.1, lam=0.01)
    # elif name == 'CPNN':
    #     if noise_std is None: noise_std = 15/255
    #     return CPNN(K=5, eta=0.5, lam=0.1, noise_std=noise_std)
    else:
        raise ValueError(f"未知模型: {name}")

# -------------------- 主程序 --------------------
if __name__ == "__main__":
    set_seed(42)
    DATA_DIR = "IXI-dataset-master/size64"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4
    SAMPLING_RATE = 0.3
    MASK_TYPE = 'cartesian'

    # 要测试的噪声水平（可根据你的权重文件修改）
    noise_levels = [15/255 * k for k in [0.5, 1, 2, 3, 4]]
    # 需要对比的模型
    model_names = ['LIPN_pro', 'MoDL', 'VarNet']

    # 评估模式：True = 每个测试噪声用对应训练噪声的权重；False = 跨噪声泛化
    use_matched_weights = True
    fixed_train_noise = 15/255   # 仅在 use_matched_weights=False 时生效

    # ---------- 固定验证集划分 ----------
    base_dataset = MRIDataset(DATA_DIR, sampling_rate=SAMPLING_RATE,
                              mask_type=MASK_TYPE, noise_std=0.0, normalize_to_01=True)
    n_total = len(base_dataset)
    n_val = int(0.2 * n_total)
    indices = list(range(n_total))
    np.random.shuffle(indices)
    val_indices = indices[:n_val]   # 固定验证集索引

    all_results = {}
    for target_noise in noise_levels:
        print(f"\n========== 测试噪声: {target_noise:.4f} ==========")
        noisy_dataset = MRIDataset(DATA_DIR, sampling_rate=SAMPLING_RATE,
                                   mask_type=MASK_TYPE, noise_std=target_noise,
                                   normalize_to_01=True)
        val_subset = torch.utils.data.Subset(noisy_dataset, val_indices)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

        models_dict = {}
        for name in model_names:
            model = build_model_from_name(name, noise_std=target_noise)

            # 权重文件路径（请确认实际文件名格式）
            if use_matched_weights:
                weight_file = f"checkpoints/{name}_best.pth"
            else:
                weight_file = f"checkpoints/{name}_best.pth"

            if os.path.exists(weight_file):
                state_dict = torch.load(weight_file, map_location=DEVICE)
                model.load_state_dict(state_dict)
                print(f"  ✓ 加载: {weight_file}")
            else:
                raise FileNotFoundError(f"权重文件不存在: {weight_file}")

            model.to(DEVICE)
            model.eval()
            models_dict[name] = model

        avg_metrics = evaluate_models_on_loader(models_dict, val_loader, DEVICE)
        all_results[target_noise] = avg_metrics

        # 打印表格
        print(f"\n测试噪声 {target_noise:.4f} 下的平均指标:")
        print("{:<12s} {:>8s} {:>10s} {:>8s} {:>8s}".format("Model", "MSE", "PSNR", "SSIM", "NMSE"))
        for name in avg_metrics:
            r = avg_metrics[name]
            print("{:<12s} {:8.6f} {:10.2f} {:8.4f} {:8.6f}".format(
                name, r['mse'], r['psnr'], r['ssim'], r['nmse']))

    # 绘制曲线
    plot_metric_vs_noise(all_results, noise_levels, metric='psnr',
                         title_suffix='(Matched Weights)' if use_matched_weights else f'(Fixed Train Noise {fixed_train_noise:.4f})')
    plot_metric_vs_noise(all_results, noise_levels, metric='ssim',
                         title_suffix='(Matched Weights)' if use_matched_weights else f'(Fixed Train Noise {fixed_train_noise:.4f})')
    plot_metric_vs_noise(all_results, noise_levels, metric='nmse',
                         title_suffix='(Matched Weights)' if use_matched_weights else f'(Fixed Train Noise {fixed_train_noise:.4f})')

    np.save('noise_evaluation_results.npy', all_results)
    print("\n✅ 所有评估完成，结果已保存。")