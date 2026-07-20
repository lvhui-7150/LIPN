import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import random

# 导入自定义模型
from LIPN import LIPN
from LIPN_pro import LIPN_pro
from DCCNN_Model import DCCNN
from ADMM_TV import ADMM_TV
from ComplexUNet import ComplexUNet
from LGD import LGD_Net
from CPNN import CPNN
from MoDL import MoDL
from ISTANet import ISTANet
from VarNet import VarNet
from fft_utils import fft2c, ifft2c

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MRIDataset(Dataset):
    def __init__(self, data_dir, sampling_rate=0.3, mask_type='random',
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

        # 添加复数高斯噪声
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
        if self.mask_type == 'random':
            mask = torch.zeros((H, W))
            num_samples = int(H * W * self.sampling_rate)
            idx = torch.randperm(H * W)[:num_samples]
            mask.view(-1)[idx] = 1.0
            return mask

        elif self.mask_type == 'radial':
            mask = torch.zeros((H, W))
            center = (H // 2, W // 2)
            num_angles = int(180 * self.sampling_rate)
            angles = torch.linspace(0, np.pi, num_angles)
            for ang in angles:
                for r in range(max(H, W)):
                    x = int(center[0] + r * np.cos(ang))
                    y = int(center[1] + r * np.sin(ang))
                    if 0 <= x < H and 0 <= y < W:
                        mask[x, y] = 1.0
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                if 0 <= x + dx < H and 0 <= y + dy < W:
                                    mask[x + dx, y + dy] = 1.0
            return (mask > 0).float()

        elif self.mask_type == 'cartesian':
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
            raise ValueError(f"Unknown mask_type: {self.mask_type}")


def calculate_metrics(gt, rec):
    if hasattr(gt, 'detach'):
        gt = gt.detach().cpu().numpy()
        rec = rec.detach().cpu().numpy()
    gt_abs = np.abs(gt)
    rec_abs = np.abs(rec)

    mse = np.mean((gt_abs - rec_abs) ** 2)
    nmse = mse / (np.mean(gt_abs ** 2) + 1e-9)
    mae = np.mean(np.abs(gt_abs - rec_abs))

    # 💡 核心修正：遵循 fastMRI 国际标准，使用 Ground Truth 的最大值作为数据标准范围
    data_range = gt_abs.max()
    psnr = 20 * np.log10(data_range / (np.sqrt(mse) + 1e-9))

    ssim_val = ssim(gt_abs, rec_abs, data_range=data_range,
                    gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

    rel_error = np.linalg.norm(gt_abs - rec_abs) / (np.linalg.norm(gt_abs) + 1e-9)
    return {'mse': mse, 'nmse': nmse, 'mae': mae, 'psnr': psnr, 'ssim': ssim_val, 'rel_error': rel_error}


def train_model(model, train_loader, val_loader, epochs, lr, device, name, best_metric='psnr'):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {
        'train_loss': [], 'val_psnr': [], 'val_ssim': [], 'val_nmse': []
    }

    best_state_dict = None
    best_epoch = 0
    best_value = -float('inf') if best_metric in ['psnr', 'ssim'] else float('inf')
    maximize = best_metric in ['psnr', 'ssim']

    for ep in range(epochs):
        model.train()
        total_loss = 0
        for b in train_loader:
            x = b['input'].to(device)
            y = b['target'].to(device)
            k0 = b['k0'].to(device)
            m = b['mask'].to(device)

            out = model(x, k0, m)
            loss = loss_fn(out, y)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * x.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_train_loss)

        # 验证步
        model.eval()
        psnr_list, ssim_list, nmse_list = [], [], []
        with torch.no_grad():
            for b in val_loader:
                x = b['input'].to(device)
                y = b['target'].to(device)
                k0 = b['k0'].to(device)
                m = b['mask'].to(device)
                out = model(x, k0, m)

                rec = torch.view_as_complex(out.permute(0, 2, 3, 1).contiguous()).cpu().numpy()
                gt = torch.view_as_complex(y.permute(0, 2, 3, 1).contiguous()).cpu().numpy()

                for i in range(len(rec)):
                    met = calculate_metrics(gt[i], rec[i])
                    psnr_list.append(met['psnr'])
                    ssim_list.append(met['ssim'])
                    nmse_list.append(met['nmse'])

        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        avg_nmse = np.mean(nmse_list)
        history['val_psnr'].append(avg_psnr)
        history['val_ssim'].append(avg_ssim)
        history['val_nmse'].append(avg_nmse)

        current = avg_psnr if best_metric == 'psnr' else (avg_ssim if best_metric == 'ssim' else avg_nmse)

        if (maximize and current > best_value) or (not maximize and current < best_value):
            best_value = current
            best_epoch = ep + 1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"{name} Epoch {ep + 1}/{epochs} | Loss {avg_train_loss:.6f} | PSNR {avg_psnr:.2f} | SSIM {avg_ssim:.4f}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)
        print(f">>> {name} 最优模型已加载 (epoch {best_epoch}, {best_metric} = {best_value:.4f})")
    return history


def plot_training_history(histories, model_names):
    epochs = range(1, len(list(histories.values())[0]['train_loss']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metrics_keys = ['train_loss', 'val_psnr', 'val_ssim', 'val_nmse']
    titles = ['Training Loss (MSE)', 'Validation PSNR (dB)', 'Validation SSIM', 'Validation NMSE']

    for idx, key in enumerate(metrics_keys):
        for name, hist in histories.items():
            axes[idx].plot(epochs, hist[key], label=name)
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel(titles[idx].split()[-1])
        axes[idx].set_title(titles[idx])
        axes[idx].legend()
        axes[idx].grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()


def plot_final_comparison(results, metrics=['psnr', 'ssim', 'nmse', 'mse']):
    model_names = list(results.keys())
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5))
    if num_metrics == 1: axes = [axes]

    for idx, metric in enumerate(metrics):
        values = [results[name][metric] for name in model_names]
        bars = axes[idx].bar(model_names, values, color='skyblue')
        axes[idx].set_title(f'Comparison of {metric.upper()}')
        axes[idx].set_ylabel(metric.upper())
        axes[idx].grid(True, axis='y')
        for bar, val in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(values),
                           f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('final_comparison.png', dpi=150)
    plt.show()


def set_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def visualize_all_models(model_dict, val_set, device, num_show=3):
    plt.figure(figsize=(20, 5 * num_show))
    for i in range(num_show):
        sample = val_set[i]
        x = sample['input'].unsqueeze(0).to(device)
        k0 = sample['k0'].unsqueeze(0).to(device)
        m = sample['mask'].unsqueeze(0).to(device)

        gt_c = torch.view_as_complex(sample['target'].permute(1, 2, 0).contiguous()).cpu().numpy()
        zero_c = torch.view_as_complex(sample['input'].permute(1, 2, 0).contiguous()).cpu().numpy()

        cols = len(model_dict) + 2
        plt.subplot(num_show, cols, i * cols + 1)
        plt.imshow(np.abs(gt_c), cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')

        plt.subplot(num_show, cols, i * cols + 2)
        plt.imshow(np.abs(zero_c), cmap='gray')
        plt.title("Zero-filled")
        plt.axis('off')

        for j, (name, model) in enumerate(model_dict.items()):
            with torch.no_grad():
                out = model(x, k0, m)
            rec = torch.view_as_complex(out.squeeze(0).permute(1, 2, 0).contiguous()).cpu().numpy()
            metrics = calculate_metrics(gt_c, rec)
            plt.subplot(num_show, cols, i * cols + 3 + j)
            plt.imshow(np.abs(rec), cmap='gray')
            plt.title(f"{name}\nPSNR: {metrics['psnr']:.2f}\nSSIM: {metrics['ssim']:.3f}")
            plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    set_seed(13)
    DATA_DIR = "IXI-dataset-master/size64"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义遍历的采样率空间 (0.1: 10倍加速, 0.2: 5倍加速, 0.3: 3.3倍加速)
    sampling_rates = [0.1, 0.15, 0.2, 0.25, 0.3]
    model_names = ["LIPN_pro", "MoDL", "VarNet"]

    # 存储性能数据: results[model_name] = [psnr_at_sr1, psnr_at_sr2, ...]
    results = {name: [] for name in model_names}

    for sr in sampling_rates:
        print(f"\n========== 正在测试采样率: {sr:.2f} (加速比: {1 / sr:.1f}x) ==========")

        # 加载对应采样率的数据集 (保持噪声水平不变，或根据需要调整)
        dataset = MRIDataset(DATA_DIR, sampling_rate=sr, mask_type='cartesian',
                             noise_std=15 / 255, normalize_to_01=True)
        val_loader = DataLoader(dataset, batch_size=4, shuffle=False)

        for name in model_names:
            # 1. 实例化模型
            if name == "LIPN_pro":
                model = LIPN_pro(K=5, noise_std=15 / 255)
            elif name == "MoDL":
                model = MoDL(K=5)
            elif name == "VarNet":
                model = VarNet(K=5)

            # 2. 加载对应的最优权重
            weight_path = f"checkpoints/{name}_best.pth"
            if os.path.exists(weight_path):
                model.load_state_dict(torch.load(weight_path))
            model.to(device).eval()

            # 3. 评估
            psnr_list = []
            with torch.no_grad():
                for b in val_loader:
                    x, k0, m = b['input'].to(device), b['k0'].to(device), b['mask'].to(device)
                    y = b['target'].to(device)
                    out = model(x, k0, m)

                    rec = torch.view_as_complex(out.permute(0, 2, 3, 1).contiguous()).cpu().numpy()
                    gt = torch.view_as_complex(y.permute(0, 2, 3, 1).contiguous()).cpu().numpy()

                    for i in range(len(rec)):
                        psnr_list.append(calculate_metrics(gt[i], rec[i])['psnr'])

            avg_psnr = np.mean(psnr_list)
            results[name].append(avg_psnr)
            print(f"模型 {name} | 采样率 {sr:.2f} | 平均 PSNR: {avg_psnr:.2f} dB")

    # 绘制对比曲线
    plt.figure(figsize=(8, 5))
    for name in model_names:
        plt.plot(sampling_rates, results[name], marker='s', linewidth=2, label=name)

    plt.title("Performance vs. Sampling Rate (Acceleration)")
    plt.xlabel("Sampling Rate")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('sampling_rate_comparison.png', dpi=300)
    plt.show()