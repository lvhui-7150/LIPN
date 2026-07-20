import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
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
    data_range = gt_abs.max()
    psnr = 20 * np.log10(data_range / (np.sqrt(mse) + 1e-9))

    ssim_val = ssim(gt_abs, rec_abs, data_range=data_range,
                    gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

    rel_error = np.linalg.norm(gt_abs - rec_abs) / (np.linalg.norm(gt_abs) + 1e-9)
    return {'mse': mse, 'nmse': nmse, 'mae': mae, 'psnr': psnr, 'ssim': ssim_val, 'rel_error': rel_error}


def train_model(model, train_loader, val_loader, epochs, lr, device, name, best_metric='psnr'):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
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

        # 验证
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

        scheduler.step()  # 更新学习率
        print(f"{name} Epoch {ep+1}/{epochs} | Loss {avg_train_loss:.6f} | PSNR {avg_psnr:.2f} | SSIM {avg_ssim:.4f}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)
        print(f">>> {name} 最优模型已加载 (epoch {best_epoch}, {best_metric} = {best_value:.4f})")
    return history, best_state_dict  # 同时返回最优状态字典


def evaluate_models(models_dict, val_loader, device):
    """对所有模型在给定验证集上进行评估，返回平均指标字典"""
    results = {name: {"mse": [], "nmse": [], "mae": [], "psnr": [], "ssim": [], "rel_error": []}
               for name in models_dict}
    with torch.no_grad():
        for batch in val_loader:
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
                    for key in results[name]:
                        results[name][key].append(met[key])
    avg_results = {}
    for name in models_dict:
        avg_results[name] = {key: np.mean(results[name][key]) for key in results[name]}
    return avg_results


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_trainable_models():
    """返回可训练模型的字典（初始实例）"""
    return {
        "VarNet": VarNet(K=5),
        "LIPN_pro": LIPN_pro(K=5),
    }


def create_nontrainable_models(noise_std):
    """返回非训练模型的字典，需要传入当前噪声水平以设置CPNN等参数"""
    return {
        # "ADMM_TV": ADMM_TV(iter_num=50, rho=0.1, lam=0.01),
        # "CPNN": CPNN(K=5, eta=0.5, lam=0.1, noise_std=noise_std),
    }


def plot_noise_curves(all_results, noise_levels, metric='psnr'):
    """绘制不同噪声水平下各模型的指标曲线"""
    plt.figure(figsize=(10, 6))
    for model_name in all_results[noise_levels[0]].keys():
        values = [all_results[ns][model_name][metric] for ns in noise_levels]
        plt.plot(noise_levels, values, marker='o', label=model_name)
    plt.xlabel('Noise Standard Deviation (k-space)')
    plt.ylabel(metric.upper())
    plt.title(f'{metric.upper()} vs Noise Level')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'noise_curves_{metric}.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    set_seed(42)
    DATA_DIR = "IXI-dataset-master/size64"
    EPOCHS = 30
    LR = 5e-5
    BATCH_SIZE = 4
    SAMPLING_RATE = 0.3
    MASK_TYPE = 'cartesian'
    noise_levels = [15/255 * k for k in [0.5, 1, 2, 3]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===================== 预先固定验证集索引 =====================
    # 构建一个无噪声的基础数据集，仅用于获取总样本数并划分索引
    base_dataset = MRIDataset(DATA_DIR, sampling_rate=SAMPLING_RATE,
                              mask_type=MASK_TYPE, noise_std=0.0,
                              normalize_to_01=True)
    n_total = len(base_dataset)
    n_val = int(0.2 * n_total)
    indices = list(range(n_total))
    np.random.shuffle(indices)
    val_indices = indices[:n_val]   # 固定的验证集索引
    train_indices = indices[n_val:] # 固定的训练集索引（可选，此处仅用于记录，实际构造使用 Subset）
    print(f"数据集总样本数: {n_total}, 验证集固定样本数: {len(val_indices)}")

    # 存储所有噪声水平下的评估结果
    all_results = {}
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    for noise_std in noise_levels:
        print(f"\n{'='*60}")
        print(f"Processing noise_std = {noise_std:.4f}")
        print(f"{'='*60}")

        # 1. 创建带当前噪声水平的数据集（同样顺序，因此索引有效）
        dataset = MRIDataset(DATA_DIR, sampling_rate=SAMPLING_RATE,
                             mask_type=MASK_TYPE, noise_std=noise_std,
                             normalize_to_01=True)
        # 使用固定索引构造训练集和验证集
        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

        # 2. 创建模型字典
        trainable_models = create_trainable_models()
        nontrainable_models = create_nontrainable_models(noise_std)
        all_models = {**trainable_models, **nontrainable_models}

        # 3. 训练所有可训练模型，并保存最优权重
        for name, model in trainable_models.items():
            print(f"\n----- Training {name} (noise_std={noise_std}) -----")
            history, best_state = train_model(model, train_loader, val_loader,
                                              EPOCHS, LR, device, name)
            # 保存最优模型权重
            if best_state is not None:
                save_path = os.path.join(save_dir, f"{name}_noise{noise_std:.4f}.pth")
                torch.save(best_state, save_path)
                print(f"最优权重已保存至: {save_path}")

        # 4. 将所有模型移到设备并设为评估模式
        for model in all_models.values():
            model.to(device)
            model.eval()

        # 5. 评估所有模型（使用同一验证集）
        print(f"\n----- Evaluating all models on noise_std={noise_std} -----")
        avg_metrics = evaluate_models(all_models, val_loader, device)
        all_results[noise_std] = avg_metrics

        # 打印当前噪声水平下的详细结果
        print(f"\nResults for noise_std = {noise_std:.4f}:")
        print("{:<15s} {:>8s} {:>10s} {:>8s} {:>8s} {:>10s}".format(
            "Model", "MSE", "NMSE", "PSNR", "SSIM", "Rel.Error"))
        for name in avg_metrics:
            r = avg_metrics[name]
            print("{:<15s} {:8.6f} {:10.6f} {:8.2f} {:10.4f} {:12.6f}".format(
                name, r['mse'], r['nmse'], r['psnr'], r['ssim'], r['rel_error']))

    # 6. 绘制各指标随噪声水平变化的曲线
    plot_noise_curves(all_results, noise_levels, metric='psnr')
    plot_noise_curves(all_results, noise_levels, metric='ssim')
    plot_noise_curves(all_results, noise_levels, metric='nmse')
    plot_noise_curves(all_results, noise_levels, metric='mse')

    # 7. 保存所有结果
    np.save('noise_experiment_results.npy', all_results)
    print("\n实验完成，所有结果已保存。")