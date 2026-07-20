import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random
import h5py
from skimage.metrics import structural_similarity as ssim

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


# ==========================================
# 1. fastMRI 专属 Dataset 类（保持优秀架构）
# ==========================================
class FastMRIDataset(Dataset):
    def __init__(self, data_dir, file_list, sampling_rate=0.3, mask_type='cartesian',
                 noise_std=0.0, target_shape=(320, 320)):
        self.data_dir = data_dir
        self.file_list = file_list
        self.sampling_rate = sampling_rate
        self.mask_type = mask_type
        self.noise_std = noise_std
        self.target_shape = target_shape
        self.examples = []

        for fname in self.file_list:
            fpath = os.path.join(self.data_dir, fname)
            with h5py.File(fpath, 'r') as f:
                num_slices = f['kspace'].shape[0]
                for slice_id in range(num_slices):
                    self.examples.append((fpath, slice_id))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        fpath, slice_id = self.examples[idx]
        with h5py.File(fpath, 'r') as f:
            kspace_np = f['kspace'][slice_id]

        kspace_torch = torch.from_numpy(kspace_np).to(torch.complex64)
        img_full = ifft2c(kspace_torch)

        # 中心裁剪去除过采样
        H, W = img_full.shape
        th, tw = self.target_shape
        start_h = (H - th) // 2
        start_w = (W - tw) // 2
        img_c = img_full[start_h:start_h + th, start_w:start_w + tw]

        # 峰值归一化
        scale = torch.max(torch.abs(img_c)) + 1e-8
        img_c = img_c / scale

        mask = self._generate_mask(th, tw)
        k_full = fft2c(img_c)
        k0 = k_full * mask

        if self.noise_std > 0:
            noise_real = torch.randn_like(k0.real) * self.noise_std
            noise_imag = torch.randn_like(k0.imag) * self.noise_std
            k0 = k0 + torch.complex(noise_real, noise_imag)

        x_und = ifft2c(k0)
        x_real = torch.view_as_real(x_und).permute(2, 0, 1).contiguous()
        y_real = torch.view_as_real(img_c).permute(2, 0, 1).contiguous()

        return {'input': x_real, 'target': y_real, 'k0': k0, 'mask': mask, 'scale': scale}

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


# ==========================================
# 2. 评估指标计算
# ==========================================
def calculate_metrics(gt, rec, scale=None):
    if hasattr(gt, 'detach'):
        gt = gt.detach().cpu().numpy()
        rec = rec.detach().cpu().numpy()
    gt_abs, rec_abs = np.abs(gt), np.abs(rec)

    # 💡 如果给出了缩放因子，恢复到原始幅度
    if scale is not None:
        if isinstance(scale, torch.Tensor):
            scale = scale.cpu().numpy()
        gt_abs = gt_abs * scale
        rec_abs = rec_abs * scale

    mse = np.mean((gt_abs - rec_abs) ** 2)
    nmse = mse / (np.mean(gt_abs ** 2) + 1e-9)
    mae = np.mean(np.abs(gt_abs - rec_abs))

    data_range = gt_abs.max()          # 此时就是原始峰值，有物理意义
    psnr = 20 * np.log10(data_range / (np.sqrt(mse) + 1e-9))
    ssim_val = ssim(gt_abs, rec_abs, data_range=data_range,
                    gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    rel_error = np.linalg.norm(gt_abs - rec_abs) / (np.linalg.norm(gt_abs) + 1e-9)
    return {'mse': mse, 'nmse': nmse, 'mae': mae, 'psnr': psnr, 'ssim': ssim_val, 'rel_error': rel_error}

# ==========================================
# 3. 训练与绘图函数
# ==========================================
def train_model(model, train_loader, val_loader, epochs, lr, device, name, best_metric='psnr'):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = {'train_loss': [], 'val_psnr': [], 'val_ssim': [], 'val_nmse': []}
    best_state_dict, best_epoch = None, 0
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

        model.eval()
        psnr_list, ssim_list, nmse_list = [], [], []
        with torch.no_grad():
            for batch_idx, b in enumerate(val_loader):
                x = b['input'].to(device)
                y = b['target'].to(device)
                k0 = b['k0'].to(device)
                m = b['mask'].to(device)
                out = model(x, k0, m)

                # 💡 核心修正：修复原诊断代码中的内存/显存泄露隐患
                # if batch_idx == 0 and (ep + 1) % 5 == 0:
                #     print(
                #         f"-> Epoch {ep + 1} [{name}] Target Max: {y.max().item():.4f} | Output Max: {out.max().item():.4f}")
                #     diag_gt = torch.view_as_complex(y[0].permute(1, 2, 0).contiguous()).cpu().numpy()
                #     diag_pred = torch.view_as_complex(out[0].permute(1, 2, 0).contiguous()).cpu().numpy()

                    # plt.figure(figsize=(8, 4))
                    # plt.subplot(1, 2, 1);
                    # plt.imshow(np.abs(diag_gt), cmap='gray');
                    # plt.title('GT');
                    # plt.axis('off')
                    # plt.subplot(1, 2, 2);
                    # plt.imshow(np.abs(diag_pred), cmap='gray');
                    # plt.title(f'{name} Pred');
                    # plt.axis('off')
                    # plt.tight_layout()
                    # plt.show()
                    # plt.close()  # 💡 显式关闭，释放内存缓冲区

                scale = b['scale']
                rec = torch.view_as_complex(out.permute(0, 2, 3, 1).contiguous()).cpu().numpy()
                gt = torch.view_as_complex(y.permute(0, 2, 3, 1).contiguous()).cpu().numpy()
                for i in range(len(rec)):
                    met = calculate_metrics(gt[i], rec[i], scale=scale[i])
                    psnr_list.append(met['psnr'])
                    ssim_list.append(met['ssim'])
                    nmse_list.append(met['nmse'])

        avg_psnr, avg_ssim, avg_nmse = np.mean(psnr_list), np.mean(ssim_list), np.mean(nmse_list)
        history['val_psnr'].append(avg_psnr)
        history['val_ssim'].append(avg_ssim)
        history['val_nmse'].append(avg_nmse)

        current = avg_psnr if best_metric == 'psnr' else (avg_ssim if best_metric == 'ssim' else avg_nmse)
        if (maximize and current > best_value) or (not maximize and current < best_value):
            best_value, best_epoch = current, ep + 1
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
    titles = ['Training Loss', 'Validation PSNR', 'Validation SSIM', 'Validation NMSE']
    ylabels = ['Training Loss (MSE)', 'PSNR (dB)', 'SSIM', 'NMSE']
    for idx, key in enumerate(metrics_keys):
        for name, hist in histories.items():
            axes[idx].plot(epochs, hist[key], label=name)
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel(ylabels[idx])
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
            axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * max(values), f'{val:.4f}',
                           ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig('final_comparison.png', dpi=150)
    plt.show()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def visualize_all_models(model_dict, val_set, device, num_show=3):
    for model in model_dict.values(): model.eval()
    plt.figure(figsize=(20, 5 * num_show))
    for i in range(num_show):
        sample = val_set[i]
        scale = sample['scale']
        x = sample['input'].unsqueeze(0).to(device)
        k0 = sample['k0'].unsqueeze(0).to(device)
        m = sample['mask'].unsqueeze(0).to(device)
        gt_c = torch.view_as_complex(sample['target'].permute(1, 2, 0).contiguous()).cpu().numpy()
        zero_c = torch.view_as_complex(sample['input'].permute(1, 2, 0).contiguous()).cpu().numpy()
        cols = len(model_dict) + 2

        plt.subplot(num_show, cols, i * cols + 1);
        plt.imshow(np.abs(gt_c), cmap='gray');
        plt.title("Ground Truth");
        plt.axis('off')
        plt.subplot(num_show, cols, i * cols + 2);
        plt.imshow(np.abs(zero_c), cmap='gray');
        plt.title("Zero-filled");
        plt.axis('off')
        for j, (name, model) in enumerate(model_dict.items()):
            with torch.no_grad(): out = model(x, k0, m)
            rec = torch.view_as_complex(out.squeeze(0).permute(1, 2, 0).contiguous()).cpu().numpy()
            metrics = calculate_metrics(gt_c, rec, scale=scale)
            plt.subplot(num_show, cols, i * cols + 3 + j);
            plt.imshow(np.abs(rec), cmap='gray')
            plt.title(f"{name}\nPSNR: {metrics['psnr']:.2f} dB\nSSIM: {metrics['ssim']:.3f}");
            plt.axis('off')
    plt.tight_layout()
    plt.show()


# ==========================================
# 4. 主运行程序
# ==========================================
if __name__ == "__main__":
    set_seed(42)

    DATA_DIR = r"C:\Users\Administrator\Desktop\论文\变分不等式第二篇\数值实验\knee_singlecoil_val\knee_singlecoil_val\singlecoil_val"

    all_h5_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.h5')]
    print(f"在原始目录中一共找到 {len(all_h5_files)} 个患者的数据文件。")

    random.seed(42)
    random.shuffle(all_h5_files)
    split = int(0.8 * len(all_h5_files))
    train_files = all_h5_files[:split]  # 约 159 个患者
    val_files = all_h5_files[split:]  # 约 40 个患者
    print(f"成功构建子集：训练集包含 {len(train_files)} 个患者，验证集包含 {len(val_files)} 个患者。")

    train_set = FastMRIDataset(DATA_DIR, train_files, sampling_rate=0.3, mask_type='cartesian', noise_std=0)
    val_set = FastMRIDataset(DATA_DIR, val_files, sampling_rate=0.3, mask_type='cartesian', noise_std=0)
    print(f"总切片数量 -> 训练集: {len(train_set)} 张, 验证集: {len(val_set)} 张")

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainable_models = {
        # "LIPN": LIPN(K=5),
        # "DCCNN": DCCNN(n=5),
        # "ComplexUNet": ComplexUNet(),
        # "LGD_Net": LGD_Net(num_iters=5),
        # "MoDL": MoDL(K=5),
        # "ISTANet": ISTANet(K=5),
        "VarNet": VarNet(K=5),
        # "LIPN_pro": LIPN_pro(K=5),
    }
    nontrainable_models = {
        "ADMM_TV": ADMM_TV(iter_num=50, rho=0.1, lam=0.01),
    #     "CPNN": CPNN(K=5, eta=0.5, lam=0.1, noise_std=15 / 255)
    }


    all_models = {**trainable_models, **nontrainable_models}
    for name, model in all_models.items(): model.to(device)

    EPOCHS = 30
    LR = 1e-4
    histories = {}

    for name, model in trainable_models.items():
        print(f"\n===== Training {name} =====")
        hist = train_model(model, train_loader, val_loader, EPOCHS, LR, device, name)
        histories[name] = hist

    if histories:
        plot_training_history(histories, list(histories.keys()))

    # 💡 核心修正：重新将 val_loader 置于外层循环，防止后续开启噪声/随机采样时发生学术不公平对比
    results = {name: {"mse": [], "nmse": [], "mae": [], "psnr": [], "ssim": [], "rel_error": []} for name in all_models}

    print("\n===== Final Evaluation on Validation Set =====")
    for name, model in all_models.items(): model.eval()

    with torch.no_grad():
        for b in val_loader:
            x = b['input'].to(device)
            y = b['target'].to(device)
            k0 = b['k0'].to(device)
            m = b['mask'].to(device)
            scale = b['scale']  # ← 添加这行

            for name, model in all_models.items():
                out = model(x, k0, m)
                rec = torch.view_as_complex(out.permute(0, 2, 3, 1).contiguous()).cpu().numpy()
                gt = torch.view_as_complex(y.permute(0, 2, 3, 1).contiguous()).cpu().numpy()

                for i in range(len(rec)):
                    met = calculate_metrics(gt[i], rec[i], scale=scale[i])  # ← 传入 scale[i]
                    for key in results[name].keys():
                        results[name][key].append(met[key])

    # 计算聚合平均值
    final_results = {}
    for name in all_models:
        final_results[name] = {key: np.mean(results[name][key]) for key in results[name].keys()}

    print("\n===== Detailed Metrics =====")
    print("{:<15s} {:>8s} {:>10s} {:>10s} {:>8s} {:>10s} {:>12s}".format("Model", "MSE", "NMSE", "MAE", "PSNR", "SSIM",
                                                                         "Rel.Error"))
    for name in all_models:
        r = final_results[name]
        print("{:<15s} {:8.6f} {:10.6f} {:10.6f} {:8.2f} {:10.4f} {:12.6f}".format(name, r['mse'], r['nmse'], r['mae'],
                                                                                   r['psnr'], r['ssim'],
                                                                                   r['rel_error']))

    plot_final_comparison(final_results, metrics=['psnr', 'ssim', 'nmse', 'mse'])
    visualize_all_models(all_models, val_set, device, num_show=3)