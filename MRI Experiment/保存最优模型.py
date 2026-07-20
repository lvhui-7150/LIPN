import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import random

# 导入自定义模型（请确保这些模块存在）
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


# -------------------- 数据集定义（与原代码相同，仅修正了注释）--------------------
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


# -------------------- 评估指标（修正 data_range = 1.0）--------------------
def calculate_metrics(gt, rec):
    """gt, rec: complex numpy arrays"""
    gt_abs = np.abs(gt)
    rec_abs = np.abs(rec)

    mse = np.mean((gt_abs - rec_abs) ** 2)
    nmse = mse / (np.mean(gt_abs ** 2) + 1e-9)
    mae = np.mean(np.abs(gt_abs - rec_abs))

    # 修正：使用数据归一化范围 1.0（符合 fastMRI 标准）
    data_range = 1.0
    psnr = 20 * np.log10(data_range / (np.sqrt(mse) + 1e-9))
    ssim_val = ssim(gt_abs, rec_abs, data_range=data_range,
                    gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

    rel_error = np.linalg.norm(gt_abs - rec_abs) / (np.linalg.norm(gt_abs) + 1e-9)
    return {'mse': mse, 'nmse': nmse, 'mae': mae, 'psnr': psnr, 'ssim': ssim_val, 'rel_error': rel_error}


# -------------------- 训练函数 --------------------
def train_model(model, train_loader, val_loader, epochs, lr, device, name,
                best_metric='psnr', save_dir='checkpoints'):
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

    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"{name}_best.pth")

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
            torch.save(best_state_dict, best_model_path)
            print(f"  ✅ 保存新最优权重 (epoch {best_epoch}, {best_metric} = {best_value:.4f})")

        print(f"{name} Epoch {ep + 1}/{epochs} | Loss {avg_train_loss:.6f} | PSNR {avg_psnr:.2f} | SSIM {avg_ssim:.4f}")

    # 加载最优权重
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    print(f">>> {name} 训练完成，最优权重来自 epoch {best_epoch} ({best_metric} = {best_value:.4f})")
    return history


def plot_training_history(histories, model_names, save_path='training_curves.png'):
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
    plt.savefig(save_path, dpi=150)
    plt.show()


def set_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------- 主程序 --------------------
if __name__ == "__main__":
    set_seed(13)

    # ========== 配置区 ==========
    DATA_DIR = "IXI-dataset-master/size64"          # 数据路径
    EPOCHS = 30                                      # 训练轮数
    BATCH_SIZE = 4                                   # 批量大小
    LR = 5e-5                                       # 学习率
    SAMPLING_RATE = 0.3                              # 固定采样率
    MASK_TYPE = 'cartesian'
    NOISE_STD = 15 / 255
    VAL_SPLIT = 0.2                                  # 验证集比例
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"使用设备: {DEVICE}")

    # 加载完整数据集并划分
    full_dataset = MRIDataset(
        DATA_DIR,
        sampling_rate=SAMPLING_RATE,
        mask_type=MASK_TYPE,
        noise_std=NOISE_STD,
        normalize_to_01=True
    )
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}，验证集大小: {len(val_dataset)}")

    # 定义要训练的模型（仅导入你需要的）
    models_to_train = {
        "LIPN_pro": LIPN_pro(K=5),
        # "MoDL": MoDL(K=5),
        # "VarNet": VarNet(K=5)
    }

    histories = {}
    for name, model in models_to_train.items():
        print(f"\n{'='*10} 开始训练 {name} {'='*10}")
        hist = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=EPOCHS,
            lr=LR,
            device=DEVICE,
            name=name,
            best_metric='psnr',
            save_dir='checkpoints'
        )
        histories[name] = hist

    # 绘制训练曲线
    print("\n绘制训练曲线...")
    plot_training_history(histories, models_to_train.keys())

    print("训练全部完成！权重已保存至 checkpoints/ 目录。")