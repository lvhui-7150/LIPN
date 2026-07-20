import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from torch.utils.data import Dataset, DataLoader, random_split
import random
import csv
from collections import defaultdict

# 导入自定义模块
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


# ---------- 3. 定义所有模型 ----------
def get_all_models():
    """定义所有需要测试的模型"""
    models = {
        "LIPN_pro": LIPN_pro(K=5),
        "LIPN": LIPN(K=5),
        "DCCNN": DCCNN(n=5),
        "ComplexUNet": ComplexUNet(),
        "LGD_Net": LGD_Net(num_iters=5),
        "MoDL": MoDL(K=5),
        "ISTANet": ISTANet(K=5),
        "VarNet": VarNet(K=5),
        "ADMM_TV": ADMM_TV(iter_num=50, rho=0.1, lam=0.01),
        "CPNN": CPNN(K=5, eta=0.5, lam=0.1, noise_std=15 / 255)
    }
    return models


# ---------- 4. 加载模型权重 ----------
def load_model_weights(model, model_name, checkpoint_dir='有噪声checkpoints'):
    """加载模型的预训练权重"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 尝试加载 best 模型，如果没有则加载 last 模型
    best_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
    last_path = os.path.join(checkpoint_dir, f"{model_name}_last.pth")

    checkpoint_path = None
    if os.path.exists(best_path):
        checkpoint_path = best_path
        print(f"  加载 {model_name} 的 best 模型")
    elif os.path.exists(last_path):
        checkpoint_path = last_path
        print(f"  加载 {model_name} 的 last 模型")
    else:
        print(f"  ⚠️ 未找到 {model_name} 的权重文件，跳过")
        return False

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # 处理不同的保存格式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        return True
    except Exception as e:
        print(f"  ❌ 加载 {model_name} 失败: {e}")
        return False


# ---------- 5. 单个模型的噪声分析 ----------
def analyze_model_noise(model, model_name, val_samples, noise_levels, seeds, device):
    """对单个模型进行噪声鲁棒性分析"""
    print(f"\n正在分析模型: {model_name}")

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
                    img_c = img_c.unsqueeze(0).to(device)  # [1, H, W]
                    mask = mask.unsqueeze(0).to(device)  # [1, H, W]

                    # 生成带噪 k-space
                    k_full = fft2c(img_c)
                    if nl > 0:
                        noise_real = torch.randn_like(k_full.real) * nl
                        noise_imag = torch.randn_like(k_full.imag) * nl
                        k0 = k_full + torch.complex(noise_real, noise_imag)
                    else:
                        k0 = k_full

                    # 准备输入
                    x_und = ifft2c(k0 * mask)
                    x_input = torch.view_as_real(x_und).permute(0, 3, 1, 2).contiguous()

                    # 重建
                    out = model(x_input, k0, mask)
                    rec = torch.view_as_complex(out.permute(0, 2, 3, 1).contiguous())
                    err = rel_error(img_c, rec).item()

                    total_rel_err += err
                    count += 1

                avg_err = total_rel_err / count
                errors[nl].append(avg_err)

    # 计算统计量
    mean_err = [np.mean(errors[nl]) for nl in noise_levels]
    std_err = [np.std(errors[nl]) for nl in noise_levels]

    # 线性拟合
    X = np.array(noise_levels).reshape(-1, 1)
    y = np.array(mean_err)
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    r2 = reg.score(X, y)

    return {
        'name': model_name,
        'noise_levels': noise_levels,
        'mean_err': mean_err,
        'std_err': std_err,
        'slope': slope,
        'intercept': intercept,
        'r2': r2
    }


# ---------- 6. 绘制对比图 ----------
def plot_comparison(results, save_path='noise_robustness_comparison.png'):
    """绘制所有模型的噪声鲁棒性对比图"""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11

    # 使用不同的颜色和线型
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：误差曲线
    ax1 = axes[0]
    for idx, result in enumerate(results):
        model_name = result['name']
        noise_levels = result['noise_levels']
        mean_err = result['mean_err']
        std_err = result['std_err']
        color = colors[idx]
        marker = markers[idx % len(markers)]

        ax1.errorbar(noise_levels, mean_err, yerr=std_err, fmt=marker,
                     color=color, capsize=3, capthick=1,
                     markersize=6, elinewidth=1, markeredgecolor='w',
                     label=model_name, alpha=0.8)

    ax1.set_xlabel('Noise standard deviation ε', fontsize=12)
    ax1.set_ylabel('Relative Reconstruction Error', fontsize=12)
    ax1.set_title('Noise Robustness Comparison', fontsize=14)
    ax1.set_xlim(-0.05, max(noise_levels) + 0.05)
    ax1.set_ylim(bottom=0)
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 右图：斜率对比（噪声敏感度）
    ax2 = axes[1]
    model_names = [r['name'] for r in results]
    slopes = [r['slope'] for r in results]
    r2_values = [r['r2'] for r in results]

    bars = ax2.barh(model_names, slopes, color=colors[:len(model_names)])
    ax2.set_xlabel('Sensitivity (Slope)', fontsize=12)
    ax2.set_title('Noise Sensitivity Comparison', fontsize=14)
    ax2.grid(True, linestyle=':', alpha=0.6, axis='x')

    # 添加 R² 值标注
    for i, (bar, r2) in enumerate(zip(bars, r2_values)):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'R²={r2:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ---------- 7. 保存详细结果 ----------
def save_detailed_results(all_results, csv_path='noise_analysis_all_models.csv'):
    """保存所有模型的详细结果到CSV"""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'NoiseLevel', 'MeanRelError', 'StdRelError', 'Slope', 'Intercept', 'R2'])

        for result in all_results:
            for nl, mean, std in zip(result['noise_levels'], result['mean_err'], result['std_err']):
                writer.writerow([
                    result['name'], nl, f'{mean:.6f}', f'{std:.6f}',
                    f'{result["slope"]:.6f}', f'{result["intercept"]:.6f}', f'{result["r2"]:.6f}'
                ])


# ---------- 8. 主程序 ----------
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 准备验证数据集
    print("\n准备验证数据集...")
    data_dir = "IXI-dataset-master/size64"
    full_dataset = MRIDataset(data_dir, sampling_rate=0.3, mask_type='cartesian', normalize_to_01=True)

    torch.manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 固定验证样本
    val_samples = []
    for img_c, mask in val_loader:
        val_samples.append((img_c.squeeze(0), mask.squeeze(0)))
    print(f"验证集大小: {len(val_samples)}")

    # 定义噪声水平和随机种子
    noise_levels = [0.00, 0.05, 0.10, 0.20, 0.40, 0.80]
    num_seeds = 5
    seeds = [2026 + i for i in range(num_seeds)]

    # 获取所有模型
    all_models = get_all_models()
    print(f"\n待分析模型列表: {list(all_models.keys())}")

    # 创建保存结果的目录
    os.makedirs('noise_analysis_results', exist_ok=True)

    # 分析每个模型
    all_results = []
    successful_models = []

    for model_name, model in all_models.items():
        print(f"\n{'=' * 50}")
        print(f"处理模型: {model_name}")

        # 加载权重（对于可训练模型）
        if model_name not in ['ADMM_TV', 'CPNN']:  # 这些模型可能不需要预训练权重
            success = load_model_weights(model, model_name, checkpoint_dir='有噪声checkpoints')
            if not success:
                print(f"跳过 {model_name}")
                continue
        else:
            # 对于不需要训练的模型，直接使用
            model.to(device)
            model.eval()
            print(f"  {model_name} 使用默认参数")

        successful_models.append(model_name)

        # 进行噪声分析
        result = analyze_model_noise(model, model_name, val_samples, noise_levels, seeds, device)
        all_results.append(result)

        # 打印该模型的拟合结果
        print(f"  {model_name}: 斜率={result['slope']:.4f}, 截距={result['intercept']:.4f}, R²={result['r2']:.4f}")

    # 保存结果
    if all_results:
        # 保存详细CSV
        save_detailed_results(all_results, 'noise_analysis_results/noise_analysis_all_models.csv')

        # 绘制对比图
        plot_comparison(all_results, 'noise_analysis_results/noise_robustness_comparison.png')

        # 打印汇总表格
        print(f"\n{'=' * 70}")
        print("噪声鲁棒性分析汇总")
        print(f"{'Model':<15} {'Slope':<10} {'Intercept':<12} {'R²':<8}")
        print("-" * 50)
        for result in all_results:
            print(f"{result['name']:<15} {result['slope']:<10.4f} {result['intercept']:<12.4f} {result['r2']:<8.4f}")

        # 找出最鲁棒的模型（斜率最小）
        most_robust = min(all_results, key=lambda x: x['slope'])
        print(f"\n🏆 噪声最鲁棒的模型: {most_robust['name']} (斜率={most_robust['slope']:.4f})")

        # 找出最敏感的模型（斜率最大）
        most_sensitive = max(all_results, key=lambda x: x['slope'])
        print(f"⚠️  噪声最敏感的模型: {most_sensitive['name']} (斜率={most_sensitive['slope']:.4f})")

        print(f"\n✅ 分析完成！结果保存在 'noise_analysis_results/' 目录")
    else:
        print("\n❌ 没有成功加载任何模型，请检查模型权重文件路径")


if __name__ == "__main__":
    main()