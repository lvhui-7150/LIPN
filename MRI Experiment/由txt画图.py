import re
import matplotlib.pyplot as plt
import numpy as np
import os

# ================== 全局样式设置 ==================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.dpi': 300,
})

# 高对比度配色（可区分6-8个模型）
COLORS = {
    'LIPN_pro': '#d62728',  # 红色（自己方法）
    'LIPN': '#1f77b4',  # 蓝色
    'VarNet': '#2ca02c',  # 绿色
    'MoDL': '#ff7f0e',  # 橙色
    'DCCNN': '#9467bd',  # 紫色
    'ISTANet': '#8c564b',  # 棕色
    'LGD_Net': '#e377c2',  # 粉色
    'ComplexUNet': '#7f7f7f',  # 灰色
    'ADMM_TV': '#bcbd22',  # 黄绿
    'CPNN': '#17becf',  # 青色
}

MARKERS = {
    'LIPN_pro': 's',  # 方块
    'LIPN': 'o',  # 圆圈
    'VarNet': '^',  # 上三角
    'MoDL': 'D',  # 菱形
    'DCCNN': 'v',  # 下三角
    'ISTANet': 'p',  # 五边形
}


# ================== 解析函数（同前，略）==================
def parse_training_log(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    pattern = re.compile(r'(\w+)\s+Epoch\s+(\d+)/\d+\s+\|\s+Loss\s+([\d\.e\-]+)\s+\|\s+PSNR\s+([\d\.]+)')
    data = {}
    for line in lines:
        m = pattern.search(line)
        if m:
            model = m.group(1)
            epoch = int(m.group(2))
            loss = float(m.group(3))
            psnr = float(m.group(4))
            if model not in data:
                data[model] = {'epoch': [], 'psnr': [], 'loss': []}
            data[model]['epoch'].append(epoch)
            data[model]['psnr'].append(psnr)
            data[model]['loss'].append(loss)
    return data


def parse_final_evaluation(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    marker = "===== Final Evaluation on Validation Set ====="
    if marker not in content:
        return {}
    section = content.split(marker)[1]
    pattern = re.compile(r'(\S+)\s+\|\s+PSNR:\s+([\d\.]+)\s+dB\s+\|\s+SSIM:\s+([\d\.]+)\s+\|\s+NMSE:\s+([\d\.]+)')
    results = {}
    for line in section.split('\n'):
        m = pattern.search(line)
        if m:
            model = m.group(1)
            psnr = float(m.group(2))
            ssim = float(m.group(3))
            nmse = float(m.group(4))
            results[model] = {'psnr': psnr, 'ssim': ssim, 'nmse': nmse}
    return results


# ================== 高级绘图函数 ==================
def plot_psnr_curves_advanced(data, save_path='training_psnr_curves.pdf'):
    models_to_plot = ['LIPN', 'LIPN_pro', 'VarNet', 'MoDL', 'DCCNN']
    fig, ax = plt.subplots(figsize=(8.5, 5.5))  # 适合双栏
    for i, model in enumerate(models_to_plot):
        if model in data:
            d = data[model]
            # 每隔5个epoch添加一个标记点，避免过密
            step = max(1, len(d['epoch']) // 6)
            ax.plot(d['epoch'], d['psnr'],
                    color=COLORS.get(model, '#333333'),
                    linewidth=2.5,
                    label=model,
                    marker=MARKERS.get(model, 'o'),
                    markevery=step,
                    markersize=7,
                    markerfacecolor='white',
                    markeredgewidth=1.5,
                    markeredgecolor=COLORS.get(model, '#333333'))
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('PSNR (dB)', fontweight='bold')
    ax.set_title('Validation PSNR during training (noiseless)', fontweight='bold')
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    # 调整刻度
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=False, right=False)
    # 设置x轴范围从1到30
    ax.set_xlim(0.8, 30.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 高级PSNR曲线已保存: {save_path}")


def plot_final_bar_advanced(results, title, save_path, metric='psnr', ylabel='PSNR (dB)'):
    if not results:
        print(f"⚠️ 无数据，跳过 {save_path}")
        return
    # 按PSNR降序排序
    models = sorted(results.keys(), key=lambda x: results[x][metric], reverse=True)
    values = [results[m][metric] for m in models]
    # 使用自定义颜色：自己方法用红色，其他用渐变色
    colors = ['#d62728' if m in ['LIPN', 'LIPN_pro'] else '#1f77b4' for m in models]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=1.2, width=0.7)
    # 添加数值标签（在柱顶上方）
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f'{val:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    # 去掉上、右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 高级柱状图已保存: {save_path}")


# ================== 主程序 ==================
if __name__ == "__main__":
    noiseless_log = "5.5无噪声结果.txt"
    noisy_log = "5.5有噪声结果.txt"

    if os.path.exists(noiseless_log):
        train_data = parse_training_log(noiseless_log)
        plot_psnr_curves_advanced(train_data, "training_psnr_curves_advanced.pdf")

        final_noiseless = parse_final_evaluation(noiseless_log)
        if final_noiseless:
            plot_final_bar_advanced(final_noiseless, "Reconstruction PSNR (Noiseless)",
                                    "noiseless_psnr_bar_advanced.pdf", metric='psnr')
    else:
        print(f"❌ 找不到 {noiseless_log}")

    if os.path.exists(noisy_log):
        final_noisy = parse_final_evaluation(noisy_log)
        if final_noisy:
            plot_final_bar_advanced(final_noisy, "Reconstruction PSNR (Noisy, δ=0.0588)",
                                    "noisy_psnr_bar_advanced.pdf", metric='psnr')
    else:
        print(f"❌ 找不到 {noisy_log}")

    print("\n📌 提示：噪声稳定性曲线请运行 error_plus.py 生成 stability_theorem_verification.png")
    print("   可将其转换为 PDF 或 EPS 以获得矢量图。")