import re
import matplotlib.pyplot as plt


def extract_data(file_path):
    epochs, losses, psnrs = [], [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r'LIPN_pro Epoch (\d+)/\d+ \| Loss ([\d\.]+) \| PSNR ([\d\.]+)', line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                psnrs.append(float(match.group(3)))
    return epochs, losses, psnrs


files = {
    r'$\delta=0$': r'C:\Users\Administrator\Desktop\论文\变分不等式第二篇\5060跑出来的结果\无噪声50轮.txt',
    r'$\delta=0.0588$': r'C:\Users\Administrator\Desktop\论文\变分不等式第二篇\5060跑出来的结果\0.05噪声50轮.txt',
    r'$\delta=0.2$': r'C:\Users\Administrator\Desktop\论文\变分不等式第二篇\5060跑出来的结果\0.2噪声50轮.txt',
    r'$\delta=0.4$': r'C:\Users\Administrator\Desktop\论文\变分不等式第二篇\5060跑出来的结果\0.4噪声50轮.txt',
}

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']  # 红、蓝、绿、紫
markers = ['o', 's', '^', 'D']

fig, ax1 = plt.subplots(figsize=(10, 6))
# 设置纯白背景
fig.patch.set_facecolor('white')
ax1.set_facecolor('white')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss (MSE)')
# 网格已去除：下面这行被注释掉
# ax1.grid(True, linestyle='--', alpha=0.3)

ax2 = ax1.twinx()
ax2.set_ylabel('Validation PSNR (dB)')
ax2.set_facecolor('white')   # 右轴背景也设为白色（通常不需要，但加了也无妨）

all_losses = []
all_psnrs = []
all_handles = []
all_labels = []

for idx, (label, path) in enumerate(files.items()):
    epochs, losses, psnrs = extract_data(path)
    color = colors[idx]
    marker = markers[idx]

    all_losses.extend(losses)
    all_psnrs.extend(psnrs)

    l1, = ax1.plot(epochs, losses, linestyle='-', color=color, linewidth=2,
                   marker=marker, markersize=4, markevery=5, alpha=0.9)
    l2, = ax2.plot(epochs, psnrs, linestyle='--', color=color, linewidth=2,
                   marker=marker, markersize=4, markevery=5, alpha=0.9)

    all_handles.extend([l1, l2])
    all_labels.extend([f'Loss ({label})', f'PSNR ({label})'])

# 放大 Y 轴上限，留出顶部空间
loss_max = max(all_losses)
psnr_max = max(all_psnrs)
ax1.set_ylim(bottom=-0.001, top=loss_max * 1.15)
ax2.set_ylim(bottom=min(all_psnrs)*0.95 - 0.001, top=psnr_max * 1.08)

# 图例置于内部顶部，无背景框
ax1.legend(all_handles, all_labels,
           loc='upper center',
           bbox_to_anchor=(0.5, 0.98),
           ncol=4,
           fontsize=7.5,
           frameon=False)   # 不要图例背景框

plt.tight_layout()
plt.savefig('training_convergence_clear.png', dpi=600, facecolor='white')
plt.show()

for label, path in files.items():
    _, losses, psnrs = extract_data(path)
    print(f"{label}: Final Loss = {losses[-1]:.6f}, Final PSNR = {psnrs[-1]:.2f} dB")