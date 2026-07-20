import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import avg_pool2d

# 解决某些环境下库冲突的问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def generate_smooth_phase(shape, kernel_size=7):
    """
    生成具有空间相关性的平滑随机相位，模拟真实 MRI 中的磁场不均匀性
    """
    noise = torch.randn(1, 1, *shape)
    # 使用平均池化进行平滑处理
    smoothed = avg_pool2d(noise, kernel_size, stride=1, padding=kernel_size // 2)
    # 将相位缩放到 [-π, π] 之间
    phase = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
    phase = (phase * 2 * np.pi - np.pi).view(shape)
    return phase


def process_ixi_to_complex(data_dir, num_samples=100, use_zero_phase=False):
    """
    处理 IXI 数据集并构建适合 CMVI 实验的复数 Tensor
    """
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.pt') and 'T1' in f]
    if not all_files:
        raise FileNotFoundError(f"未在 {data_dir} 发现 .pt 文件。")

    num_samples = min(num_samples, len(all_files))
    selected_files = np.random.choice(all_files, num_samples, replace=False)

    complex_images = []

    print(f"正在处理 {num_samples} 个样本...")

    for file_name in selected_files:
        file_path = os.path.join(data_dir, file_name)
        # 加载幅度图并归一化
        amplitude = torch.load(file_path).float()
        amplitude = amplitude / (amplitude.max() + 1e-8)

        if use_zero_phase:
            img_complex = torch.complex(amplitude, torch.zeros_like(amplitude))
        else:
            phase = generate_smooth_phase(amplitude.shape)
            # 根据 Euler 公式生成复数图像: z = |z| * exp(i * phi)
            img_complex = amplitude.to(torch.complex64) * torch.exp(1j * phase)

        complex_images.append(img_complex)

    # 将 List 转换为 [N, H, W] 的 Tensor
    dataset_tensor = torch.stack(complex_images)

    # 额外生成一个用于逆问题实验的采样掩码 (Cartesian Mask 示例)
    # 假设 40% 的采样率
    mask = torch.randn(amplitude.shape) > 0.25

    return dataset_tensor, mask


def save_and_visualize(dataset, mask, save_dir, filename, num_vis=3):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, filename)

    # 构建保存字典，包含数据和实验参数
    data_to_save = {
        "gt_complex": dataset,
        "mask": mask,
        "description": "Complex-valued IXI dataset for CMVI research."
    }

    torch.save(data_to_save, save_path)
    print(f"✅ 处理完成！数据已保存至 (新文件): {save_path}")

    # 可视化校验
    plt.figure(figsize=(12, 4))
    for i in range(num_vis):
        img = dataset[i].numpy()

        plt.subplot(1, num_vis, i + 1)
        plt.imshow(np.abs(img), cmap='gray')
        plt.title(f"Sample {i + 1} Amplitude")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 配置路径
    SOURCE_DIR = "IXI-dataset-master/size64"
    OUTPUT_DIR = "IXI-dataset-master/processed"

    # 运行处理逻辑
    # 建议使用随机相位 (use_zero_phase=False) 来体现复值变分不等式的优势
    img_tensor, sampling_mask = process_ixi_to_complex(
        SOURCE_DIR,
        num_samples=100,
        use_zero_phase=False
    )

    # 保存结果（存为新文件，不覆盖原始 IXI 数据）
    save_and_visualize(
        img_tensor,
        sampling_mask,
        OUTPUT_DIR,
        "complex_ixi_data_v1.pt"
    )

