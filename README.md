# LIPN 实验

论文 "A Complex-Valued Variational Inequality Method Based on Learned Inexact Proximal Operators" 的实验代码。

## 目录结构

```
实验部分/
├── README.md               ← 本文件
├── 去噪实验/                ← 自然图像去噪
│   ├── README.md
│   ├── denoising_experiment.py
│   ├── denoising-datasets-main/
│   └── ...
├── MRI实验/                 ← 复数域 MRI 重建
│   ├── README.md
│   ├── LIPN.py
│   ├── IXI-dataset-master/
│   └── ...
├── denoising_results/       ← 去噪实验输出（模型 + 图表）
└── 按模板/                   ← LaTeX 论文源文件
    ├── main.tex
    ├── fig/
    └── ...
```

## 两个实验

| 实验 | 数据 | 目的 | 对比模型 |
|---|---|---|---|
| 去噪 | BSD400 → BSD68 | 验证框架通用性 | BM3D, DnCNN, IRCNN |
| MRI | IXI 脑部 MRI | 核心应用：复数域 VI | VarNet, MoDL, ComplexUNet 等 8 个 |

## 环境

- Python 3.10+, PyTorch 2.x, CUDA 11.8+
- `pip install bm3d scikit-image matplotlib tqdm pillow`
