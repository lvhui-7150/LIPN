# 去噪实验 (Natural Image Denoising)

## 概述

在 BSD68 数据集上验证 LIPN 框架在自然图像去噪任务上的通用性。
对比 LIPN vs DnCNN vs IRCNN vs BM3D，在 σ=15/25/50 三个噪声级别。

## 文件说明

| 文件 | 功能 |
|---|---|
| `denoising_experiment.py` | 主脚本：训练 + 评估 + 结果表 + 图表 |
| `inspect_model.py` | 加载已训练模型，查看 PSNR/SSIM 和可学习参数 |
| `resume_training.py` | 从中断的 checkpoint 续训 |
| `gen_analysis_figs.py` | 生成分析图（不动点残差衰减、学习到的 η 值） |
| `fix_test_noise.py` | 修正 BSD68 测试噪声图（一次性工具） |
| `denoising-datasets-main/` | 数据集：BSD400（训练）、BSD68（测试） |
| `denoising_results/` | 输出：模型权重、训练曲线、结果表 |

## 快速开始

```bash
# 快速训练（30 epochs，σ=15/25/50）
python denoising_experiment.py --mode quick

# 只训练一个 epoch 验证代码
python denoising_experiment.py --mode dryrun

# 评估已保存的模型
python denoising_experiment.py --mode eval --sigma 15 25 50

# 查看 LIPN σ=25 的性能和 η 值
python inspect_model.py LIPN 25

# 一次性查看所有模型
python inspect_model.py all 15 25 50
```

## 架构

```
LIPN: G_θ(z) = z + H(z)（tail 零初始化）
      z_{k+1} = G_θ(z_k - η_k · F(z_k)), F(z) = z - y
      K = 5 迭代, 297K 参数

DnCNN: 17 层 Conv+BN+ReLU, 残差学习, 556K 参数
IRCNN: 7 层空洞卷积, 残差学习, 223K 参数
BM3D:  经典非学习方法（无需训练）
```

## 训练配置

| 参数 | 值 |
|---|---|
| 训练集 | BSD400（400 张灰度图） |
| 测试集 | BSD68（68 张灰度图） |
| Patch 大小 | 64×64 |
| Batch Size | 12 |
| Optimizer | Adam, lr=3e-5, weight_decay=1e-4 |
| Scheduler | StepLR(step=15, gamma=0.5) |
| Loss | L1 |
| 随机种子 | 42（完全可复现） |
