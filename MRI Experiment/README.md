# MRI 重建实验 (Complex-Valued MRI Reconstruction)

## 概述

在 IXI 脑部 MRI 数据集上验证 LIPN 在复数域欠采傅里叶重建中的性能。
对比 LIPN vs VarNet、MoDL、ComplexUNet、DCCNN、ADMM-TV、CPNN、LGD、ISTANet 等多个基线。

## 文件说明

### 核心模型

| 文件 | 模型 |
|---|---|
| `LIPN.py` | **LIPN**：Learned Inexact Proximal Network（本文提出） |
| `LIPN_pro.py` | LIPN 增强版：相位等变复数卷积 + 通道注意力 |
| `VarNet.py` | End-to-End Variational Network |
| `MoDL.py` | Model-based Deep Learning |
| `ComplexUNet.py` | 复数 U-Net |
| `DCCNN_Model.py` | 深度级联 CNN |
| `ADMM_TV.py` | ADMM + Total Variation |
| `ISTANet.py` | ISTA-Net |
| `CPNN.py` | Complex Proximal Neural Network |
| `LGD.py` | Learned Gradient Descent |

### 工具模块

| 文件 | 功能 |
|---|---|
| `fft_utils.py` | 中心化 FFT/IFFT |
| `fastMRIdata.py` | fastMRI 数据加载器 |
| `error_plus.py` | 误差分析工具 |

### 实验脚本

| 文件 | 功能 |
|---|---|
| `保存最优模型.py` | 完整训练流程 |
| `无噪声模式.py` | 无噪声 MRI 重建 |
| `加入噪声.py` | 有噪声（多噪声级别）重建 |
| `遍历采样率.py` | 不同欠采率对比 |
| `遍历噪声.py` | 不同噪声水平对比 |
| `重新训练模型遍历噪声.py` | 重训模型 + 噪声遍历 |
| `全部模型的噪声收敛性分析.py` | 所有模型的噪声收敛性 |
| `6.2收敛性分析的图.py` | 收敛性分析图 |
| `6.3噪声收敛性分析.py` | 噪声收敛性分析 |
| `相位一致性验证（无噪声）.py` | 无噪声相位一致性 |
| `相位一致性检验（有噪声）.py` | 有噪声相位一致性 |
| `不同噪声不同相位加载最优模型.py` | 多条件加载最优模型 |
| `新的数据集的噪声训练.py` | 新数据集噪声训练 |
| `由txt画图.py` | 文本数据作图 |
| `图像处理.py` | 图像预处理 |

### 数据集

- `IXI-dataset-master/`：IXI T1 加权脑部 MRI，1154 张，resize 到 64×64
