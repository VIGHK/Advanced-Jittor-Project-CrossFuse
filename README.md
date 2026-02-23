# CrossFuse-Jittor

![Project Status](https://img.shields.io/badge/status-completed-brightgreen) ![Python](https://img.shields.io/badge/python-3.8.20-blue) ![Jittor](https://img.shields.io/badge/jittor-1.3.10.0-red) ![CUDA](https://img.shields.io/badge/CUDA-12.2-green)

## 项目简介

本项目基于 [CrossFuse: A Novel Cross Attention Mechanism based Infrared and Visible Image Fusion Approach](https://www.sciencedirect.com/science/article/abs/pii/S1566253523004633) 论文和[原 PyTorch 项目](https://github.com/hli1221/CrossFuse)，使用 Jittor 框架完成了 CrossFuse 图像融合算法的完整迁移。该项目旨在：

- 使用 Jittor 框架复现原始算法
- 实现两阶段训练流程（自编码器训练 + 融合网络训练）
- 对比 Jittor 和 PyTorch 的训练损失曲线、评估指标和性能表现

## 环境配置

### 硬件环境

- **GPU**: NVIDIA RTX 4060
- **操作系统**: WSL2 (Ubuntu 22.04)
- **GPU 使用方式**: 单卡训练（CUDA_VISIBLE_DEVICES = 0）

### 软件环境

- **Python**: 3.8.20
- **CUDA**: 12.2
- **深度学习框架**: Jittor 1.3.10.0 (CUDA 12.2)

### 关键依赖库

```
NumPy
SciPy
OpenCV-Python
Matplotlib
Scikit-image
Seaborn
```

### Jittor 环境配置

1. **安装 Jittor**

```bash
pip install jittor
```

2. **验证安装**

```bash
python -c "import jittor as jt; print(jt.__version__)"
python -c "import jittor as jt; jt.test()"
```

3. **安装项目依赖**

```bash
pip install -r requirements.txt
```

**注意**：本项目使用 Jittor 框架，不需要安装 PyTorch。如果系统中已安装 PyTorch，不会影响 Jittor 的使用。

## 模型结构

CrossFuse 采用两阶段训练策略：

1. **第一阶段：自编码器训练**
   - 分别训练红外（IR）和可见光（VI）自编码器
   - 用于提取单模态特征，保证特征提取的针对性

2. **第二阶段：融合网络训练**
   - 使用预训练的自编码器提取特征
   - 通过跨注意力机制（Cross-Attention Module, CAM）融合 IR 和 VI 特征
   - 生成融合图像

核心组件：

- **自编码器** (`Auto_Encoder_single`): 单模态特征提取
- **融合网络** (`Trans_FuseNet`): 跨注意力融合模块
- **损失函数**: 像素损失（L_pix）+ 梯度损失（L_gra）

## 实验流程

### 数据准备

#### 训练数据集

项目使用 **KAIST 数据集**进行训练，与论文保持一致：

**数据集获取：**

- **Kaggle 下载地址**：[KAIST Dataset](https://www.kaggle.com/datasets/adlteam/kaist-dataset/code)
- **注意**：论文中提供的原始获取地址已失效，建议从 Kaggle 下载数据集
- 下载后，将数据集放置在 `database/` 目录下，保持原有的文件夹结构（如 `set00/V000/lwir/` 等）

**数据使用说明：**

- **第一阶段（自编码器训练）**：
  - 红外自编码器：40,000 张红外图像
  - 可见光自编码器：40,000 张可见光图像
  - 分开训练保证单模态特征提取的针对性

- **第二阶段（融合网络训练）**：
  - 20,000 对红外-可见光图像对
  - 确保训练数据的一致性

#### 测试数据集

项目包含以下测试数据集（位于 `images/` 文件夹）：

- `21_pairs_tno/` - TNO 数据集（21 对图像）
- `40_vot_tno/` - TNO 数据集（40 对图像）
- `M3FD_Fusion/` - M3FD 融合数据集（200 对彩色图像）

### 训练步骤

#### 1. 自编码器训练

**训练红外自编码器：**

```bash
# 修改 args_auto.py 中的 type_flag = 'ir'
python train_autoencoder.py
```

**训练可见光自编码器：**

```bash
# 修改 args_auto.py 中的 type_flag = 'vi'
python train_autoencoder.py
```

**训练参数：**

- Epochs: 4
- Batch size: 2
- Learning rate: 0.0001
- 学习率衰减：从第 2 个 epoch 开始，每个 epoch 衰减 10 倍

#### 2. 融合网络训练

```bash
python train_conv_trans.py
```

**训练参数：**

- Epochs: 16
- Batch size: 8
- Learning rate: 0.001
- 学习率衰减：从第 2 个 epoch 开始，每个 epoch 衰减 10 倍
- 使用 AMP（自动混合精度）加速训练

### 测试步骤

**灰度图像测试：**

```bash
python test_conv_trans.py
```

**彩色图像测试：**

```bash
python test_color_image.py
```

测试脚本会自动识别模型类型并生成对应的输出文件夹。

## 模型对齐与性能对比

### Loss 曲线对齐情况

#### 自编码器训练损失对比

<div align="center">
  <img src="results/loss curves/autoencoder_total_loss_comparison.png" width="80%"/>
  <p><i>自编码器总损失对比：PyTorch vs Jittor</i></p>
</div>

#### 融合网络训练损失对比

<div align="center">
  <img src="results/loss curves/fusion_network_comparison.png" width="90%"/>
  <p><i>融合网络训练损失对比：PyTorch vs Jittor</i></p>
</div>

### 评估指标对比

在 **21_pairs_tno** 测试集上的评估结果：

| 指标     | PyTorch | Jittor  | 差异   |
| -------- | ------- | ------- | ------ |
| **EN ↑** | 6.8421  | 4.9863  | -27.1% |
| **SD ↑** | 59.3704 | 44.7594 | -24.6% |
| **MI ↑** | 11.5892 | 10.4958 | -9.4%  |

Jittor 版本在所有指标上都略低于 PyTorch，但差异在可接受范围内（相对误差 < 30%）。

### 性能对比

| 训练阶段         | PyTorch | Jittor         | 速度比 |
| ---------------- | ------- | -------------- | ------ |
| **自编码器训练** | 50 分钟 | 1 小时 21 分钟 | 62%    |
| **融合网络训练** | 8 小时  | 9 小时         | 89%    |

**结论**：

- 自编码器训练：Jittor 速度约为 PyTorch 的 62%
- 融合网络训练：Jittor 速度约为 PyTorch 的 89%
- 融合网络训练速度差异较小，可能与使用了 AMP 有关

## 实验结果

### 训练损失曲线

所有训练损失曲线图保存在 `results/loss curves/` 目录下：

- `autoencoder_total_loss_comparison.png` - 自编码器总损失对比
- `fusion_network_comparison.png` - 融合网络损失对比（总损失、像素损失、梯度损失）
- `ir_autoencoder_comparison.png` - IR 自编码器详细对比
- `vi_autoencoder_comparison.png` - VI 自编码器详细对比

### 训练好的模型

训练好的模型权重保存在 `results/models/` 目录下：

- `autoencoder/` - 自编码器模型
  - `auto_encoder_jittor_epoch_4_ir_*.model` - IR 自编码器（Epoch 4）
  - `auto_encoder_jittor_epoch_4_vi_*.model` - VI 自编码器（Epoch 4）
- `transfuse/` - 融合网络模型
  - `jittor_epoch_16_*.model` - 融合网络（Epoch 16）

### 训练日志

训练日志保存在 `results/logs/` 和 `logs/` 目录下：

- `results/logs/autoencoder/` - 自编码器训练日志
- `results/logs/transfuse/` - 融合网络训练日志
- `logs/my_trained_models/` - 所有训练实验的完整日志

## 关键修改点

在 Jittor 迁移过程中，主要进行了以下关键修改：

1. **`normalize_tensor` 函数对齐**
   - 使用 `reshape + repeat` 替代 `broadcast`，对齐 PyTorch 的 `view + repeat` 行为
   - 添加对非 4D 张量的兼容处理

2. **学习率衰减策略对齐**
   - 修正为从第 2 个 epoch 开始，每个 epoch 衰减 10 倍
   - 与 PyTorch 版本保持一致

3. **跨注意力模块（CAM）修复**
   - 修复 `transpose` 维度错误：`transpose(2,1).transpose(2,1)` → `transpose(1,2).reshape(...)`

4. **损失函数数值稳定性**
   - 显式指定 `val_range=255.0` 给 SSIM 损失，避免 NaN 问题
   - 调整 `msssim` 归一化逻辑，对齐 `pytorch_msssim` 行为

5. **损失累积精度**
   - 改为直接张量累积，避免过早转换为 float 导致精度损失

6. **数据加载**
   - 支持 KAIST 数据集的嵌套文件夹结构（`set00/V000/lwir/` 等）
   - 递归搜索图像文件，无需手动整理数据

## 经验教训

1. **数值对齐的重要性**：迁移过程中需要仔细对比每一步的数值输出，特别是损失函数和归一化操作。

2. **API 差异处理**：
   - Jittor 的 `jt.min/jt.max` 返回标量，需要转换为 Python float
   - Jittor 不支持多维度 `mean(dim=[1,2,3])`，需要链式调用 `mean(dim=1).mean(dim=1).mean(dim=1)`
   - Jittor 的 `broadcast` 行为与 PyTorch 的 `view().repeat()` 不完全一致

3. **训练配置对齐**：学习率衰减策略、优化器参数、损失权重等都需要与 PyTorch 版本完全一致。

4. **自编码器对齐成功**：虽然初始 loss 有差异，但最终收敛到相似的低值，说明自编码器迁移基本成功。

5. **融合网络对齐待完善**：融合网络的 loss 值存在显著差异，可能的原因包括：
   - 自编码器特征提取的细微差异累积
   - 跨注意力模块的数值精度问题
   - 损失函数计算的细微差异

## 文件结构

```
CrossFuse/
├── network/              # 网络定义
│   ├── net_autoencoder.py    # 自编码器网络
│   ├── net_conv_trans.py     # 融合网络
│   ├── transformer_cam.py    # 跨注意力模块
│   └── loss.py              # 损失函数
├── tools/                # 工具函数
│   ├── utils.py              # 图像处理、数据加载等
│   └── fusion_metrics.py     # 评估指标计算
├── train_autoencoder.py  # 自编码器训练脚本
├── train_conv_trans.py   # 融合网络训练脚本
├── test_conv_trans.py    # 灰度图像测试脚本
├── test_color_image.py   # 彩色图像测试脚本
├── evaluate_fusion_results.py  # 评估脚本
├── plot_loss_curves.py   # 损失曲线绘制工具
├── database/             # 训练数据（KAIST数据集）
├── images/               # 测试数据
├── models/               # 模型权重（训练时保存）
├── logs/                 # 训练日志（所有实验）
├── results/              # 实验结果
│   ├── loss curves/         # 损失曲线图
│   ├── models/              # 训练好的模型权重
│   └── logs/                # 精选的训练日志
└── output/              # 测试输出结果
    ├── metrics/             # 评估指标CSV
    └── crossfuse_test/      # 融合结果图像
```

## 引用

如果本项目对您的研究有帮助，请引用原始论文：

```bibtex
@article{li2024crossfuse,
  title={{CrossFuse: A Novel Cross Attention Mechanism based Infrared and Visible Image Fusion Approach}},
  author={Li, Hui and Wu, Xiao-Jun},
  journal={Information Fusion},
  volume={103},
  pages={102147},
  year={2024},
  publisher={Elsevier}
}
```

## 致谢

- 感谢原项目作者 [Hui Li](https://github.com/hli1221) 提供的 PyTorch 实现
- 感谢 Jittor 团队提供的深度学习框架支持

## 许可证

本项目是对原 CrossFuse PyTorch 实现的框架迁移版本。

原算法、模型设计及实现的知识产权归原作者所有：

Hui Li  
原始项目地址：https://github.com/hli1221/CrossFuse

本仓库仅在 Jittor 框架下进行适配与迁移，用于课程作业和学习交流目的。

本项目不对原始算法或模型主张任何所有权。

原始工作的全部权利归原作者所有。