# CrossFuse 项目运行指南

## 📋 目录
1. [环境配置](#环境配置)
2. [依赖安装](#依赖安装)
3. [数据准备](#数据准备)
4. [配置文件修改](#配置文件修改)
5. [运行步骤](#运行步骤)
6. [常见问题](#常见问题)

---

## 🔧 环境配置

### 系统要求
- **Python**: 3.7（推荐）
- **PyTorch**: >= 1.8.0
- **CUDA**: 如果使用GPU训练，需要安装CUDA（推荐CUDA 10.2或11.0+）

### 检查Python版本
```bash
python --version  # 应该显示 Python 3.7.x
```

---

## 📦 依赖安装

### 方法1：使用requirements.txt（推荐）

```bash
# 进入项目目录
cd CrossFuse

# 安装依赖
pip install -r requirements.txt
```

### 方法2：手动安装

```bash
# 安装PyTorch（根据你的CUDA版本选择）
# CPU版本
pip install torch torchvision

# GPU版本（CUDA 11.0）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu110

# 其他依赖
pip install numpy opencv-python scipy matplotlib seaborn timm
```

### 验证安装
```bash
python -c "import torch; print(torch.__version__)"
python -c "import cv2; print(cv2.__version__)"
python -c "import timm; print('timm installed')"
```

---

## 📁 数据准备

### 训练数据（KAIST数据集）

项目需要KAIST数据集进行训练，包含：
- **红外图像**（lwir文件夹）
- **可见光图像**（visible文件夹）

#### 下载KAIST数据集
- 官方链接：https://soonminhwang.github.io/rgbt-ped-detection/
- 数据集结构应该是：
```
KAIST/
├── lwir/          # 红外图像
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── visible/       # 可见光图像
    ├── image1.png
    ├── image2.png
    └── ...
```

#### 如果没有KAIST数据集（仅测试）
可以跳过训练步骤，直接使用预训练模型进行测试。

### 测试数据

项目已包含测试数据集（在`images/`文件夹中）：
- `21_pairs_tno.zip` - TNO数据集（21对图像）
- `40_vot_tno.zip` - TNO数据集（40对图像）
- `M3FD_Fusion.zip` - M3FD融合数据集

**解压测试数据：**
```bash
cd CrossFuse/images
# 解压需要的测试数据集
unzip 21_pairs_tno.zip  # Linux/Mac
# 或使用解压软件在Windows上解压
```

解压后结构：
```
images/
└── 21_pairs_tno/
    ├── ir/        # 红外图像
    └── vis/       # 可见光图像
```

---

## ⚙️ 配置文件修改

### 1. 修改 `args_auto.py`（自编码器训练配置）

**重要修改项：**
```python
# 修改数据路径（第13行）
path = ['你的KAIST数据集路径/lwir/']  # 例如：['D:/datasets/KAIST/lwir/']

# 如果没有GPU，设置为False（第16行）
cuda = False  # 如果有GPU且已安装CUDA，保持True

# 如果内存不足，减小batch size（第19行）
batch = 1  # 默认是2，可以改为1

# 如果数据集较小，减小训练样本数（第22行）
train_num = 1000  # 默认40000，测试时可以改小
```

### 2. 修改 `args_trans.py`（融合网络训练配置）

**重要修改项：**
```python
# 修改数据路径（第10行）
path_ir = ['你的KAIST数据集路径/lwir/']

# 如果没有GPU，设置为False（第12行）
cuda = False

# 如果内存不足，减小batch size（第15行）
batch = 4  # 默认是8，可以改为4或更小

# 修改预训练自编码器路径（第27-28行）
# 如果已经训练了自编码器，确保路径正确
resume_model_auto_ir = "./models/autoencoder/auto_encoder_epoch_4_ir.model"
resume_model_auto_vi = "./models/autoencoder/auto_encoder_epoch_4_vi.model"
```

### 3. GPU设置

如果使用GPU，检查`train_autoencoder.py`和`train_conv_trans.py`中的GPU设置：
```python
# 第27行，设置使用的GPU编号
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第0块GPU
# 如果有多个GPU，可以改为 "0,1" 使用多GPU
```

---

## 🚀 运行步骤

### 方案A：使用预训练模型进行测试（推荐，最快）

如果你只想测试融合效果，可以使用项目提供的预训练模型：

#### 1. 确保预训练模型存在
检查`models/`文件夹：
```
models/
├── autoencoder/
│   ├── auto_encoder_epoch_4_ir.model
│   └── auto_encoder_epoch_4_vi.model
└── transfuse/
    └── fusetrans_epoch_32_bs_8_num_20k_lr_0.1_s1_c1.model
```

#### 2. 解压测试数据
```bash
cd CrossFuse/images
# 解压21_pairs_tno.zip
```

#### 3. 运行测试脚本

**测试灰度图像（TNO数据集）：**
```bash
cd CrossFuse
python test_conv_trans.py
```

**测试彩色图像（M3FD数据集）：**
```bash
python test_color_image.py
```

#### 4. 查看结果
融合结果保存在：`output/crossfuse_test/` 文件夹中

---

### 方案B：完整训练流程

#### 步骤1：训练红外图像自编码器

```bash
cd CrossFuse

# 1. 修改 args_auto.py 中的路径和配置
# 2. 运行训练脚本
python train_autoencoder.py
```

**训练参数说明：**
- `type_flag = 'ir'` - 训练红外图像自编码器
- 训练完成后，模型保存在 `models/autoencoder/auto_encoder_epoch_4_ir.model`

**预期输出：**
```
Start training.....
BATCH SIZE 2.
Train images number 40000.
...
Checkpoint, trained model saved at: ./models/autoencoder/auto_encoder_epoch_4_ir.model
```

#### 步骤2：训练可见光图像自编码器

修改 `args_auto.py`：
```python
path = ['你的KAIST数据集路径/visible/']  # 改为可见光路径
type_flag = 'vi'  # 改为'vi'
```

运行训练：
```bash
python train_autoencoder.py
```

训练完成后，模型保存在 `models/autoencoder/auto_encoder_epoch_4_vi.model`

#### 步骤3：训练融合网络

```bash
# 1. 确保 args_trans.py 中的自编码器路径正确
# 2. 运行融合网络训练
python train_conv_trans.py
```

**训练参数说明：**
- 需要预训练的自编码器（步骤1和2）
- 训练时间较长（32个epoch）
- 模型保存在 `models/transfuse/` 文件夹

**预期输出：**
```
Resuming, initializing fusion net using weight from ./models/autoencoder/...
Start training.....
...
Checkpoint, trained model saved at: ./models/transfuse/fusetrans_epoch_32.model
```

#### 步骤4：测试融合效果

参考方案A的测试步骤。

---

## 🔍 常见问题

### Q1: `ModuleNotFoundError: No module named 'timm'`
**解决方案：**
```bash
pip install timm
```

### Q2: `CUDA out of memory`
**解决方案：**
1. 减小batch size（在配置文件中）
2. 减小图像尺寸（Height和Width）
3. 使用CPU训练（设置`cuda = False`）

### Q3: `FileNotFoundError: 找不到数据路径`
**解决方案：**
1. 检查`args_auto.py`和`args_trans.py`中的路径是否正确
2. 确保路径使用正斜杠`/`或双反斜杠`\\`
3. Windows路径示例：`path = ['D:/datasets/KAIST/lwir/']`

### Q4: 训练时loss不下降
**解决方案：**
1. 检查学习率是否合适（可以尝试更小的学习率）
2. 确保数据路径正确，数据能正常加载
3. 检查数据预处理是否正确

### Q5: 测试时找不到模型文件
**解决方案：**
1. 确保`models/`文件夹中有预训练模型
2. 检查`test_conv_trans.py`中的模型路径是否正确
3. 如果模型文件名不同，修改测试脚本中的路径

### Q6: 没有GPU怎么办？
**解决方案：**
1. 在配置文件中设置`cuda = False`
2. 使用CPU训练（会很慢，建议只用于测试）
3. 使用Google Colab等免费GPU平台

---

## 📝 快速测试命令总结

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 解压测试数据
cd images && unzip 21_pairs_tno.zip && cd ..

# 3. 运行测试（使用预训练模型）
python test_conv_trans.py

# 4. 查看结果
# 结果在 output/crossfuse_test/ 文件夹中
```

---

## 💡 提示

1. **首次运行建议**：先运行测试脚本，确保环境配置正确
2. **训练时间**：完整训练需要较长时间，建议先用小数据集测试
3. **GPU加速**：如果有GPU，强烈建议使用GPU训练
4. **数据路径**：Windows系统注意路径格式，建议使用正斜杠`/`

---

## 📞 需要帮助？

如果遇到问题，可以：
1. 查看项目README.md
2. 检查GitHub Issues
3. 联系作者：lihui.cv@jiangnan.edu.cn

