# Image_Recognition_Coursework
# README

## 项目概述
本项目运用 Unet++ 模型执行图像分割任务，重点针对 CamVid 数据集中的道路场景图像展开训练与测试工作。项目代码涵盖了数据加载、图像增强、模型构建、训练以及验证等一整套完整流程。

## 软硬件环境
### 硬件环境
- **GPU**：NVIDIA CUDA 兼容的 GPU（推荐使用具有较高显存的 GPU，如 NVIDIA RTX 3080 或更高型号，以支持较大的 batch size 和复杂的模型训练）
- **CPU**：多核心处理器（推荐 8 核心或以上，以加快数据预处理和模型训练速度）
- **内存**：至少 16GB RAM（推荐 32GB 或以上，以避免在处理大型数据集时出现内存不足的问题）

### 软件环境
- **Python**：3.7 或更高版本
- **PyTorch**：1.7 或更高版本（确保与 CUDA 版本兼容）
- **CUDA**：10.1 或更高版本（根据 GPU 型号和 PyTorch 版本选择合适的 CUDA 版本）https://github.com/Cabonhydrate/Image_Recognition_Coursework/blob/main/README.md
- **cuDNN**：7.6 或更高版本（与 CUDA 版本相匹配，以加速深度学习模型的训练）
- **其他依赖库**：numpy、opencv-python、matplotlib、albumentations、segmentation_models_pytorch 等（可通过 `pip install -r requirements.txt` 安装项目依赖文件中列出的所有库）

## 数据集
### 数据集来源
本项目采用的是 CamVid 数据集，该数据集涵盖多种道路场景的图像以及与之对应的像素级标注。数据集可通过以下 GitHub 仓库克隆下载：
https://github.com/alexgkendall/SegNet-Tutorial

### 数据集结构
下载后的数据集应具备如下结构：
```
CamVid/
├── train/
├── trainannot/
├── val/
└── valannot/
```
其中，`train` 和 `val` 文件夹分别存放训练集和验证集的图像文件，`trainannot` 和 `valannot` 文件夹包含对应的标注文件。

### 数据集用法
- **数据加载**：通过 `Dataset` 类加载训练集和验证集的图像及其标注。在 `Dataset` 类中，指定了图像文件夹路径、标注文件夹路径、类别列表、图像增强方法和预处理方法等参数。
- **图像增强**：使用 `albumentations` 库定义了训练集和验证集的图像增强方法。训练集的增强方法包含水平翻转、平移缩放旋转、填充、随机裁剪、高斯噪声、透视变换、亮度对比度调整、锐化、模糊、运动模糊等操作；验证集的增强方法主要是填充操作，以确保图像分辨率能被 32 整除。
- **数据预处理**：通过 `get_preprocessing` 函数定义了图像预处理方法，包括使用预训练模型的归一化函数和将图像转换为张量的操作。

## 模型训练
### 模型构建
使用 `segmentation_models_pytorch` 库中的 `UnetPlusPlus` 类构建 Unet++ 模型。模型的编码器为 `se_resnext50_32x4d`，编码器权重使用预训练的 `imagenet` 权重，类别数为 1（针对 `car` 类别进行分割），激活函数为 `sigmoid`。

### 训练流程
- **数据加载器**：通过 `DataLoader` 类创建训练集和验证集的数据加载器，设置合适的 `batch size` 和 `num_workers` 参数，以加快数据加载速度。
- **损失函数和评估指标**：定义了 `Dice Loss` 作为损失函数，`IoU` 作为评估指标。
- **优化器**：使用 `Adam` 优化器，初始学习率为 0.0001。
- **训练和验证循环**：创建训练和验证的 `Epoch` 对象，进行 40 轮次的迭代训练。在每轮迭代中，运行训练和验证循环，记录损失和评估指标。若当前轮次的 `IoU` 评分高于之前轮次的最大值，则保存当前模型。当迭代轮次达到 25 时，将优化器的学习率降低到 1e-5。

## 运行项目
- **环境准备**：确保已安装上述软硬件环境，并安装项目依赖库。
- **数据准备**：克隆下载 CamVid 数据集，并解压到指定目录（默认为当前目录）。
- **运行训练脚本**：在终端中运行 `python Unet++.py` 命令，开始模型训练。训练过程中，会在控制台输出每轮次的损失和评估指标，并在训练完成后保存最佳模型到当前目录下。

## 联系方式
如有任何问题或建议，请联系项目作者。 
