import os
# 设置环境变量，指定可见的CUDA设备为0。这意味着程序将使用编号为0的GPU进行计算（需要根据自己的电脑进行更改）。
# 在多GPU环境中，可以通过修改这个值来选择使用不同的GPU，或者设置为空字符串来禁用GPU使用，使用CPU进行计算。
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  

import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
from timm.layers.activations import Swish
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import segmentation_models_pytorch.utils
import ssl
# 在Python中，默认情况下，会对通过网络请求获取的HTTPS资源进行严格的SSL证书验证。
# 但在某些情况下，比如开发环境中自签名证书或者网络配置问题导致证书验证失败时，通过将默认的https上下文设置为不验证证书的方式，
# 可以避免相关的SSL错误，使程序能够顺利进行网络相关操作（此处可能用于后续下载数据集等涉及网络请求的情况）。
ssl._create_default_https_context = ssl._create_unverified_context


# ---------------------------------------------------------------
### 加载数据

# 自定义数据集类，继承自BaseDataset，用于加载CamVid数据集。这个类负责读取图像数据、对应的掩码数据（用于图像分割任务的标签），
# 并可根据需要进行数据增强和预处理操作，以准备好适合输入到深度学习模型的数据格式。
class Dataset(BaseDataset):
    """CamVid数据集。进行图像读取，图像增强增强和图像预处理.

    Args:
        images_dir (str): 图像文件夹所在路径，用于指定存放原始图像数据的文件夹位置，程序会从这个文件夹中读取图像文件。
        masks_dir (str): 图像分割的标签图像所在路径，与images_dir相对应，存放着和原始图像对应的分割标签图像（掩码图像），每个像素值表示对应的类别。
        class_values (list): 用于图像分割的所有类别数，是一个包含了需要在分割任务中关注的类别名称（字符串形式）的列表，用于从掩码图像中提取特定类别信息。
        augmentation (albumentations.Compose): 数据传输管道，是由albumentations库构建的一系列图像变换操作的组合，用于对图像和掩码进行数据增强，增加数据的多样性，提高模型的泛化能力。
        preprocessing (albumentations.Compose): 数据预处理，同样是albumentations库构建的一系列图像变换操作的组合，通常用于对图像进行归一化等预处理操作，使其符合模型输入的要求。
    """
    # CamVid数据集中用于图像分割的所有标签类别，定义了数据集中所有可能出现的类别名称。
    # 这些类别名称与掩码图像中的像素值对应，后续会通过索引来获取特定类别对应的掩码信息。
    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist', 'unlabelled']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        # 获取指定图像文件夹（images_dir）下的所有文件名（不包含路径），返回的是一个列表，列表中的每个元素是一个文件名，
        # 这些文件名将作为数据集的索引标识，用于后续根据文件名来读取对应的图像和掩码文件。
        self.ids = os.listdir(images_dir)  
        # 根据之前获取的文件名列表（self.ids），构建每个图像文件的完整路径列表。
        # os.path.join函数用于将图像文件夹路径（images_dir）和每个文件名拼接起来，形成完整的文件路径，方便后续使用cv2.imread等函数读取图像数据。
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]  
        # 与构建图像文件路径类似，根据文件名列表（self.ids）构建对应的掩码文件（标签图像）的完整路径列表，用于后续读取掩码数据。
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]  

        # 将传入的类别名称（字符串形式，存储在classes参数中）转换为对应的类别数值索引。
        # 通过遍历classes列表，对于每个类别名称，在CLASSES列表中查找其对应的索引位置（使用index方法），
        # 这样得到的索引值将用于后续从掩码图像中准确提取出对应类别的像素信息，例如提取出所有属于'car'类别的像素位置。
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]  

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # 使用cv2.imread函数读取指定索引（i）对应的图像文件，读取的图像数据默认是以BGR（蓝、绿、红）颜色通道顺序存储的，
        # 这是OpenCV库默认的图像存储格式，但在很多深度学习和常见的图像处理场景中，更常用的是RGB（红、绿、蓝）格式，所以接下来进行格式转换。
        image = cv2.imread(self.images_fps[i])  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 以灰度模式（参数0表示）读取指定索引（i）对应的掩码图像（标签图像）。
        # 掩码图像通常是单通道的，每个像素的值代表了对应的类别，例如0可能表示背景，1表示某个具体的目标类别等，具体取决于数据集的定义。
        mask = cv2.imread(self.masks_fps[i], 0)  

        # 获取图像当前的高度（h）和宽度（w），这两个维度信息将用于后续判断图像是否需要进行尺寸调整，
        # 例如进行填充操作使其满足模型输入要求（常见的是尺寸能被某些特定数值整除，方便后续的卷积等操作）。
        h, w = image.shape[:2]  
        # 计算需要填充的高度，使得高度能被32整除。
        # 首先通过整除运算（h // 32）得到当前高度包含多少个32像素的块，然后加1（如果有余数的话）再乘以32，
        # 如果当前高度本身已经能被32整除（即h % 32 == 0），则保持原高度不变，直接赋值给new_h。
        new_h = (h // 32 + 1) * 32 if h % 32!= 0 else h  
        # 与计算需要填充的高度类似，计算需要填充的宽度，使得宽度能被32整除，以满足后续模型输入或者其他处理的尺寸要求。
        new_w = (w // 32 + 1) * 32 if w % 32!= 0 else w  

        # 如果计算得到的新高度（new_h）与原高度（h）不相等，或者新宽度（new_w）与原宽度（w）不相等，说明图像的尺寸不符合要求，需要进行填充操作。
        if new_h!= h or new_w!= w:  
            # 使用cv2.copyMakeBorder函数对图像进行边界填充，使其尺寸变为满足要求的尺寸（new_h和new_w）。
            # 参数0表示在上下左右四个方向上填充的起始位置（这里都是从边界开始填充），new_h - h和new_w - w分别表示在垂直和水平方向上需要填充的像素数量，
            # cv2.BORDER_CONSTANT表示填充的模式为常数填充，value=0表示使用数值0来填充边界像素，这样可以保证填充区域的像素值统一为0，通常对于图像背景区域是合适的处理方式。
            # 同时，对掩码图像也进行同样的填充操作，以保持图像和掩码的尺寸一致，方便后续的处理和对应关系。
            image = cv2.copyMakeBorder(image, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT, value=0)  
            mask = cv2.copyMakeBorder(mask, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT, value=0)  

        # 从标签掩码中提取特定的类别对应的掩码。
        # 遍历self.class_values列表中的每个类别索引值（v），对于掩码图像（mask），通过布尔索引（mask == v）创建一个布尔类型的掩码，
        # 该掩码中对应类别像素位置为True，其他位置为False，这样就得到了每个特定类别对应的掩码信息，例如只提取出图像中所有属于'car'类别的像素位置信息。
        masks = [(mask == v) for v in self.class_values]  
        # 将提取的多个类别掩码沿着最后一维（即新增加的维度，表示不同类别）堆叠在一起，形成一个多通道的掩码图像，
        # 例如，如果有3个类别需要提取，那么堆叠后的掩码图像将是一个三维张量，最后一维的大小为3，每个通道对应一个类别的掩码信息。
        # 最后将数据类型转换为float类型，这通常是为了方便后续在深度学习模型中进行数值计算和处理，符合常见的数据格式要求。
        mask = np.stack(masks, axis=-1).astype('float')  

        # 如果定义了图像增强操作（self.augmentation不为None），则应用图像增强。
        # 通过调用self.augmentation对象的__call__方法（在albumentations库中，Compose对象可以像函数一样调用），
        # 传入图像（image）和掩码（mask）数据，按照定义好的增强操作序列对图像和掩码进行相应的变换，例如翻转、旋转、添加噪声等，
        # 增强后的数据会以字典形式返回，从中提取出增强后的图像和掩码数据更新对应的变量。
        if self.augmentation:  
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # 如果定义了图像预处理操作（self.preprocessing不为None），则应用图像预处理。
        # 与应用图像增强操作类似，通过调用self.preprocessing对象的__call__方法，传入图像和掩码数据，
        # 按照定义好的预处理操作序列对图像进行归一化等预处理，使其符合模型输入的数值范围和格式要求，更新图像和掩码变量。
        if self.preprocessing:  
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        # 返回数据集的大小，即图像的数量，通过返回self.ids列表的长度来表示，因为self.ids列表中包含了数据集中所有图像的文件名，其长度就代表了图像的总数。
        return len(self.ids)  


# ---------------------------------------------------------------
### 图像增强

# 定义获取训练集图像增强操作的函数，该函数返回一个albumentations的组合变换对象（Compose类型），
# 这个对象包含了一系列针对训练集图像的增强操作，旨在增加训练数据的多样性，帮助模型学习到更具泛化能力的特征。
def get_training_augmentation():
    train_transform = [
        # 水平翻转图像，概率为0.5。这是一种常见的数据增强方式，通过随机水平翻转图像，
        # 可以让模型学习到图像在不同方向上的特征，增加数据的多样性，同时不会改变图像中物体的语义信息，例如汽车不管是朝左还是朝右，其本质特征不变。
        albu.HorizontalFlip(p=0.5),  
        # 进行缩放、旋转和平移操作，用于模拟图像在不同视角、位置下的变化情况，进一步丰富训练数据的多样性。
        # scale_limit=0.5表示缩放比例的变化范围在0.5倍到1.5倍之间（以原始尺寸为基础），rotate_limit=0表示不进行旋转操作（这里设置为0，可根据实际需求调整旋转角度范围），
        # shift_limit=0.1表示在水平和垂直方向上平移的最大比例为图像尺寸的0.1倍，p=1表示一定应用该变换（即每次都会进行这个组合变换，只是具体的缩放、平移数值是随机的），
        # border_mode=0表示边界填充模式，用于处理在变换过程中图像超出边界的情况，这里采用默认的填充方式（可能是复制边界像素等方式，具体取决于OpenCV底层实现）。
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),  
        # 确保图像的最小高度和宽度为320，根据图像当前的尺寸情况进行填充操作，使图像尺寸满足最小要求。
        # border_mode=0同样表示边界填充模式，这里会根据需要在图像的四周填充像素，以达到指定的最小高度和宽度，保证后续处理（如裁剪等）的尺寸一致性。
        albu.PadIfNeeded(min_height=320, min_width=320, border_mode=0),  
        # 随机裁剪图像为320x320大小，从原始图像中随机选取一个320x320的区域作为裁剪后的图像，
        # 通过这种方式可以让模型学习到图像不同局部区域的特征，避免模型过度依赖于图像整体的特定布局，提高泛化能力。
        albu.RandomCrop(height=320, width=320),  
        # 添加高斯噪声，概率为0.2。高斯噪声是一种常见的模拟真实环境中噪声干扰的方式，给图像添加适量的噪声可以使模型更加鲁棒，
        # 能够在面对实际应用中可能存在的噪声情况时依然准确地进行图像分割，提高模型的泛化性能。
        albu.GaussNoise(p=0.2),  
        # 进行透视变换，概率为0.5。透视变换可以改变图像的视角，模拟不同角度拍摄或者物体在空间中不同位置的视觉效果，
        # 进一步增加训练数据的多样性，让模型学习到物体在不同透视情况下的特征，有助于提高模型在复杂场景下的分割准确性。
        albu.Perspective(p=0.5),  
        # 从以下三种对比度、亮度、伽马校正操作中随机选择一种进行应用，概率为0.9。
        # 这些操作可以改变图像的视觉效果，例如CLAHE（对比度受限的自适应直方图均衡化）可以增强图像的局部对比度，
        # RandomBrightnessContrast可以随机调整图像的亮度和对比度，RandomGamma可以对图像的灰度值进行伽马校正，
        # 通过随机选择并应用这些操作，使得模型能够适应不同视觉效果下的图像特征，增强模型的泛化能力。
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        # 从以下三种锐化、模糊、运动模糊操作中随机选择一种进行应用，概率为0.9。
        # Sharpen操作可以增强图像的边缘和细节信息，Blur操作可以使图像变得模糊，MotionBlur操作模拟物体运动产生的模糊效果，
        # 通过随机应用这些操作，让模型学习到不同清晰度状态下图像的特征，提高对各种图像质量情况的适应能力。
        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        # 从以下两种亮度对比度、色调饱和度值调整操作中随机选择一种进行应用，概率为0.9。
        # RandomBrightnessContrast再次提供了亮度和对比度的随机调整，HueSaturationValue可以对图像的色调、饱和度和亮度值进行调整，
        # 进一步丰富图像的视觉变化情况，使模型能够学习到更广泛的图像特征表示，提升在不同颜色、亮度等条件下的分割性能。
        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


# 定义获取验证集图像增强操作的函数，返回一个albumentations的组合变换对象，主要用于调整图像尺寸使其满足特定要求，
# 通常验证集不需要像训练集那样进行大量的数据增强操作，主要是保证图像尺寸等符合模型输入以及后续评估指标计算的要求。
def get_validation_augmentation():
    """调整图像使得图片的分辨率长宽能被32整除"""
    def _pad_to_divisible(image, **kwargs):
        h, w = image.shape[:2]
        new_h = (h // 32 + 1) * 32 if h % 32!= 0 else h
        new_w = (w // 32 + 1) * 32 if w % 32!= 0 else w
        return cv2.copyMakeBorder(image, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT, value=0)

    test_transform = [
        # 应用自定义的填充函数，确保图像尺寸能被32整除，使得验证集图像的尺寸与训练集处理后的尺寸要求一致，方便模型进行统一的输入和处理。
        albu.Lambda(image=_pad_to_divisible),  
    ]