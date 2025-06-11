import imgaug.augmenters as iaa


def get_augmentation_pipeline():
    """
    定义数据增强流程。
    这个流程会同时应用到图像和对应的分割掩码上。
    """
    # 使用imgaug定义一系列增强操作
    # Sequential表示按顺序应用这些操作
    # sometimes表示有50%的概率应用其中的一组增强
    seq = iaa.Sequential([
        # 50%的概率水平翻转图像
        iaa.Fliplr(0.5),

        # 50%的概率应用仿射变换
        iaa.Sometimes(
            0.5,
            iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},  # 缩放90%-110%
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # 平移-10%到10%
                rotate=(-15, 15),  # 旋转-15度到15度
                shear=(-8, 8),  # 剪切-8度到8度
                order=[0, 1],  # 使用最近邻或双线性插值
                cval=0,  # 填充值为0
                mode='constant'
            )
        ),

        # 50%的概率应用以下一项或多项增强
        iaa.SomeOf((0, 3),
                   [
                       # 弹性变形，模拟组织的非刚性形变
                       iaa.ElasticTransformation(alpha=(0.2, 0.8), sigma=0.25),
                       # 添加高斯噪声
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                       # 随机改变图像亮度
                       iaa.Multiply((0.8, 1.2), per_channel=0.2)
                   ],
                   # 对选中的增强器随机排序
                   random_order=True
                   )
    ], random_order=False)  # 整体流程按固定顺序执行

    return seq