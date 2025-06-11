import tensorflow as tf
from tensorflow.keras import backend as K


def dice_coefficient(y_true, y_pred, smooth=1e-5):
    """
    计算所有类别的平均Dice系数。
    y_true 和 y_pred 都应该是 one-hot 编码格式，形状为 (batch_size, H, W, num_classes)
    """
    # 预测结果是sigmoid输出，范围(0,1)，真实标签是0或1
    # 我们需要确保y_true是浮点数类型以进行乘法
    y_true = tf.cast(y_true, tf.float32)

    # 忽略背景通道（通道0）的计算，因为它通常会不成比例地拉高分数
    y_true_fg = y_true[..., 1:]
    y_pred_fg = y_pred[..., 1:]

    # 计算交集和并集
    # axes=[0,1,2] 表示在批次、高、宽维度上求和，保留类别维度
    intersection = K.sum(y_true_fg * y_pred_fg, axis=[0, 1, 2])
    union = K.sum(y_true_fg, axis=[0, 1, 2]) + K.sum(y_pred_fg, axis=[0, 1, 2])

    # 计算每个类别的dice分数，然后取平均
    dice = K.mean((2. * intersection + smooth) / (union + smooth))

    return dice


def dice_loss(y_true, y_pred):
    """
    Dice损失函数，其目标是最大化Dice系数。
    """
    return 1 - dice_coefficient(y_true, y_pred)
