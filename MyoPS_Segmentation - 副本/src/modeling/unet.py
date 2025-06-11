from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
    Dropout,
    BatchNormalization,
    Activation
)


def conv_block(input_tensor, num_filters):
    """一个标准的卷积块，包含两个卷积层"""
    x = Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def build_unet(input_shape, num_classes):
    """
    构建一个标准的U-Net模型。

    Args:
        input_shape (tuple): 模型输入尺寸，例如 (256, 256, 3)。
        num_classes (int): 分割的类别总数。

    Returns:
        A Keras Model instance.
    """
    inputs = Input(input_shape)

    # --- 编码器 (收缩路径) ---
    # 第1层
    c1 = conv_block(inputs, 16)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(0.1)(p1)

    # 第2层
    c2 = conv_block(p1, 32)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(0.1)(p2)

    # 第3层
    c3 = conv_block(p2, 64)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(0.2)(p3)

    # 第4层
    c4 = conv_block(p3, 128)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(0.2)(p4)

    # --- 瓶颈层 ---
    bn = conv_block(p4, 256)

    # --- 解码器 (扩张路径) ---
    # 第6层
    u6 = UpSampling2D((2, 2))(bn)
    u6 = concatenate([u6, c4])  # 跳跃连接
    c6 = conv_block(u6, 128)
    c6 = Dropout(0.2)(c6)

    # 第7层
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])  # 跳跃连接
    c7 = conv_block(u7, 64)
    c7 = Dropout(0.2)(c7)

    # 第8层
    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])  # 跳跃连接
    c8 = conv_block(u8, 32)
    c8 = Dropout(0.1)(c8)

    # 第9层
    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])  # 跳跃连接
    c9 = conv_block(u9, 16)
    c9 = Dropout(0.1)(c9)

    # --- 输出层 ---
    # 使用1x1卷积将特征图转换为最终的分割掩码
    # 使用sigmoid激活函数，因为我们是多标签分割（每个通道是一个二元分割问题）
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)

    # 创建并返回模型
    model = Model(inputs=[inputs], outputs=[outputs], name='UNet')

    return model
