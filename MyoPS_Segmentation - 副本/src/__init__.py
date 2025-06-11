# src/data_processing/visualization.py
import nibabel as nib
import matplotlib.pyplot as plt


def visualize_image(image_path):
    """
    可视化图像，展示其中的一部分
    """
    img = nib.load(image_path)  # 加载图像
    img_data = img.get_fdata()  # 获取图像数据

    # 选择一个切片显示
    slice_index = img_data.shape[2] // 2  # 选择中间的切片
    slice_data = img_data[:, :, slice_index]

    plt.imshow(slice_data, cmap='gray')
    plt.title(f"Slice {slice_index} of {image_path}")
    plt.show()


# 可视化某张处理后的图像
visualize_image('data/raw/Train/Case3001/Case3001_C0.nii.gz')  # 修改为您的图像路径
