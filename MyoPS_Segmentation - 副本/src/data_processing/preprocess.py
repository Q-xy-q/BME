import numpy as np
import nibabel as nib
import tensorflow as tf


def load_nii_as_array(path: str) -> np.ndarray:
    """Loads a .nii.gz file and returns it as a numpy array."""
    return nib.load(path).get_fdata()


def center_crop(img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Performs a center crop on the input image.

    Args:
        img: The input image as a numpy array.
        target_size: A tuple (height, width) for the crop dimensions.

    Returns:
        The cropped image.
    """
    h, w = img.shape[:2]
    th, tw = target_size

    if h < th or w < tw:
        raise ValueError(f"Target size ({th}, {tw}) is larger than image size ({h}, {w}).")

    start_x = (w - tw) // 2
    start_y = (h - th) // 2

    return img[start_y:start_y + th, start_x:start_x + tw]


def preprocess_image(c0_img: np.ndarray, lge_img: np.ndarray, t2_img: np.ndarray,
                     target_size: tuple[int, int]) -> np.ndarray:
    """
    Preprocesses a set of three single-channel images for the model.

    Args:
        c0_img, lge_img, t2_img: The three input image channels as 2D numpy arrays.
        target_size: The target size for center cropping.

    Returns:
        A preprocessed 3-channel image as a numpy array.
    """
    # Stack images into a 3-channel image
    full_img = np.stack([c0_img, lge_img, t2_img], axis=-1)

    # Crop to target size
    cropped_img = center_crop(full_img, target_size)

    # Z-score normalization
    mean = np.mean(cropped_img)
    std = np.std(cropped_img)
    epsilon = 1e-8  # To avoid division by zero

    normalized_img = (cropped_img - mean) / (std + epsilon)

    return normalized_img.astype(np.float32)


def preprocess_mask(mask: np.ndarray, target_size: tuple[int, int], label_map: dict, num_classes: int) -> np.ndarray:
    """
    Preprocesses a ground truth mask for the model.

    Args:
        mask: The ground truth mask as a 2D numpy array with original labels.
        target_size: The target size for center cropping.
        label_map: A dictionary mapping original labels to categorical indices.
        num_classes: The total number of classes (including background).

    Returns:
        A one-hot encoded mask.
    """
    # Crop to target size
    cropped_mask = center_crop(mask, target_size)

    # Map original labels (200, 500, etc.) to categorical indices (1, 2, 3...)
    output_mask = np.zeros_like(cropped_mask, dtype=np.int32)
    for original_label, new_label in label_map.items():
        output_mask[cropped_mask == original_label] = new_label

    # Convert to one-hot encoding
    one_hot_mask = tf.keras.utils.to_categorical(output_mask, num_classes=num_classes)

    return one_hot_mask.astype(np.float32)
