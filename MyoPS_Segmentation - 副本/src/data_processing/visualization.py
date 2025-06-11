import yaml
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_processing.loader import CardiacDataGenerator


def visualize_data_generator_output(
        data_config_path='configs/data_config.yaml',
        train_config_path='configs/train_config.yaml',
        split_file='train_ids.json',
        num_samples_to_show=4
):
    """
    Loads a batch of data using the CardiacDataGenerator and visualizes
    the preprocessed images and masks.
    """
    # Load configs
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    with open(train_config_path, 'r') as f:
        train_config = yaml.safe_load(f)

    split_dir = Path(data_config['split_dir'])
    batch_size = train_config['batch_size']

    if num_samples_to_show > batch_size:
        print(f"Warning: Number of samples to show ({num_samples_to_show}) is greater than batch size ({batch_size}).")
        print(f"Adjusting to show {batch_size} samples.")
        num_samples_to_show = batch_size

    # Load patient IDs
    with open(split_dir / split_file, 'r') as f:
        patient_ids = json.load(f)

    print("Initializing Data Generator for visualization...")
    # Initialize generator (shuffle=False to get consistent results for debugging)
    generator = CardiacDataGenerator(
        patient_ids=patient_ids,
        data_config_path=data_config_path,
        batch_size=batch_size,
        shuffle=False
    )

    print("Loading one batch of data...")
    # Get one batch of data
    X_batch, y_batch = generator[0]

    # Convert one-hot encoded masks back to single-channel label maps for visualization
    y_batch_labels = np.argmax(y_batch, axis=-1)

    print("Generating plots...")
    fig, axes = plt.subplots(num_samples_to_show, data_config['n_channels'] + 1, figsize=(15, 4 * num_samples_to_show))
    fig.suptitle("Data Generator Output Verification", fontsize=16)

    for i in range(num_samples_to_show):
        # --- Visualize Image Channels ---
        # The image was Z-score normalized. We'll rescale it to [0,1] for better visualization.
        img_sample = X_batch[i]
        img_display = (img_sample - img_sample.min()) / (img_sample.max() - img_sample.min() + 1e-8)

        # C0 Channel
        ax = axes[i, 0]
        ax.imshow(img_display[:, :, 0], cmap='gray')
        ax.set_title(f"Sample {i + 1}: Processed C0")
        ax.axis('off')

        # LGE Channel
        ax = axes[i, 1]
        ax.imshow(img_display[:, :, 1], cmap='gray')
        ax.set_title(f"Sample {i + 1}: Processed LGE")
        ax.axis('off')

        # T2 Channel
        ax = axes[i, 2]
        ax.imshow(img_display[:, :, 2], cmap='gray')
        ax.set_title(f"Sample {i + 1}: Processed T2")
        ax.axis('off')

        # --- Visualize Ground Truth Mask ---
        ax = axes[i, 3]
        # Using a discrete colormap
        cmap = plt.get_cmap('tab10', data_config['n_classes'])
        im = ax.imshow(y_batch_labels[i], cmap=cmap, interpolation='none', vmin=0, vmax=data_config['n_classes'] - 1)
        ax.set_title(f"Sample {i + 1}: Processed GT Mask")
        ax.axis('off')

    # Add a single colorbar for the masks
    cbar = fig.colorbar(im, ax=axes[:, -1].ravel().tolist(), ticks=np.arange(data_config['n_classes']), fraction=0.05,
                        pad=0.04)
    cbar.set_label("Class Labels (0=BG, 1=RV, 2=LV-Pool, 3=LV-Myo, 4=Edema, 5=Scar)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    visualize_data_generator_output()
