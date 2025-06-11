import os
import json
import yaml
import numpy as np
import tensorflow as tf
from pathlib import Path

from src.data_processing.preprocess import load_nii_as_array, preprocess_image, preprocess_mask


class CardiacDataGenerator(tf.keras.utils.Sequence):
    """
    Keras Sequence for loading and preprocessing cardiac MRI data on-the-fly.
    """

    def __init__(self, patient_ids: list, data_config_path: str, batch_size: int, shuffle: bool = True):
        """
        Initialization.
        Args:
            patient_ids: List of patient IDs for this generator.
            data_config_path: Path to the data configuration YAML file.
            batch_size: The size of each batch.
            shuffle: Whether to shuffle the data at the end of each epoch.
        """
        self.patient_ids = patient_ids
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Load data configuration
        with open(data_config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.train_dir = Path(self.config['train_dir'])
        self.target_size = tuple(self.config['img_size'])
        self.label_map = self.config['label_map']
        self.n_classes = self.config['n_classes']
        self.n_channels = self.config['n_channels']

        self.samples = self._create_sample_list()
        self.on_epoch_end()

    def _create_sample_list(self) -> list:
        """Creates a list of all (patient_id, slice_index) samples."""
        samples = []
        print(f"Creating sample list for {len(self.patient_ids)} patients...")
        for patient_id in self.patient_ids:
            gd_path = self.train_dir / patient_id / f"{patient_id}{self.config['gd_suffix']}"
            if not gd_path.exists():
                print(f"Warning: Ground truth for {patient_id} not found. Skipping.")
                continue

            # Load the ground truth to find the number of slices
            gd_img = load_nii_as_array(str(gd_path))
            num_slices = gd_img.shape[2]

            for slice_idx in range(num_slices):
                samples.append((patient_id, slice_idx))
        print(f"Found {len(samples)} total slices.")
        return samples

    def __len__(self) -> int:
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.samples) / self.batch_size))

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate one batch of data."""
        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of samples for this batch
        batch_samples = [self.samples[k] for k in batch_indexes]

        # Generate data
        X, y = self._generate_data(batch_samples)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate_data(self, batch_samples: list) -> tuple[np.ndarray, np.ndarray]:
        """Generates data containing batch_size samples."""
        # Initialization
        X = np.empty((self.batch_size, *self.target_size, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size, *self.target_size, self.n_classes), dtype=np.float32)

        for i, (patient_id, slice_idx) in enumerate(batch_samples):
            # Construct paths
            patient_dir = self.train_dir / patient_id
            c0_path = patient_dir / f"{patient_id}{self.config['c0_suffix']}"
            lge_path = patient_dir / f"{patient_id}{self.config['lge_suffix']}"
            t2_path = patient_dir / f"{patient_id}{self.config['t2_suffix']}"
            gd_path = patient_dir / f"{patient_id}{self.config['gd_suffix']}"

            # Load 3D volumes
            c0_vol = load_nii_as_array(str(c0_path))
            lge_vol = load_nii_as_array(str(lge_path))
            t2_vol = load_nii_as_array(str(t2_path))
            gd_vol = load_nii_as_array(str(gd_path))

            # Extract 2D slices
            c0_slice = c0_vol[:, :, slice_idx]
            lge_slice = lge_vol[:, :, slice_idx]
            t2_slice = t2_vol[:, :, slice_idx]
            gd_slice = gd_vol[:, :, slice_idx]

            # Preprocess and store
            X[i,] = preprocess_image(c0_slice, lge_slice, t2_slice, self.target_size)
            y[i,] = preprocess_mask(gd_slice, self.target_size, self.label_map, self.n_classes)

        return X, y
