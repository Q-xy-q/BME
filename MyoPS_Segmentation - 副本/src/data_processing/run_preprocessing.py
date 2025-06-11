import os
import json
import random
import yaml
from pathlib import Path


def create_data_splits(config_path='configs/data_config.yaml'):
    """
    Scans the training data directory, creates a train/validation split,
    and saves the patient IDs into JSON files.
    """
    # Load data configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    train_dir = Path(config['train_dir'])
    split_dir = Path(config['split_dir'])
    validation_split = config['validation_split']
    seed = config['seed']

    # Create the output directory if it doesn't exist
    split_dir.mkdir(parents=True, exist_ok=True)

    # Get all patient IDs (subdirectories in the training directory)
    patient_ids = [d.name for d in train_dir.iterdir() if d.is_dir()]
    patient_ids.sort()

    if not patient_ids:
        print(f"Error: No patient folders found in {train_dir}.")
        print("Please ensure your training data is structured as 'data/raw/Train/CaseXXXX/'.")
        return

    print(f"Found {len(patient_ids)} patients in {train_dir}.")

    # Set random seed for reproducibility
    random.seed(seed)
    random.shuffle(patient_ids)

    # Split IDs into training and validation sets
    split_index = int(len(patient_ids) * (1 - validation_split))
    train_ids = patient_ids[:split_index]
    val_ids = patient_ids[split_index:]

    print(f"Splitting data: {len(train_ids)} for training, {len(val_ids)} for validation.")

    # Save the splits to JSON files
    train_split_path = split_dir / 'train_ids.json'
    val_split_path = split_dir / 'val_ids.json'

    with open(train_split_path, 'w') as f:
        json.dump(train_ids, f, indent=4)

    with open(val_split_path, 'w') as f:
        json.dump(val_ids, f, indent=4)

    print(f"Train/validation splits saved successfully to {split_dir}.")


if __name__ == '__main__':
    create_data_splits()