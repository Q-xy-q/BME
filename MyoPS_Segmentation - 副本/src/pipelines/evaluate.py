import os
import torch
from torch.utils.data import DataLoader
from src.data_processing.loader import HDF5Dataset
from src.modeling.backbone import get_model
from src.modeling.metrics import MetricEvaluator
from src.data_processing.loader import load_case_list

def evaluate_pipeline(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_list = load_case_list(cfg['val_split'])
    val_dataset = HDF5Dataset(cfg['val_h5'], val_list)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = get_model(cfg['model_name'], cfg['in_channels'], cfg['out_channels']).to(device)
    model.load_state_dict(torch.load(cfg['ckpt_path']))
    model.eval()

    evaluator = MetricEvaluator(cfg['out_channels'])
    evaluator.reset()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            evaluator(outputs, labels)

    dice_scores = evaluator.dice_metric.aggregate(reduction=None).cpu().numpy()
    for i, score in enumerate(dice_scores):
        print(f"Class {i+1} Dice: {score:.4f}")
    print(f"Mean Dice: {dice_scores.mean():.4f}")

if __name__ == "__main__":
    import yaml
    with open("configs/train_config.yaml") as f:
        cfg = yaml.safe_load(f)
    evaluate_pipeline(cfg)
