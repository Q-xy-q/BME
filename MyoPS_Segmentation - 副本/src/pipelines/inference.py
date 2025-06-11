import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader
from src.data_processing.loader import HDF5Dataset
from src.modeling.backbone import get_model
from src.data_processing.loader import load_case_list

def save_nifti_prediction(pred_array, affine, save_path):
    pred_img = nib.Nifti1Image(pred_array.astype(np.uint8), affine)
    nib.save(pred_img, save_path)

def inference_pipeline(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_list = load_case_list(cfg['test_split'])
    test_dataset = HDF5Dataset(cfg['test_h5'], test_list)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = get_model(cfg['model_name'], cfg['in_channels'], cfg['out_channels']).to(device)
    model.load_state_dict(torch.load(cfg['ckpt_path']))
    model.eval()

    os.makedirs(cfg['save_pred_dir'], exist_ok=True)

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()  # [H, W, D]

            case_id = test_list[i]  # 使用文件名作为 ID
            affine = np.eye(4)  # 若需要保持原始 affine，可通过数据预处理传递
            save_path = os.path.join(cfg['save_pred_dir'], f"{case_id}_pred.nii.gz")
            save_nifti_prediction(preds, affine, save_path)
            print(f"Saved: {save_path}")

if __name__ == "__main__":
    import yaml
    with open("configs/train_config.yaml") as f:
        cfg = yaml.safe_load(f)
    inference_pipeline(cfg)
