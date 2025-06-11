import torch
import torch.nn as nn
from monai.networks.nets import UNet

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = UNet(
            dimensions=3,  # 正确传递dimensions=3
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2
        )

    def forward(self, x):
        return self.model(x)

def get_model(name, in_channels, out_channels):
    if name == 'unet':
        return UNet3D(in_channels, out_channels)
    else:
        raise ValueError(f"Unknown model: {name}")
