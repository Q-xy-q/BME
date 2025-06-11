import torch
import torch.nn as nn

class DiceCELoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_ce=0.5):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        num_classes = preds.shape[1]

        # Clamp target values to valid range
        targets = torch.clamp(targets, min=0, max=num_classes - 1)

        dice_loss = self.dice(preds, targets)
        ce_loss = self.ce(preds, targets)

        return self.weight_dice * dice_loss + self.weight_ce * ce_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        num_classes = preds.shape[1]
        preds = torch.softmax(preds, dim=1)

        # 🔧 去掉 channel 维度
        targets = targets.squeeze(1)  # [B, D, H, W]
        targets = torch.clamp(targets, 0, num_classes - 1)  # 避免非法标签

        targets_onehot = torch.nn.functional.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()
        targets = targets.squeeze(1)
        intersection = torch.sum(preds * targets_onehot)
        union = torch.sum(preds + targets_onehot)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice

