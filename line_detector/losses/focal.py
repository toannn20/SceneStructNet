import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()

        pos_loss = torch.log(pred + 1e-6) * torch.pow(1 - pred, self.alpha) * pos_mask
        neg_loss = (torch.log(1 - pred + 1e-6) *
                    torch.pow(pred, self.alpha) *
                    torch.pow(1 - target, self.beta) * neg_mask)

        num_pos = pos_mask.sum()
        if num_pos == 0:
            return -(neg_loss.sum())
        return -(pos_loss.sum() + neg_loss.sum()) / num_pos


class SSNLoss(nn.Module):
    def __init__(self, weight_class=4.0, weight_line=5.0, alpha=2, beta=4):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, beta=beta)
        self.weight_class = weight_class
        self.weight_line = weight_line

    def forward(self, pred, target_heatmaps):
        loss_top = self.focal(pred[:, 0:1], target_heatmaps[:, 0:1])
        loss_bot = self.focal(pred[:, 1:2], target_heatmaps[:, 1:2])
        loss_line = self.focal(pred[:, 2:3], target_heatmaps[:, 2:3])

        loss_class = self.weight_class * (loss_top + loss_bot)
        loss_seg = self.weight_line * loss_line

        return {
            "total": loss_class + loss_seg,
            "loss_top": loss_top,
            "loss_bot": loss_bot,
            "loss_line": loss_line,
            "loss_class": loss_class,
            "loss_seg": loss_seg,
        }
