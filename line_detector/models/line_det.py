import torch.nn as nn

from models.backbone import MobileNetV3Backbone
from models.fpn import FPN
from models.head import LineHead


class LineDetectNet(nn.Module):
    def __init__(self, pretrained=True, fpn_channels=256):
        super().__init__()
        self.backbone = MobileNetV3Backbone(pretrained=pretrained)
        self.fpn = FPN(self.backbone.out_channels, fpn_channels)
        self.head = LineHead(fpn_channels)

    def forward(self, x):
        features = self.backbone(x)
        fpn_outs = self.fpn(features)
        return self.head(fpn_outs[0])

    def backbone_parameters(self):
        return list(self.backbone.parameters())

    def head_parameters(self):
        return list(self.fpn.parameters()) + list(self.head.parameters())
