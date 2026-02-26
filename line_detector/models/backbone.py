import torch.nn as nn
from torchvision import models


class MobileNetV3Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        )
        features = base.features
        self.stage0 = features[:4]
        self.stage1 = features[4:7]
        self.stage2 = features[7:13]
        self.stage3 = features[13:]
        self.out_channels = [24, 40, 112, 960]

    def forward(self, x):
        c1 = self.stage0(x)
        c2 = self.stage1(c1)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        return c1, c2, c3, c4
