import torch
import torch.nn as nn


class LineHead(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.hm_start = nn.Conv2d(64, 1, 1)
        self.hm_end = nn.Conv2d(64, 1, 1)
        self.hm_line = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.shared(x)
        return torch.cat([
            torch.sigmoid(self.hm_start(x)),
            torch.sigmoid(self.hm_end(x)),
            torch.sigmoid(self.hm_line(x)),
        ], dim=1)
