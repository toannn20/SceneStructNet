import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from data.transforms import augment

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def draw_gaussian(heatmap, cx, cy, sigma):
    h, w = heatmap.shape
    size = int(6 * sigma + 1)
    x0, y0 = int(cx - size // 2), int(cy - size // 2)
    x_min, x_max = max(0, x0), min(w, x0 + size)
    y_min, y_max = max(0, y0), min(h, y0 + size)
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            val = math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
            if val > heatmap[y, x]:
                heatmap[y, x] = val


def draw_line_gaussian(heatmap, x1, y1, x2, y2, sigma=1.0, num_points=50):
    for i in range(num_points + 1):
        t = i / num_points
        draw_gaussian(heatmap, x1 + t * (x2 - x1), y1 + t * (y2 - y1), sigma)


class VerticalLineDataset(Dataset):
    def __init__(self, json_path, images_dir, input_size=512,
                 heatmap_stride=4, is_train=True,
                 sigma_endpoint=2.0, sigma_line=1.0):
        with open(json_path) as f:
            self.records = json.load(f)
        self.images_dir = Path(images_dir)
        self.input_size = input_size
        self.hm_size = input_size // heatmap_stride
        self.is_train = is_train
        self.sigma_endpoint = sigma_endpoint
        self.sigma_line = sigma_line
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.records)

    def _load_image(self, filename):
        img = cv2.imread(str(self.images_dir / filename))
        if img is None:
            img = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _build_heatmaps(self, lines, scale_x, scale_y):
        hm_scale = self.hm_size / self.input_size
        hm_start = np.zeros((self.hm_size, self.hm_size), dtype=np.float32)
        hm_end = np.zeros((self.hm_size, self.hm_size), dtype=np.float32)
        hm_line = np.zeros((self.hm_size, self.hm_size), dtype=np.float32)
        for l in lines:
            x1h = l["x1"] * scale_x * hm_scale
            y1h = l["y1"] * scale_y * hm_scale
            x2h = l["x2"] * scale_x * hm_scale
            y2h = l["y2"] * scale_y * hm_scale
            draw_gaussian(hm_start, x1h, y1h, self.sigma_endpoint)
            draw_gaussian(hm_end, x2h, y2h, self.sigma_endpoint)
            draw_line_gaussian(hm_line, x1h, y1h, x2h, y2h, self.sigma_line)
        return hm_start, hm_end, hm_line

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = self._load_image(rec["filename"])
        orig_h, orig_w = img.shape[:2]
        lines = [dict(l) for l in rec["lines"]]

        if self.is_train:
            img, lines, orig_w, orig_h = augment(img, lines, orig_w, orig_h)

        img = cv2.resize(img, (self.input_size, self.input_size))
        scale_x = self.input_size / orig_w
        scale_y = self.input_size / orig_h

        hm_start, hm_end, hm_line = self._build_heatmaps(lines, scale_x, scale_y)

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = self.normalize(img_t)
        heatmaps = torch.stack([
            torch.from_numpy(hm_start),
            torch.from_numpy(hm_end),
            torch.from_numpy(hm_line),
        ])

        gt_lines = [
            [l["x1"] * scale_x / self.input_size,
             l["y1"] * scale_y / self.input_size,
             l["x2"] * scale_x / self.input_size,
             l["y2"] * scale_y / self.input_size]
            for l in lines
        ]

        return {
            "image": img_t,
            "heatmaps": heatmaps,
            "gt_lines": gt_lines,
            "filename": rec["filename"],
            "orig_size": (rec["width"], rec["height"]),
        }


def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "heatmaps": torch.stack([b["heatmaps"] for b in batch]),
        "gt_lines": [b["gt_lines"] for b in batch],
        "filename": [b["filename"] for b in batch],
        "orig_size": [b["orig_size"] for b in batch],
    }
