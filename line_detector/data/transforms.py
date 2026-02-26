import math
import random

import cv2
import numpy as np


def horizontal_flip(img, lines, orig_w):
    img = cv2.flip(img, 1)
    flipped = []
    for l in lines:
        x1 = orig_w - 1 - l["x1"]
        x2 = orig_w - 1 - l["x2"]
        y1, y2 = l["y1"], l["y2"]
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dx, dy = x2 - x1, y2 - y1
        angle = math.degrees(math.atan2(abs(dx), abs(dy)))
        flipped.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                         "angle_from_vertical": round(angle, 2)})
    return img, flipped


def color_jitter(img, alpha_range=(0.7, 1.3), beta_range=(-20, 20)):
    alpha = random.uniform(*alpha_range)
    beta = random.randint(*beta_range)
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def random_crop(img, lines, orig_w, orig_h, min_factor=0.8, max_factor=1.0):
    factor = random.uniform(min_factor, max_factor)
    new_w = int(orig_w * factor)
    new_h = int(orig_h * factor)
    x_off = random.randint(0, orig_w - new_w)
    y_off = random.randint(0, orig_h - new_h)
    img = img[y_off:y_off + new_h, x_off:x_off + new_w]

    cropped = []
    for l in lines:
        x1, y1 = l["x1"] - x_off, l["y1"] - y_off
        x2, y2 = l["x2"] - x_off, l["y2"] - y_off
        if 0 <= x1 < new_w and 0 <= y1 < new_h and 0 <= x2 < new_w and 0 <= y2 < new_h:
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            dx, dy = x2 - x1, y2 - y1
            angle = math.degrees(math.atan2(abs(dx), abs(dy)))
            cropped.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                             "angle_from_vertical": round(angle, 2)})

    return img, cropped, new_w, new_h


def augment(img, lines, orig_w, orig_h,
            flip_prob=0.5, jitter_prob=0.5, crop_prob=0.3):
    if random.random() < flip_prob:
        img, lines = horizontal_flip(img, lines, orig_w)

    if random.random() < jitter_prob:
        img = color_jitter(img)

    if random.random() < crop_prob:
        img, lines, orig_w, orig_h = random_crop(img, lines, orig_w, orig_h)

    return img, lines, orig_w, orig_h
