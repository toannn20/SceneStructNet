import math

import cv2
import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def tensor_to_bgr(image_tensor):
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = np.clip((img * IMAGENET_STD + IMAGENET_MEAN) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def draw_lines(img_bgr, lines, color=(0, 255, 0), thickness=2, show_score=True):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    for line in lines:
        x1, y1 = int(line["x1"] * w), int(line["y1"] * h)
        x2, y2 = int(line["x2"] * w), int(line["y2"] * h)
        cv2.line(out, (x1, y1), (x2, y2), color, thickness)
        cv2.circle(out, (x1, y1), 4, (0, 0, 255), -1)
        cv2.circle(out, (x2, y2), 4, (255, 0, 0), -1)
        if show_score and "score" in line:
            angle = line.get("angle_from_vertical",
                             math.degrees(math.atan2(abs(x2 - x1), abs(y2 - y1))))
            cv2.putText(out, f"{angle:.1f}° {line['score']:.2f}",
                        (x1 + 4, y1 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return out


def draw_gt_pred_side_by_side(img_bgr, gt_lines, pred_lines, max_preds=50):
    gt_img = draw_lines(img_bgr, gt_lines, color=(0, 255, 0), show_score=False)
    pred_img = draw_lines(img_bgr, pred_lines[:max_preds], color=(0, 0, 255))
    combined = np.concatenate([gt_img, pred_img], axis=1)
    w = img_bgr.shape[1]
    cv2.putText(combined, f"GT ({len(gt_lines)})", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(combined, f"Pred ({len(pred_lines)})", (w + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return combined
