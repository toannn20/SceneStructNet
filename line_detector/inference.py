import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from config import cfg
from metrics.sap import extract_peaks, pair_endpoints
from models.line_det import LineDetectNet
from utils.visualization import draw_lines


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def preprocess(img_bgr, input_size):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size, input_size)).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)


def infer(model, img_bgr, device, input_size=512,
          peak_threshold=0.3):
    orig_h, orig_w = img_bgr.shape[:2]
    output = model(preprocess(img_bgr, input_size).to(device))
    hm_size = output.shape[-1]
    pred_np = output[0].cpu().numpy()

    tx, ty, ts = extract_peaks(pred_np[0], peak_threshold)
    bx, by, bs = extract_peaks(pred_np[1], peak_threshold)
    lines = pair_endpoints(tx, ty, ts, bx, by, bs, hm_size)

    result = []
    for l in lines:
        x1, y1 = l["x1"] * orig_w, l["y1"] * orig_h
        x2, y2 = l["x2"] * orig_w, l["y2"] * orig_h
        angle = math.degrees(math.atan2(abs(x2 - x1), abs(y2 - y1)))
        result.append({
            "x1": round(x1, 1), "y1": round(y1, 1),
            "x2": round(x2, 1), "y2": round(y2, 1),
            "angle_from_vertical": round(angle, 2),
            "score": round(l["score"], 4),
        })
    return result


def pixel_to_normalized(lines, orig_w, orig_h):
    return [{**l,
             "x1": l["x1"] / orig_w, "y1": l["y1"] / orig_h,
             "x2": l["x2"] / orig_w, "y2": l["y2"] / orig_h}
            for l in lines]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pth")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="inference_out")
    parser.add_argument("--peak_threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_vis", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = LineDetectNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded (epoch {ckpt['epoch']})")

    input_path = Path(args.input)
    image_paths = ([input_path] if input_path.is_file() else
                   [p for p in input_path.rglob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])

    if not args.no_vis:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            orig_h, orig_w = img.shape[:2]
            lines = infer(model, img, device, cfg.input_size,
                          args.peak_threshold)

            print(f"{img_path.name}: {len(lines)} lines")
            for i, l in enumerate(lines):
                print(f"  [{i}] ({l['x1']:.0f},{l['y1']:.0f})"
                      f"→({l['x2']:.0f},{l['y2']:.0f}) "
                      f"angle={l['angle_from_vertical']:.1f}° score={l['score']:.3f}")

            if not args.no_vis:
                norm_lines = pixel_to_normalized(lines, orig_w, orig_h)
                vis = draw_lines(img, norm_lines)
                cv2.imwrite(str(Path(args.output_dir) / img_path.name), vis)


if __name__ == "__main__":
    main()
