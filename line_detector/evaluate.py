import argparse
import json
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader

from config import cfg
from data.dataset import VerticalLineDataset, collate_fn
from metrics.sap import evaluate_heatmaps, extract_peaks, pair_endpoints
from models.line_det import LineDetectNet
from utils.visualization import tensor_to_bgr, draw_gt_pred_side_by_side


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pth")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--images_dir", default="data/images")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vis_dir", default="eval_vis")
    parser.add_argument("--num_vis", type=int, default=20)
    parser.add_argument("--peak_threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def visualize(model, dataset, device, vis_dir, num_vis, peak_threshold):
    Path(vis_dir).mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for idx in range(min(num_vis, len(dataset))):
            sample = dataset[idx]
            output = model(sample["image"].unsqueeze(0).to(device))
            hm_size = output.shape[-1]
            pred_np = output[0].cpu().numpy()
            tx, ty, ts = extract_peaks(pred_np[0], peak_threshold)
            bx, by, bs = extract_peaks(pred_np[1], peak_threshold)
            pred_lines = pair_endpoints(tx, ty, ts, bx, by, bs, hm_size)

            img_bgr = tensor_to_bgr(sample["image"])
            gt_lines = [{"x1": g[0], "y1": g[1], "x2": g[2], "y2": g[3]}
                        for g in sample["gt_lines"]]
            combined = draw_gt_pred_side_by_side(img_bgr, gt_lines, pred_lines)
            cv2.imwrite(str(Path(vis_dir) / f"vis_{idx:04d}.jpg"), combined)

    print(f"Saved {min(num_vis, len(dataset))} visualizations to {vis_dir}/")


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = LineDetectNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint — epoch {ckpt['epoch']}")

    dataset = VerticalLineDataset(
        json_path=f"{args.data_dir}/{args.split}.json",
        images_dir=args.images_dir,
        input_size=cfg.input_size, heatmap_stride=cfg.heatmap_stride,
        is_train=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=collate_fn)
    print(f"Evaluating {len(dataset)} images ({args.split} set)\n")

    metrics = evaluate_heatmaps(model, loader, device, cfg.sap_thresholds,
                                args.peak_threshold)

    print("=" * 40)
    for k, v in metrics.items():
        print(f"  {k} = {v:.2f}")
    print("=" * 40)

    out = Path(args.checkpoint).parent / f"eval_{args.split}.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics → {out}")

    if args.visualize:
        visualize(model, dataset, device, args.vis_dir,
                  args.num_vis, args.peak_threshold)


if __name__ == "__main__":
    main()
