import math
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment


def extract_peaks(heatmap, threshold=0.3, min_distance=3):
    filtered = maximum_filter(heatmap, size=min_distance * 2 + 1)
    peaks = (heatmap == filtered) & (heatmap > threshold)
    ys, xs = np.where(peaks)
    scores = heatmap[ys, xs]
    order = np.argsort(-scores)
    return xs[order], ys[order], scores[order]


def pair_endpoints(start_xs, start_ys, start_scores, end_xs, end_ys, end_scores,
                   hm_size, max_lines=200):
    lines = []
    for i in range(min(len(start_xs), max_lines)):
        for j in range(min(len(end_xs), max_lines)):
            dx = float(end_xs[j] - start_xs[i])
            dy = float(end_ys[j] - start_ys[i])
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                continue
            lines.append({
                "x1": float(start_xs[i]) / hm_size,
                "y1": float(start_ys[i]) / hm_size,
                "x2": float(end_xs[j]) / hm_size,
                "y2": float(end_ys[j]) / hm_size,
                "score": float(start_scores[i]) * float(end_scores[j]),
            })
    lines.sort(key=lambda x: -x["score"])
    return lines


def _line_distance(pred, gt, img_size=128):
    s = img_size
    d1 = (math.sqrt((pred["x1"] - gt["x1"]) ** 2 + (pred["y1"] - gt["y1"]) ** 2) +
          math.sqrt((pred["x2"] - gt["x2"]) ** 2 + (pred["y2"] - gt["y2"]) ** 2)) * s
    d2 = (math.sqrt((pred["x1"] - gt["x2"]) ** 2 + (pred["y1"] - gt["y2"]) ** 2) +
          math.sqrt((pred["x2"] - gt["x1"]) ** 2 + (pred["y2"] - gt["y1"]) ** 2)) * s
    return min(d1, d2) / 2.0


def compute_sap(all_preds, all_gts, threshold, img_size=128):
    tp_list, fp_list, scores_list = [], [], []
    total_gt = 0

    for preds, gts in zip(all_preds, all_gts):
        total_gt += len(gts)
        if not preds:
            continue
        if not gts:
            for p in preds:
                tp_list.append(0); fp_list.append(1); scores_list.append(p["score"])
            continue

        cost = np.array([[_line_distance(p, g, img_size) for g in gts] for p in preds])
        row_ind, col_ind = linear_sum_assignment(cost)
        matched = {r for r, c in zip(row_ind, col_ind) if cost[r, c] <= threshold}

        for i, p in enumerate(preds):
            scores_list.append(p["score"])
            tp_list.append(1 if i in matched else 0)
            fp_list.append(0 if i in matched else 1)

    if not total_gt or not scores_list:
        return 0.0

    order = np.argsort(-np.array(scores_list))
    tp_cs = np.cumsum(np.array(tp_list)[order])
    fp_cs = np.cumsum(np.array(fp_list)[order])
    precision = tp_cs / (tp_cs + fp_cs + 1e-6)
    recall = tp_cs / (total_gt + 1e-6)

    return float(np.trapezoid(
        np.concatenate([[1], precision]),
        np.concatenate([[0], recall])
    )) * 100.0


def evaluate_heatmaps(model, dataloader, device, thresholds=(5, 10, 15),
                      peak_threshold=0.3):
    import torch
    model.eval()
    all_preds, all_gts = [], []

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch["image"].to(device))
            hm_size = outputs.shape[-1]

            for b in range(outputs.shape[0]):
                pred_np = outputs[b].cpu().numpy()
                tx, ty, ts = extract_peaks(pred_np[0], peak_threshold)
                bx, by, bs = extract_peaks(pred_np[1], peak_threshold)
                all_preds.append(
                    pair_endpoints(tx, ty, ts, bx, by, bs, hm_size)
                )
                all_gts.append([
                    {"x1": g[0], "y1": g[1], "x2": g[2], "y2": g[3]}
                    for g in batch["gt_lines"][b]
                ])

    return {f"sAP{t}": compute_sap(all_preds, all_gts, threshold=t) for t in thresholds}
