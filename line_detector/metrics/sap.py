import math
import numpy as np
from scipy.ndimage import maximum_filter


def extract_peaks(heatmap, threshold=0.3, min_distance=3):
    filtered = maximum_filter(heatmap, size=min_distance * 2 + 1)
    peaks = (heatmap == filtered) & (heatmap > threshold)
    ys, xs = np.where(peaks)
    scores = heatmap[ys, xs]
    order = np.argsort(-scores)
    return xs[order], ys[order], scores[order]


def pair_endpoints(start_xs, start_ys, start_scores, end_xs, end_ys, end_scores,
                   hm_size, max_lines=200, top_k=300):
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
    return lines[:top_k]


def _line_distance(pred, gt, scale=128.0):
    """Squared endpoint distance with min over both orderings (matches LINEA).
    
    Coordinates are scaled from [0,1] to [0, scale] before computing distance,
    so thresholds (5, 10, 15) are in the scaled pixel space.
    """
    px1, py1 = pred["x1"] * scale, pred["y1"] * scale
    px2, py2 = pred["x2"] * scale, pred["y2"] * scale
    gx1, gy1 = gt["x1"] * scale, gt["y1"] * scale
    gx2, gy2 = gt["x2"] * scale, gt["y2"] * scale

    d1 = (px1 - gx1) ** 2 + (py1 - gy1) ** 2 + (px2 - gx2) ** 2 + (py2 - gy2) ** 2
    d2 = (px1 - gx2) ** 2 + (py1 - gy2) ** 2 + (px2 - gx1) ** 2 + (py2 - gy1) ** 2
    return min(d1, d2)


def _greedy_matching(preds, gts, threshold):
    """Greedy TP/FP assignment in score-sorted order (matches LINEA's msTPFP).

    For each pred (already score-sorted), find the closest GT.
    If distance < threshold and that GT hasn't been matched yet, it's a TP.
    Returns tp, fp arrays and the minimum distance found.
    """
    n_pred = len(preds)
    tp = np.zeros(n_pred)
    fp = np.zeros(n_pred)

    if not gts:
        fp[:] = 1
        return tp, fp, float("inf")

    # Precompute distance matrix and find closest GT for each pred
    n_gt = len(gts)
    dist_matrix = np.array([[_line_distance(p, g) for g in gts] for p in preds])
    closest_gt = np.argmin(dist_matrix, axis=1)
    closest_dist = dist_matrix[np.arange(n_pred), closest_gt]
    min_dist = float(closest_dist.min()) if len(closest_dist) > 0 else float("inf")

    hit = np.zeros(n_gt, dtype=bool)
    for i in range(n_pred):
        if closest_dist[i] < threshold and not hit[closest_gt[i]]:
            hit[closest_gt[i]] = True
            tp[i] = 1
        else:
            fp[i] = 1

    return tp, fp, min_dist


def _compute_ap(tp, fp):
    """VOC-style AP with envelope precision (matches LINEA's ap method)."""
    recall = tp.copy()
    precision = tp / np.maximum(tp + fp, 1e-9)

    # Prepend (0,0) and append (1,0) sentinels
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[0.0], precision, [0.0]])

    # Make precision monotonically decreasing (envelope)
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    # Sum rectangular areas where recall changes
    idx = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1])
    return float(ap)


def compute_sap(all_preds, all_gts, threshold):
    """Compute structural AP at a given threshold (matches LINEA evaluation)."""
    all_tp, all_fp, all_scores = [], [], []
    n_gt_total = 0
    min_dist_overall = float("inf")

    for preds, gts in zip(all_preds, all_gts):
        n_gt_total += len(gts)
        if not preds:
            continue

        # preds are already sorted by score (from pair_endpoints)
        scores = np.array([p["score"] for p in preds])
        tp, fp, min_dist = _greedy_matching(preds, gts, threshold)

        if min_dist < min_dist_overall:
            min_dist_overall = min_dist

        all_scores.append(scores)
        all_tp.append(tp)
        all_fp.append(fp)

    if n_gt_total == 0 or not all_scores:
        return 0.0, min_dist_overall

    # Concatenate across all images and sort globally by score
    scores = np.concatenate(all_scores)
    tp = np.concatenate(all_tp)
    fp = np.concatenate(all_fp)

    order = np.argsort(-scores)
    tp_cum = np.cumsum(tp[order]) / n_gt_total
    fp_cum = np.cumsum(fp[order]) / n_gt_total

    return _compute_ap(tp_cum, fp_cum) * 100.0, min_dist_overall


def evaluate_heatmaps(model, dataloader, device, thresholds=(5, 10, 15),
                      peak_threshold=0.3):
    import torch
    from tqdm import tqdm
    model.eval()
    all_preds, all_gts = [], []
    total_start_peaks, total_end_peaks, total_lines = 0, 0, 0
    hm_max_start, hm_max_end = 0.0, 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  sAP  ", leave=True,
                          bar_format="{l_bar}{bar:25}{r_bar}"):
            outputs = model(batch["image"].to(device))
            hm_size = outputs.shape[-1]

            for b in range(outputs.shape[0]):
                pred_np = outputs[b].cpu().numpy()
                hm_max_start = max(hm_max_start, float(pred_np[0].max()))
                hm_max_end = max(hm_max_end, float(pred_np[1].max()))

                tx, ty, ts = extract_peaks(pred_np[0], peak_threshold)
                bx, by, bs = extract_peaks(pred_np[1], peak_threshold)
                total_start_peaks += len(tx)
                total_end_peaks += len(bx)

                lines = pair_endpoints(tx, ty, ts, bx, by, bs, hm_size)
                total_lines += len(lines)
                all_preds.append(lines)
                all_gts.append([
                    {"x1": g[0], "y1": g[1], "x2": g[2], "y2": g[3]}
                    for g in batch["gt_lines"][b]
                ])

    n_imgs = len(all_preds)
    n_gt = sum(len(g) for g in all_gts)
    print(f"    heatmap_max: start={hm_max_start:.3f} end={hm_max_end:.3f} | "
          f"peaks: {total_start_peaks}+{total_end_peaks} | "
          f"lines: {total_lines} | gt: {n_gt} ({n_imgs} imgs)")

    results = {}
    for t in thresholds:
        sap_val, min_dist = compute_sap(all_preds, all_gts, threshold=t)
        results[f"sAP{t}"] = sap_val
        print(f"    sAP{t}: {sap_val:.1f}  (min_dist={min_dist:.2f}, threshold={t})")

    return results

