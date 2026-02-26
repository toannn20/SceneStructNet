import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import cfg
from data.dataset import VerticalLineDataset, collate_fn
from losses.focal import SSNLoss
from metrics.sap import evaluate_heatmaps
from models.line_det import LineDetectNet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--images_dir", default="data/images")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no_pretrained", action="store_true")
    return parser.parse_args()


def build_loaders(data_dir, images_dir, batch_size):
    train_ds = VerticalLineDataset(
        json_path=f"{data_dir}/train.json", images_dir=images_dir,
        input_size=cfg.input_size, heatmap_stride=cfg.heatmap_stride, is_train=True,
        sigma_endpoint=cfg.heatmap_sigma_endpoint, sigma_line=cfg.heatmap_sigma_line,
    )
    val_ds = VerticalLineDataset(
        json_path=f"{data_dir}/val.json", images_dir=images_dir,
        input_size=cfg.input_size, heatmap_stride=cfg.heatmap_stride, is_train=False,
        sigma_endpoint=cfg.heatmap_sigma_endpoint, sigma_line=cfg.heatmap_sigma_line,
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=cfg.num_workers, collate_fn=collate_fn,
                   pin_memory=True, drop_last=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                   num_workers=cfg.num_workers, collate_fn=collate_fn,
                   pin_memory=True),
    )


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    accum = {}
    t0 = time.time()
    for i, batch in enumerate(loader):
        pred = model(batch["image"].to(device))
        losses = criterion(pred, batch["heatmaps"].to(device))

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        for k, v in losses.items():
            accum[k] = accum.get(k, 0.0) + v.item()

        if (i + 1) % 20 == 0 or (i + 1) == len(loader):
            n = i + 1
            print(f"  [{epoch}][{n}/{len(loader)}] "
                  f"loss={accum['total']/n:.4f} "
                  f"cls={accum['loss_class']/n:.4f} "
                  f"seg={accum['loss_seg']/n:.4f} "
                  f"({time.time()-t0:.1f}s)")

    return {k: v / len(loader) for k, v in accum.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    accum = {}
    for batch in loader:
        for k, v in criterion(model(batch["image"].to(device)),
                              batch["heatmaps"].to(device)).items():
            accum[k] = accum.get(k, 0.0) + v.item()
    return {k: v / len(loader) for k, v in accum.items()}


def main():
    args = get_args()
    if args.epochs:      cfg.epochs = args.epochs
    if args.batch_size:  cfg.batch_size = args.batch_size
    if args.device:      cfg.device = args.device
    if args.checkpoint_dir: cfg.checkpoint_dir = args.checkpoint_dir

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_loaders(args.data_dir, args.images_dir, cfg.batch_size)
    print(f"Device: {device} | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    model = LineDetectNet(pretrained=not args.no_pretrained).to(device)
    criterion = SSNLoss(weight_class=cfg.loss_weight_class, weight_line=cfg.loss_weight_line,
                        alpha=cfg.loss_focal_alpha, beta=cfg.loss_focal_beta)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone_parameters(), "lr": cfg.backbone_lr},
        {"params": model.head_parameters(), "lr": cfg.base_lr},
    ], weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-6
    )

    start_epoch, best_val_loss, history = 1, float("inf"), []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        history = ckpt.get("history", [])
        print(f"Resumed from epoch {ckpt['epoch']}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_m = validate(model, val_loader, criterion, device)
        scheduler.step()

        val_loss = val_m["total"]
        row = {"epoch": epoch, "train": train_m, "val": val_m}
        history.append(row)

        ckpt = {"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "best_val_loss": best_val_loss, "history": history}

        torch.save(ckpt, cfg.last_model_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, cfg.best_model_path)
            print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
        else:
            print(f"  val_loss={val_loss:.4f} (best={best_val_loss:.4f})")

        if epoch % 10 == 0:
            sap = evaluate_heatmaps(model, val_loader, device, cfg.sap_thresholds,
                                    cfg.peak_threshold)
            print(f"  sAP5={sap['sAP5']:.1f} sAP10={sap['sAP10']:.1f} sAP15={sap['sAP15']:.1f}")
            row["sap"] = sap

        with open(f"{cfg.checkpoint_dir}/history.json", "w") as f:
            json.dump(history, f, indent=2)

    print("\nFinal evaluation:")
    sap = evaluate_heatmaps(model, val_loader, device, cfg.sap_thresholds,
                            cfg.peak_threshold)
    print(f"sAP5={sap['sAP5']:.1f} | sAP10={sap['sAP10']:.1f} | sAP15={sap['sAP15']:.1f}")


if __name__ == "__main__":
    main()
