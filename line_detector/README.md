# LineDetectNet (SSN) - Line Detection

Full pipeline for detecting lines in images, calculating tilt angles from detected endpoints,
and evaluating with structural Average Precision (sAP) metrics.

## Architecture
- **Backbone**: MobileNetV3-Large (ImageNet pretrained)
- **Neck**: Feature Pyramid Network (FPN, 256ch)
- **Head**: 3-channel heatmap (start endpoint, end endpoint, line body)
- **Loss**: Focal Loss — classification weight=4, line weight=5 (LINEA paper)
- **Metrics**: sAP5, sAP10, sAP15 (structural Average Precision)

## Setup

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Dataset Structure

Your CVAT-labeled dataset should look like:

```
dataset_root/
├── architecture_photography/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── ...
│   └── annotations.xml      # CVAT XML 1.1 export
├── city_photography/
│   ├── images/
│   └── annotations.xml
└── street_style_fashion_photography/
    ├── images/
    └── annotations.xml
```

Labels in CVAT must use polyline type with label name `vertical_line`.

## Step 1 — Prepare Data

Split 80/20 per category and convert CVAT XML → training JSON:

```bash
python prepare_data.py \
    --dataset_root /path/to/dataset_root \
    --output_dir data \
    --images_out_dir data/images \
    --train_ratio 0.8
```

Output:
- `data/train.json` — training split
- `data/val.json` — validation split
- `data/images/` — copied and organized images

## Step 2 — Train

```bash
python train.py \
    --data_dir data \
    --images_dir data/images \
    --epochs 50 \
    --batch_size 8
```

Optional flags:
- `--resume checkpoints/last.pth` — continue from checkpoint
- `--no_pretrained` — train from scratch
- `--device cpu` — force CPU

Training output:
- `checkpoints/best.pth` — best model by validation loss
- `checkpoints/last.pth` — latest checkpoint
- `checkpoints/history.json` — full loss/metric log

sAP is computed every 10 epochs during training.

## Step 3 — Evaluate

```bash
python evaluate.py \
    --checkpoint checkpoints/best.pth \
    --data_dir data \
    --images_dir data/images \
    --split val \
    --visualize \
    --vis_dir eval_vis \
    --num_vis 30
```

Output:
```
========================================
  sAP5  = 42.10
  sAP10 = 58.70
  sAP15 = 65.40
========================================
```

## Step 4 — Inference

Single image:
```bash
python inference.py \
    --checkpoint checkpoints/best.pth \
    --input /path/to/image.jpg \
    --output_dir inference_out
```

Folder:
```bash
python inference.py \
    --checkpoint checkpoints/best.pth \
    --input /path/to/images/ \
    --output_dir inference_out
```

Output per image:
```
img001.jpg: 5 lines detected
  [0] (312,45) -> (308,892)  angle=0.3°  score=0.847
  [1] (128,12) -> (134,901)  angle=0.4°  score=0.792
```

## Hyperparameters (config.py)

| Parameter | Default | Notes |
|---|---|---|
| input_size | 512 | Input resolution |
| epochs | 50 | Training epochs |
| batch_size | 8 | Adjust to VRAM |
| base_lr | 2e-4 | Head/FPN learning rate |
| backbone_lr | 1e-4 | Backbone learning rate (10x smaller) |
| weight_decay | 1e-4 | AdamW weight decay |
| loss_weight_class | 4.0 | Classification loss coeff (LINEA) |
| loss_weight_line | 5.0 | Line body loss coeff (LINEA) |
| peak_threshold | 0.3 | Heatmap peak detection threshold |

## Performance Targets (RTX 3080)

| Backbone | Latency | Notes |
|---|---|---|
| MobileNetV3 (default) | ~8ms | 125 FPS |
| ResNet18 | ~12ms | 83 FPS |
| ResNet50 | ~25ms | 40 FPS |
