from dataclasses import dataclass, field


@dataclass
class Config:
    data_root: str = "data"
    images_dir: str = "data/images"
    train_json: str = "data/train.json"
    val_json: str = "data/val.json"

    input_size: int = 512
    heatmap_stride: int = 4

    backbone: str = "mobilenetv3"
    num_queries: int = 1100
    embed_dim: int = 256
    num_decoder_layers: int = 3

    epochs: int = 50
    batch_size: int = 8
    num_workers: int = 2
    base_lr: float = 2e-4
    backbone_lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    loss_weight_line: float = 5.0
    loss_weight_class: float = 4.0
    loss_focal_alpha: int = 2
    loss_focal_beta: int = 4

    heatmap_sigma_endpoint: float = 2.0
    heatmap_sigma_line: float = 1.0
    peak_threshold: float = 0.3

    checkpoint_dir: str = "checkpoints"
    best_model_path: str = "checkpoints/best.pth"
    last_model_path: str = "checkpoints/last.pth"

    sap_thresholds: list = field(default_factory=lambda: [5, 10, 15])
    device: str = "cuda"

    split_train_ratio: float = 0.8
    categories: list = field(default_factory=lambda: [
        "architecture_photography",
        "city_photography",
        "street_style_fashion_photography"
    ])


cfg = Config()
