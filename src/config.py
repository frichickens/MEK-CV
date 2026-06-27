"""Per-dataset and per-arch hyperparameter bundles.

The two datasets share the 7-emotion taxonomy but differ enough that the same
recipe is suboptimal on both:
  • FER2013 — 48×48 noisy grayscale, 28.7k train, severe imbalance.
      → small input, RandomCrop 48→44, strong aug, mixup, weighted sampler.
  • RAF-DB  — 100×100 clean aligned RGB faces, 12.3k train, moderate imbalance.
      → resize to 224 to fully use ImageNet-pretrained features,
        gentler aug, no mixup, lower LR, fewer epochs.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DatasetCfg:
    name: str
    train_dir: str
    test_dir: str
    num_classes: int
    img_size: int               # square Resize target
    crop_size: int              # then RandomCrop / CenterCrop to this
    crop_padding: int
    val_split: float
    use_weighted_sampler: bool
    rotation_deg: float
    color_jitter: float
    random_erasing_p: float
    norm_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    norm_std:  List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class TrainCfg:
    arch: str
    epochs: int
    batch_size: int
    lr: float
    momentum: float     = 0.9
    weight_decay: float = 1e-4
    label_smooth: float = 0.1
    mixup_alpha: float  = 0.2
    dropout: float      = 0.4
    early_stop: int     = 25
    num_workers: int    = 4
    seed: int           = 42


# Deeper backbones: smaller LR, slightly more dropout. Multipliers are applied
# to the dataset-default base LR in `make_train_cfg` (fallback for archs not in
# the validation-tuned table below, e.g. resnet50).
_ARCH_TUNING = {
    "resnet18": {"lr_mult": 1.0, "dropout": 0.40},
    "resnet34": {"lr_mult": 0.7, "dropout": 0.45},
    "resnet50": {"lr_mult": 0.5, "dropout": 0.50},
}

# Validation-selected best baseline (SGD-Nesterov + CosineAnnealing) hyperparameters
# from the Kaggle tuning campaign — see results-CV-project.md. Keyed by (dataset, arch).
# Selected on validation accuracy; test reported only (no test-set leakage).
_BEST_BASELINE = {
    ("fer2013", "resnet18"): {"lr": 0.014, "epochs": 100},  # lr14e3-wd1e4
    ("fer2013", "resnet34"): {"lr": 0.007, "epochs": 100},  # default lr (1e-2·0.7)
    ("rafdb",   "resnet18"): {"lr": 0.003, "epochs": 80},   # lr3e3, 80 ep
    ("rafdb",   "resnet34"): {"lr": 0.001, "epochs": 80},   # lr1e3, 80 ep
}


def fer2013(root: str = "/kaggle/input/datasets/msambare/fer2013") -> DatasetCfg:
    return DatasetCfg(
        name="fer2013",
        train_dir=f"{root}/train",
        test_dir=f"{root}/test",
        num_classes=7,
        img_size=48,
        crop_size=44,
        crop_padding=2,
        val_split=0.1,
        use_weighted_sampler=True,    # disgust ≈ 436 vs happy ≈ 7215 → essential
        rotation_deg=10,
        color_jitter=0.2,
        random_erasing_p=0.5,
    )


def rafdb(root: str = "/kaggle/input/raf-db-dataset/DATASET") -> DatasetCfg:
    """RAF-DB defaults.

    Default root matches the popular `shuvoalok/raf-db-dataset` Kaggle layout:
        <root>/train/<class_idx>/*.jpg
        <root>/test/<class_idx>/*.jpg
    Override via CLI `--data-root` for any other layout.
    """
    return DatasetCfg(
        name="rafdb",
        train_dir=f"{root}/train",
        test_dir=f"{root}/test",
        num_classes=7,
        img_size=224,                 # full ImageNet input — exploits pretrained features
        crop_size=224,
        crop_padding=8,
        val_split=0.1,
        use_weighted_sampler=True,
        rotation_deg=15,
        color_jitter=0.3,
        random_erasing_p=0.25,        # cleaner data needs less occlusion
    )


DATASETS = {"fer2013": fer2013, "rafdb": rafdb}


def make_train_cfg(arch: str, dataset_name: str) -> TrainCfg:
    if arch not in _ARCH_TUNING:
        raise ValueError(f"Unknown arch: {arch}. Pick one of {list(_ARCH_TUNING)}.")
    tune = _ARCH_TUNING[arch]

    best = _BEST_BASELINE.get((dataset_name, arch))

    if dataset_name == "fer2013":
        return TrainCfg(
            arch=arch,
            epochs=best["epochs"] if best else 100, batch_size=128,
            lr=best["lr"] if best else 1e-2 * tune["lr_mult"],
            dropout=tune["dropout"],
            mixup_alpha=0.2,
            label_smooth=0.1,
        )
    if dataset_name == "rafdb":
        return TrainCfg(
            arch=arch,
            epochs=best["epochs"] if best else 80, batch_size=64,
            lr=best["lr"] if best else 1e-3 * tune["lr_mult"],
            dropout=tune["dropout"],
            mixup_alpha=0.0,           # mixup gives smaller wins on clean datasets
            label_smooth=0.05,
        )
    raise ValueError(f"Unknown dataset: {dataset_name}")
