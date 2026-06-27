"""MEK-specific config: dataset bundles + per-arch tuning.

Differences from src/config.py:
  • Both datasets are forced to 224×224 input — the AC loss needs a 7×7
    feature map, which only resolves cleanly at the standard ResNet input size.
    For FER2013 this means upsampling 48 → 224 (the network has plenty of
    capacity, and upsampling is a no-op for the loss math).
  • RandomHorizontalFlip is removed from the training transform — MEK
    introduces a *deterministic* flip via the paired-flip dataloader, and
    randomizing it would break the AC-loss assumption.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class MEKDatasetCfg:
    name: str
    train_dir: str
    test_dir: str
    num_classes: int
    img_size: int
    crop_size: int
    crop_padding: int
    val_split: float
    use_weighted_sampler: bool
    rotation_deg: float
    color_jitter: float
    random_erasing_p: float
    norm_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    norm_std:  List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class MEKTrainCfg:
    arch: str
    epochs: int
    batch_size: int
    lr: float
    momentum: float       = 0.9
    weight_decay: float   = 1e-4
    label_smooth: float   = 0.1        # ε in LSR2 — paper's ResNet recipe uses 0.1 (Swin repo used 0.3)
    flip_loss_weight: float = 2.0      # λ in `loss = LSR + λ·ACLoss` — paper's ResNet recipe uses 2 (Swin repo used 0.1)
    sched_gamma: float    = 0.9        # ExponentialLR decay (paper recipe 0.9; webcam runner uses 0.95)
    dropout: float        = 0.4
    early_stop: int       = 25
    num_workers: int      = 4
    seed: int             = 42


# Per-arch knobs (more dropout for deeper backbones). lr_mult is only a fallback
# for archs not in the validation-tuned _BEST_MEK table below (e.g. resnet50).
_ARCH_TUNING = {
    "resnet18": {"lr_mult": 1.0, "dropout": 0.40},
    "resnet34": {"lr_mult": 0.7, "dropout": 0.45},
    "resnet50": {"lr_mult": 0.5, "dropout": 0.50},
}

# Validation-selected best MEK (Adam + ExponentialLR γ=0.9) hyperparameters from the
# Kaggle tuning campaign — see results-CV-project.md. Keyed by (dataset, arch).
# Selected on validation mean-class accuracy; test reported only (no test-set leakage).
#   RAF-DB RN34 uses ε=0.2 (stronger re-balance) to lift fear/disgust above RN18.
#   FER-2013 wants a much lighter λ (0.25 / 0.1) than RAF-DB's paper λ=2.
_BEST_MEK = {
    ("fer2013", "resnet18"): {"lr": 1e-4, "epochs": 80, "flip_loss_weight": 0.25, "label_smooth": 0.1},
    ("fer2013", "resnet34"): {"lr": 1e-4, "epochs": 80, "flip_loss_weight": 0.1,  "label_smooth": 0.1},
    ("rafdb",   "resnet18"): {"lr": 3e-4, "epochs": 60, "flip_loss_weight": 2.0,  "label_smooth": 0.1},
    ("rafdb",   "resnet34"): {"lr": 2e-4, "epochs": 60, "flip_loss_weight": 2.0,  "label_smooth": 0.2},
}


def fer2013(root: str = "/kaggle/input/datasets/msambare/fer2013") -> MEKDatasetCfg:
    return MEKDatasetCfg(
        name="fer2013",
        train_dir=f"{root}/train",
        test_dir=f"{root}/test",
        num_classes=7,
        img_size=224,                   # upsampled — MEK needs 7×7 feature map
        crop_size=224,
        crop_padding=8,
        val_split=0.1,
        use_weighted_sampler=True,
        rotation_deg=10,
        color_jitter=0.2,
        random_erasing_p=0.5,
    )


def rafdb(root: str = "/kaggle/input/raf-db-dataset/DATASET") -> MEKDatasetCfg:
    return MEKDatasetCfg(
        name="rafdb",
        train_dir=f"{root}/train",
        test_dir=f"{root}/test",
        num_classes=7,
        img_size=224,
        crop_size=224,
        crop_padding=8,
        val_split=0.1,
        use_weighted_sampler=True,
        rotation_deg=15,
        color_jitter=0.3,
        random_erasing_p=0.25,
    )


DATASETS = {"fer2013": fer2013, "rafdb": rafdb}


def make_train_cfg(arch: str, dataset_name: str) -> MEKTrainCfg:
    if arch not in _ARCH_TUNING:
        raise ValueError(f"Unknown arch: {arch}. Pick one of {list(_ARCH_TUNING)}.")
    tune = _ARCH_TUNING[arch]

    best = _BEST_MEK.get((dataset_name, arch))

    if dataset_name == "fer2013":
        return MEKTrainCfg(
            arch=arch,
            epochs=best["epochs"] if best else 80, batch_size=64,
            lr=best["lr"] if best else 1e-4,
            dropout=tune["dropout"],
            label_smooth=best["label_smooth"] if best else 0.1,
            flip_loss_weight=best["flip_loss_weight"] if best else 2.0,
        )
    if dataset_name == "rafdb":
        return MEKTrainCfg(
            arch=arch,
            epochs=best["epochs"] if best else 60, batch_size=64,
            lr=best["lr"] if best else 1e-4,
            dropout=tune["dropout"],
            label_smooth=best["label_smooth"] if best else 0.1,
            flip_loss_weight=best["flip_loss_weight"] if best else 2.0,
        )
    raise ValueError(f"Unknown dataset: {dataset_name}")
