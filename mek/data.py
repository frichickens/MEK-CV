"""Dataloader for MEK training: yields (img, label, img_hflipped) per sample.

Reuses the same ImageFolder layout as src/data.py:
    <root>/train/<class>/*.jpg
    <root>/test/<class>/*.jpg

Two extra requirements vs. plain training:
  • The flipped image is needed by the ACLoss (flip-consistency).
  • We compute and expose per-class training counts so the trainer can build
    the balance-weight tensor.
"""
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, Dataset
from torchvision import datasets, transforms

from .config import MEKDatasetCfg


class _PairedFlipDataset(Dataset):
    """Wraps any (img, label) dataset and additionally returns the H-flipped img."""
    def __init__(self, base: Dataset):
        self.base = base
        self._flip = transforms.RandomHorizontalFlip(p=1.0)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return img, label, self._flip(img)


def _train_transform(cfg: MEKDatasetCfg) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomCrop(cfg.crop_size, padding=cfg.crop_padding),
        transforms.RandomRotation(cfg.rotation_deg),
        transforms.ColorJitter(brightness=cfg.color_jitter, contrast=cfg.color_jitter),
        transforms.ToTensor(),
        transforms.Normalize(cfg.norm_mean, cfg.norm_std),
        transforms.RandomErasing(p=cfg.random_erasing_p, scale=(0.02, 0.2)),
    ])


def _webcam_train_transform(cfg: MEKDatasetCfg) -> transforms.Compose:
    """Heavier, webcam-oriented augmentation.

    Simulates the lighting / scale / focus / mild-pose variation a live camera
    introduces, so a model trained with it degrades far less when deployed on
    demo.py's webcam input (the dataset→webcam domain gap, not test accuracy, is
    what usually breaks live demos).

    Still NO RandomHorizontalFlip: the deterministic paired flip is what the AC
    loss consumes. Every op here acts on `img`, and `_PairedFlipDataset` flips the
    final tensor, so `img_flip == hflip(img)` still holds exactly.
    """
    return transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomResizedCrop(cfg.crop_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),     # slight off-angle faces
        transforms.RandomRotation(cfg.rotation_deg),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.3),  # motion/focus blur
        transforms.RandomGrayscale(p=0.1),                             # robustness to lighting/colour
        transforms.ToTensor(),
        transforms.Normalize(cfg.norm_mean, cfg.norm_std),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),            # occlusion (hand, hair)
    ])


def _eval_transform(cfg: MEKDatasetCfg) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.CenterCrop(cfg.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(cfg.norm_mean, cfg.norm_std),
    ])


def build_mek_loaders(
    cfg: MEKDatasetCfg,
    batch_size: int,
    num_workers: int,
    seed: int,
    webcam: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], np.ndarray]:
    """Returns (train_loader, val_loader, test_loader, classes, train_class_counts).

    The train loader yields triples (img, label, img_hflipped). Val and test
    loaders yield standard (img, label) pairs (no flip needed at eval time).

    NOTE: we deliberately do *not* RandomHorizontalFlip in the train transform.
    The deterministic flip on top of `img` (done by `_PairedFlipDataset`) IS the
    augmentation MEK uses, and adding random flips would break the AC loss
    (which assumes `img_flipped` is exactly H-flip of `img`).
    """
    train_tf = _webcam_train_transform(cfg) if webcam else _train_transform(cfg)
    eval_tf  = _eval_transform(cfg)

    full_aug  = datasets.ImageFolder(cfg.train_dir, transform=train_tf)
    full_eval = datasets.ImageFolder(cfg.train_dir, transform=eval_tf)
    test_ds   = datasets.ImageFolder(cfg.test_dir,  transform=eval_tf)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(full_aug))
    val_size = int(len(full_aug) * cfg.val_split)
    val_idx, train_idx = idx[:val_size], idx[val_size:]

    train_ds = _PairedFlipDataset(Subset(full_aug,  train_idx))
    val_ds   = Subset(full_eval, val_idx)

    if cfg.use_weighted_sampler:
        labels = np.array(full_aug.targets)[train_idx]
        counts = np.bincount(labels, minlength=cfg.num_classes).astype(np.float64)
        weights = 1.0 / np.maximum(counts, 1.0)
        sample_w = torch.from_numpy(weights[labels]).double()
        sampler  = WeightedRandomSampler(sample_w, num_samples=len(labels), replacement=True)
        shuffle = False
    else:
        sampler, shuffle = None, True

    # train_class_counts is computed over the *training subset* (not the
    # full folder) so the balance weights match what the model actually sees.
    train_labels = np.array(full_aug.targets)[train_idx]
    train_class_counts = np.bincount(train_labels, minlength=cfg.num_classes)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, full_aug.classes, train_class_counts
