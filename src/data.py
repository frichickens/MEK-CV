"""Dataloader factory for FER-style datasets (train/test ImageFolder layout)."""
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

from .config import DatasetCfg


def _train_transform(cfg: DatasetCfg) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomCrop(cfg.crop_size, padding=cfg.crop_padding),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(cfg.rotation_deg),
        transforms.ColorJitter(brightness=cfg.color_jitter, contrast=cfg.color_jitter),
        transforms.ToTensor(),
        transforms.Normalize(cfg.norm_mean, cfg.norm_std),
        transforms.RandomErasing(p=cfg.random_erasing_p, scale=(0.02, 0.2)),
    ])


def _eval_transform(cfg: DatasetCfg) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.CenterCrop(cfg.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(cfg.norm_mean, cfg.norm_std),
    ])


def build_loaders(
    cfg: DatasetCfg,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Build (train, val, test) loaders + class names.

    Two ImageFolders point at the same train dir so the val Subset can use the
    clean transform and the train Subset uses augmentations.
    """
    train_tf = _train_transform(cfg)
    eval_tf  = _eval_transform(cfg)

    full_aug  = datasets.ImageFolder(cfg.train_dir, transform=train_tf)
    full_eval = datasets.ImageFolder(cfg.train_dir, transform=eval_tf)
    test_ds   = datasets.ImageFolder(cfg.test_dir,  transform=eval_tf)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(full_aug))
    val_size = int(len(full_aug) * cfg.val_split)
    val_idx, train_idx = idx[:val_size], idx[val_size:]

    train_ds = Subset(full_aug,  train_idx)
    val_ds   = Subset(full_eval, val_idx)

    if cfg.use_weighted_sampler:
        labels = np.array(full_aug.targets)[train_idx]
        counts = np.bincount(labels, minlength=cfg.num_classes).astype(np.float64)
        weights = 1.0 / np.maximum(counts, 1.0)
        sample_w = torch.from_numpy(weights[labels]).double()
        sampler = WeightedRandomSampler(sample_w, num_samples=len(labels), replacement=True)
        shuffle = False
    else:
        sampler, shuffle = None, True

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, full_aug.classes
