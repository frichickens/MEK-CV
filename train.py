"""Train any of {resnet18, resnet34, resnet50} on {fer2013, rafdb}.

Usage:
  python train.py --arch resnet18 --dataset fer2013
  python train.py --arch resnet50 --dataset rafdb --data-root /path/to/raf-db
  python train.py --arch resnet18 --dataset fer2013 --epochs 60 --no-tta
  python train.py --arch resnet18 --dataset fer2013 --wandb

The recipes are dataset-specific (see src/config.py). Pass --epochs / --lr /
batch-size to override individual fields without editing config.py.
"""
import argparse
import os

import torch

from src.config import DATASETS, make_train_cfg
from src.data import build_loaders
from src.models import build_model, AVAILABLE_ARCHS
from src.trainer import Trainer
from src.tta import evaluate_tencrop
from src.utils import set_seed, plot_history
from src.wandb_utils import init_wandb, finish_wandb, log_wandb


def run(
    arch: str,
    dataset: str,
    data_root: str = None,
    epochs: int = None,
    lr: float = None,
    batch_size: int = None,
    ckpt: str = None,
    do_tta: bool = True,
    show_plot: bool = True,
    use_wandb: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    ds_cfg = DATASETS[dataset](root=data_root) if data_root else DATASETS[dataset]()
    train_cfg = make_train_cfg(arch, dataset)
    if epochs:
        train_cfg.epochs = epochs
    if lr:
        train_cfg.lr = lr
    if batch_size:
        train_cfg.batch_size = batch_size

    set_seed(train_cfg.seed)
    print(f"Dataset: {ds_cfg.name} (img={ds_cfg.img_size} crop={ds_cfg.crop_size}) -> {ds_cfg.train_dir}")
    print(f"Arch: {train_cfg.arch} (lr={train_cfg.lr:.2e} bs={train_cfg.batch_size} "
          f"epochs={train_cfg.epochs} mixup={train_cfg.mixup_alpha} dropout={train_cfg.dropout})")

    train_loader, val_loader, test_loader, classes = build_loaders(
        ds_cfg,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        seed=train_cfg.seed,
    )
    print(f"Classes: {classes}")
    print(f"Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}")

    model = build_model(arch, num_classes=ds_cfg.num_classes, dropout=train_cfg.dropout)

    with torch.no_grad():
        out = model.to(device)(torch.zeros(2, 3, ds_cfg.crop_size, ds_cfg.crop_size, device=device))
        assert out.shape == (2, ds_cfg.num_classes), f"Bad output shape {out.shape}"
    print("Model output shape OK.")

    ckpt_path = ckpt or f"{train_cfg.arch}_{ds_cfg.name}_best.pth"
    trainer = Trainer(model, train_cfg, ds_cfg, device)

    wandb_enabled = False
    if use_wandb:
        wandb_enabled = init_wandb(
            project="fer-emotion-recognition",
            entity=None,
            run_name=f"baseline_{arch}_{dataset}",
            config={
                "arch": arch,
                "dataset": dataset,
                "epochs": train_cfg.epochs,
                "lr": train_cfg.lr,
                "batch_size": train_cfg.batch_size,
                "mixup_alpha": train_cfg.mixup_alpha,
                "dropout": train_cfg.dropout,
                "label_smooth": train_cfg.label_smooth,
                "optimizer": "SGD",
                "momentum": train_cfg.momentum,
                "weight_decay": train_cfg.weight_decay,
            },
        )

    def _epoch_log(epoch, tr, va, lr):
        log_wandb({
            "epoch": epoch,
            "train/loss": tr["loss"], "train/accuracy": tr["accuracy"],
            "train/f1": tr["f1"], "train/precision": tr["precision"], "train/recall": tr["recall"],
            "val/loss": va["loss"], "val/accuracy": va["accuracy"],
            "val/f1": va["f1"], "val/precision": va["precision"], "val/recall": va["recall"],
            "lr": lr,
        }, step=epoch)

    history, best_val = trainer.fit(
        train_loader, val_loader, ckpt_path=ckpt_path,
        log_fn=_epoch_log if wandb_enabled else None,
    )

    test_res = trainer.evaluate(test_loader, return_per_class=True)
    print(f"\nTest results:")
    print(f" acc: {test_res['accuracy']:.4f}")
    print(f" f1 (macro): {test_res['f1']:.4f}")
    print(f" precision: {test_res['precision']:.4f}")
    print(f" recall: {test_res['recall']:.4f}")
    print(f" mean class acc: {test_res['mean_class_acc']:.4f} "
          f"<- imbalance-robust headline metric (matches MEK)")
    print(f" per-class acc:")
    for cls, acc in zip(classes, test_res["per_class_acc"]):
        print(f" {cls:12s} {acc:.4f}")

    if wandb_enabled:
        log_wandb({"test/accuracy": test_res["accuracy"],
                   "test/f1": test_res["f1"],
                   "test/mean_class_acc": test_res["mean_class_acc"]})
        finish_wandb()

    if do_tta:
        tta_acc = evaluate_tencrop(model, ds_cfg.test_dir, ds_cfg, device)
        print(f" 10-crop TTA: {tta_acc:.4f}")

    if show_plot:
        plot_history(history, title=f"{train_cfg.arch} / {ds_cfg.name}")

    return history, best_val, test_res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=AVAILABLE_ARCHS, required=True)
    p.add_argument("--dataset", choices=list(DATASETS), required=True)
    p.add_argument("--data-root", default=None,
                   help="Root containing train/ and test/. Defaults to the dataset's standard Kaggle path.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--wandb", action="store_true", help="Log run to Weights & Biases")
    args = p.parse_args()

    run(
        arch=args.arch,
        dataset=args.dataset,
        data_root=args.data_root,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        ckpt=args.ckpt,
        do_tta=not args.no_tta,
        show_plot=not args.no_plot,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
