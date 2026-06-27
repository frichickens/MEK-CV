"""Train MEK + (resnet18|resnet34|resnet50) on (fer2013|rafdb).

Usage:
  python train_mek.py --arch resnet18 --dataset rafdb
  python train_mek.py --arch resnet50 --dataset fer2013 --epochs 60
  python train_mek.py --arch resnet18 --dataset rafdb --resume ckpt.pth --eval-only
  python train_mek.py --arch resnet18 --dataset rafdb --wandb
"""
import argparse
import os

import torch

from mek.config import DATASETS, make_train_cfg
from mek.data import build_mek_loaders
from mek.losses import compute_balance_weights
from mek.model import MEKResNet, AVAILABLE_ARCHS
from mek.trainer import MEKTrainer
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
    resume: str = None,
    eval_only: bool = False,
    show_plot: bool = True,
    use_wandb: bool = False,
    face_weights: str = None,
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
          f"epochs={train_cfg.epochs} eps_lsr={train_cfg.label_smooth} lam_flip={train_cfg.flip_loss_weight})")

    train_loader, val_loader, test_loader, classes, counts = build_mek_loaders(
        ds_cfg,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        seed=train_cfg.seed,
    )
    print(f"Classes: {classes}")
    print(f"Counts: {dict(zip(classes, counts.tolist()))}")
    print(f"Sizes: Train={len(train_loader.dataset)} | "
          f"Val={len(val_loader.dataset)} | Test={len(test_loader.dataset)}")

    balance_w = compute_balance_weights(counts)
    print(f"Balance weights (mean=1): {balance_w.numpy().round(3).tolist()}")

    model = MEKResNet(
        arch, num_classes=ds_cfg.num_classes, dropout=train_cfg.dropout,
        face_weights=face_weights,
    )

    with torch.no_grad():
        out_logits, out_hm = model.to(device)(torch.zeros(2, 3, ds_cfg.crop_size, ds_cfg.crop_size, device=device))
        assert out_logits.shape == (2, ds_cfg.num_classes), f"Bad logits shape {out_logits.shape}"
        assert out_hm.shape[0] == 2 and out_hm.shape[1] == ds_cfg.num_classes, f"Bad hm shape {out_hm.shape}"
    print(f"Forward OK: logits={tuple(out_logits.shape)} hm={tuple(out_hm.shape)}")

    if resume:
        state = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded checkpoint from {resume}")

    ckpt_path = ckpt or f"mek_{train_cfg.arch}_{ds_cfg.name}_best.pth"
    trainer = MEKTrainer(model, train_cfg, ds_cfg, balance_weights=balance_w, device=device)

    wandb_enabled = False
    if use_wandb:
        wandb_enabled = init_wandb(
            project="fer-emotion-recognition",
            entity=None,
            run_name=f"mek_{arch}_{dataset}",
            config={
                "arch": arch,
                "dataset": dataset,
                "method": "MEK",
                "epochs": train_cfg.epochs,
                "lr": train_cfg.lr,
                "batch_size": train_cfg.batch_size,
                "dropout": train_cfg.dropout,
                "label_smooth": train_cfg.label_smooth,
                "flip_loss_weight": train_cfg.flip_loss_weight,
                "optimizer": "Adam",
                "weight_decay": train_cfg.weight_decay,
                "backbone": model.backbone_source,
            },
        )

    def _epoch_log(epoch, tr, va, lr):
        log_wandb({
            "epoch": epoch,
            "train/loss": tr["loss"], "train/accuracy": tr["accuracy"],
            "train/f1": tr["f1"], "train/precision": tr["precision"], "train/recall": tr["recall"],
            "train/lsr_loss": tr["lsr_loss"], "train/flip_loss": tr["flip_loss"],
            "val/loss": va["loss"], "val/accuracy": va["accuracy"],
            "val/f1": va["f1"], "val/precision": va["precision"], "val/recall": va["recall"],
            "lr": lr,
        }, step=epoch)

    if not eval_only:
        history, best_val = trainer.fit(
            train_loader, val_loader, ckpt_path=ckpt_path,
            log_fn=_epoch_log if wandb_enabled else None,
        )
    else:
        history, best_val = None, None
        print("Skipping training -- eval-only mode.")

    test_res = trainer.evaluate(test_loader, return_per_class=True)
    print(f"\nTest results:")
    print(f" acc: {test_res['accuracy']:.4f}")
    print(f" f1 (macro): {test_res['f1']:.4f}")
    print(f" precision: {test_res['precision']:.4f}")
    print(f" recall: {test_res['recall']:.4f}")
    print(f" mean class acc: {test_res['mean_class_acc']:.4f} "
          f"<- MEK's main metric (robust to imbalance)")
    print(f" per-class acc:")
    for cls, acc in zip(classes, test_res["per_class_acc"]):
        print(f" {cls:12s} {acc:.4f}")

    if wandb_enabled:
        log_wandb({
            "test/accuracy": test_res["accuracy"],
            "test/f1": test_res["f1"],
            "test/mean_class_acc": test_res["mean_class_acc"],
        })
        finish_wandb()

    if history is not None and show_plot:
        plot_history(history, title=f"MEK / {train_cfg.arch} / {ds_cfg.name}")

    return history, best_val, test_res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=AVAILABLE_ARCHS, required=True)
    p.add_argument("--dataset", choices=list(DATASETS), required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--resume", default=None, help="Path to checkpoint to load before training/eval.")
    p.add_argument("--eval-only", action="store_true", help="Skip training, evaluate the resumed checkpoint on test.")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--wandb", action="store_true", help="Log run to Weights & Biases")
    p.add_argument("--face-weights", default=None,
                   help="Path to face-recognition pretrained weights to init the backbone — "
                        "the paper uses an MS-Celeb-1M face init. Prefer a .pth (loaded safely "
                        "with weights_only=True); a .pkl is unpickled only if its digest is in "
                        "the source-pinned allow-list in mek/model.py. Falls back to ImageNet "
                        "if missing/unmatched.")
    args = p.parse_args()

    run(
        arch=args.arch,
        dataset=args.dataset,
        data_root=args.data_root,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        ckpt=args.ckpt,
        resume=args.resume,
        eval_only=args.eval_only,
        show_plot=not args.no_plot,
        use_wandb=args.wandb,
        face_weights=args.face_weights,
    )


if __name__ == "__main__":
    main()
