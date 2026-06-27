"""Train a webcam-robust MEK model for the live demo.

Same MEK method as train_mek.py (the best method), but tuned for deployment on
demo.py's webcam input rather than for squeezing the last point of test accuracy:

  • heavier "in-the-wild" augmentation (scale / blur / lighting / mild pose) so
    the dataset → webcam domain gap hurts less — see mek/data.py
    `_webcam_train_transform`;
  • EMA weight averaging for a small, reliable stability/accuracy gain;
  • defaults to ResNet-18 + RAF-DB with an MS-Celeb-1M face-recognition backbone
    (pass --face-weights resnet18_msceleb.pth) — the paper's best-reproducible
    setup AND the fastest for real-time/CPU webcam inference. RAF-DB is in-the-wild
    RGB and generalizes to a camera far better than FER2013's grayscale 48×48.

The webcam recipe stays live-tuned (gentler label smoothing, lighter AC weight,
heavy augmentation, EMA) rather than the benchmark recipe — see run().

Checkpoint is saved as  mek_webcam_<arch>_<dataset>_best.pth  so demo.py picks it
up automatically.

Usage:
  python train_mek_webcam.py --face-weights /path/to/resnet18_msceleb.pth
  python train_mek_webcam.py --arch resnet18 --dataset rafdb --data-root /path/to/raf-db \
      --face-weights /path/to/resnet18_msceleb.pth
  python train_mek_webcam.py --epochs 80 --wandb
"""
import argparse

import torch

from mek.config import DATASETS, make_train_cfg
from mek.data import build_mek_loaders
from mek.losses import compute_balance_weights
from mek.model import MEKResNet, AVAILABLE_ARCHS
from mek.trainer import MEKTrainer
from src.utils import set_seed, plot_history
from src.wandb_utils import init_wandb, finish_wandb, log_wandb


def run(
    arch: str = "resnet18",
    dataset: str = "rafdb",
    data_root: str = None,
    epochs: int = None,
    lr: float = None,
    batch_size: int = None,
    ckpt: str = None,
    ema_decay: float = 0.999,
    show_plot: bool = True,
    use_wandb: bool = False,
    face_weights: str = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    ds_cfg = DATASETS[dataset](root=data_root) if data_root else DATASETS[dataset]()
    train_cfg = make_train_cfg(arch, dataset)

    # Validation-selected webcam-deployment recipe (results-CV-project.md, 2026-06-10):
    # the benchmark recipe make_train_cfg returns (e.g. RAF-DB RN18 lr=3e-4, λ=2, ε=0.1,
    # γ=0.9, 60 ep) is tuned for clean-test accuracy. The deployment winner instead uses
    # lr=1e-4, ExpLR γ=0.95, 80 ep, ε=0.15, λ=0.1 + EMA — gentler smoothing and a lighter
    # AC weight let the live model commit to confident predictions, while the slower LR
    # decay over a longer schedule and the heavy webcam augmentation (webcam=True) + EMA
    # do the robustness work. CLI --lr/--epochs/--batch-size still override.
    train_cfg.lr = lr if lr else 1e-4
    train_cfg.epochs = epochs if epochs else 80
    if batch_size:
        train_cfg.batch_size = batch_size
    train_cfg.label_smooth = 0.15
    train_cfg.flip_loss_weight = 0.1
    train_cfg.sched_gamma = 0.95

    set_seed(train_cfg.seed)
    print(f"Webcam-robust MEK | Dataset: {ds_cfg.name} (img={ds_cfg.img_size} crop={ds_cfg.crop_size})")
    print(f"Arch: {train_cfg.arch} (lr={train_cfg.lr:.2e} bs={train_cfg.batch_size} "
          f"epochs={train_cfg.epochs} eps_lsr={train_cfg.label_smooth} lam_flip={train_cfg.flip_loss_weight}) "
          f"| EMA decay={ema_decay}")

    train_loader, val_loader, test_loader, classes, counts = build_mek_loaders(
        ds_cfg,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        seed=train_cfg.seed,
        webcam=True,                       # heavier, webcam-oriented augmentation
    )
    print(f"Classes: {classes}")
    print(f"Counts: {dict(zip(classes, counts.tolist()))}")
    print(f"Sizes: Train={len(train_loader.dataset)} | "
          f"Val={len(val_loader.dataset)} | Test={len(test_loader.dataset)}")

    balance_w = compute_balance_weights(counts)
    print(f"Balance weights (mean=1): {balance_w.numpy().round(3).tolist()}")

    model = MEKResNet(arch, num_classes=ds_cfg.num_classes, dropout=train_cfg.dropout,
                      face_weights=face_weights)

    with torch.no_grad():
        out_logits, out_hm = model.to(device)(torch.zeros(2, 3, ds_cfg.crop_size, ds_cfg.crop_size, device=device))
        assert out_logits.shape == (2, ds_cfg.num_classes), f"Bad logits shape {out_logits.shape}"
    print(f"Forward OK: logits={tuple(out_logits.shape)} hm={tuple(out_hm.shape)}")

    ckpt_path = ckpt or f"mek_webcam_{train_cfg.arch}_{ds_cfg.name}_best.pth"
    trainer = MEKTrainer(
        model, train_cfg, ds_cfg, balance_weights=balance_w, device=device,
        use_ema=True, ema_decay=ema_decay,
    )

    wandb_enabled = False
    if use_wandb:
        wandb_enabled = init_wandb(
            project="fer-emotion-recognition",
            entity=None,
            run_name=f"mek_webcam_{arch}_{dataset}",
            config={
                "arch": arch, "dataset": dataset, "method": "MEK-webcam",
                "epochs": train_cfg.epochs, "lr": train_cfg.lr,
                "batch_size": train_cfg.batch_size, "dropout": train_cfg.dropout,
                "label_smooth": train_cfg.label_smooth,
                "flip_loss_weight": train_cfg.flip_loss_weight,
                "ema_decay": ema_decay, "augmentation": "webcam",
                "backbone": model.backbone_source,
            },
        )

    def _epoch_log(epoch, tr, va, lr_now):
        log_wandb({
            "epoch": epoch,
            "train/loss": tr["loss"], "train/accuracy": tr["accuracy"],
            "train/f1": tr["f1"], "train/lsr_loss": tr["lsr_loss"], "train/flip_loss": tr["flip_loss"],
            "val/loss": va["loss"], "val/accuracy": va["accuracy"], "val/f1": va["f1"],
            "lr": lr_now,
        }, step=epoch)

    history, best_val = trainer.fit(
        train_loader, val_loader, ckpt_path=ckpt_path,
        log_fn=_epoch_log if wandb_enabled else None,
    )

    test_res = trainer.evaluate(test_loader, return_per_class=True)
    print(f"\nTest results (EMA weights):")
    print(f" acc: {test_res['accuracy']:.4f}")
    print(f" f1 (macro): {test_res['f1']:.4f}")
    print(f" mean class acc: {test_res['mean_class_acc']:.4f}")
    print(f" per-class acc:")
    for cls, acc in zip(classes, test_res["per_class_acc"]):
        print(f" {cls:12s} {acc:.4f}")
    print(f"\nSaved webcam checkpoint -> {ckpt_path}")

    if wandb_enabled:
        log_wandb({
            "test/accuracy": test_res["accuracy"],
            "test/f1": test_res["f1"],
            "test/mean_class_acc": test_res["mean_class_acc"],
        })
        finish_wandb()

    if show_plot:
        plot_history(history, title=f"MEK-webcam / {train_cfg.arch} / {ds_cfg.name}")

    return history, best_val, test_res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=AVAILABLE_ARCHS, default="resnet18")
    p.add_argument("--dataset", choices=list(DATASETS), default="rafdb")
    p.add_argument("--data-root", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--wandb", action="store_true", help="Log run to Weights & Biases")
    p.add_argument("--face-weights", default=None,
                   help="Face-recognition pretrained backbone (e.g. resnet18_msceleb.pth) — "
                        "the best webcam init. .pth loads safely; falls back to ImageNet if missing.")
    args = p.parse_args()

    run(
        arch=args.arch,
        dataset=args.dataset,
        data_root=args.data_root,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        ckpt=args.ckpt,
        ema_decay=args.ema_decay,
        show_plot=not args.no_plot,
        use_wandb=args.wandb,
        face_weights=args.face_weights,
    )


if __name__ == "__main__":
    main()
