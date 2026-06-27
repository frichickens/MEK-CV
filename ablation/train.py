"""Ablation study runner — reproduces the MEK paper's two-module ablation.

Paper: "Ablation study of our proposed two modules re-balanced attention consistency
(RAC) and re-balanced smooth labels (RSL). Both of the two modules can improve the
performance based on the baseline, while they can cooperate to achieve the best
performance." (Zhang et al., NeurIPS 2023, arxiv 2310.19636)

This is the exact 2×2: each module independently toggled on top of the baseline.

  baseline   -- plain CrossEntropy, no attention consistency        (neither module)
  rsl        -- + Re-balanced Smooth Labels                         (RSL only)
  rac        -- + Re-balanced Attention Consistency (flip AC loss)  (RAC only)
  rsl_rac    -- RSL + RAC                                            (full MEK, best)

Defaults to ResNet-18 (the paper's ablation backbone). For a faithful reproduction
pass the MS-Celeb-1M face backbone — the paper's init — via --face-weights; without
it the study still runs (on ImageNet) and the relative module gains still hold.

Usage:
  python ablation/train.py --face-weights /path/to/resnet18_msceleb.pth
  python ablation/train.py --dataset rafdb --face-weights /path/to/resnet18_msceleb.pth
  python ablation/train.py --dataset rafdb --epochs 60 --wandb

Overrides: --arch, --dataset, --epochs, --batch-size, --data-root, --no-plot.
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn

from mek.config import DATASETS, make_train_cfg
from mek.data import build_mek_loaders
from mek.losses import ReBalancedLabelSmoothing, compute_balance_weights
from mek.model import MEKResNet, AVAILABLE_ARCHS
from mek.trainer import MEKTrainer
from src.utils import set_seed
from src.wandb_utils import init_wandb, finish_wandb, log_wandb


# -- The paper's 2×2 ablation: each module (RSL, RAC) toggled on the baseline -------
VARIANTS = {
    "baseline": {"rsl": False, "rac": False},   # plain CE, no AC
    "rsl":      {"rsl": True,  "rac": False},   # + re-balanced smooth labels
    "rac":      {"rsl": False, "rac": True},    # + re-balanced attention consistency
    "rsl_rac":  {"rsl": True,  "rac": True},    # full MEK (both modules)
}


def build_criterion(use_rsl, balance_weights, label_smooth):
    """RSL on → Re-balanced Smooth Labels; RSL off → plain CrossEntropy (the baseline)."""
    if use_rsl:
        return ReBalancedLabelSmoothing(epsilon=label_smooth, balance_weights=balance_weights)
    return nn.CrossEntropyLoss()


# -- Single variant runner ---------------------------------------------
def run_variant(variant, use_rsl, use_rac, arch, dataset, data_root,
                epochs, batch_size, device, face_weights=None, wandb_enabled=False):
    ds_cfg = DATASETS[dataset](root=data_root) if data_root else DATASETS[dataset]()
    train_cfg = make_train_cfg(arch, dataset)
    if epochs:
        train_cfg.epochs = epochs
    if batch_size:
        train_cfg.batch_size = batch_size
    # RAC = the flip attention-consistency loss; disable it by zeroing its weight.
    train_cfg.flip_loss_weight = train_cfg.flip_loss_weight if use_rac else 0.0

    set_seed(train_cfg.seed)

    train_loader, val_loader, test_loader, classes, counts = build_mek_loaders(
        ds_cfg,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        seed=train_cfg.seed,
    )
    balance_w = compute_balance_weights(counts).to(device)

    model = MEKResNet(
        arch, num_classes=ds_cfg.num_classes, dropout=train_cfg.dropout,
        face_weights=face_weights,
    ).to(device)

    criterion = build_criterion(use_rsl, balance_w, train_cfg.label_smooth).to(device)

    trainer = MEKTrainer(
        model, train_cfg, ds_cfg,
        balance_weights=balance_w,
        device=device,
        criterion=criterion,
    )

    os.makedirs("ablation/ckpt", exist_ok=True)
    ckpt_path = f"ablation/ckpt/{variant}_{arch}_{dataset}_best.pth"

    # Single W&B run spans all variants, so we namespace metrics per variant and
    # let the step auto-increment (explicit per-epoch steps would collide across
    # variants, which W&B rejects as non-monotonic). Plot vs the logged epoch.
    def _epoch_log(epoch, tr, va, lr):
        log_wandb({
            f"{variant}/epoch": epoch,
            f"{variant}/train/loss": tr["loss"], f"{variant}/train/accuracy": tr["accuracy"],
            f"{variant}/train/f1": tr["f1"],
            f"{variant}/train/lsr_loss": tr["lsr_loss"], f"{variant}/train/flip_loss": tr["flip_loss"],
            f"{variant}/val/loss": va["loss"], f"{variant}/val/accuracy": va["accuracy"],
            f"{variant}/val/f1": va["f1"],
            f"{variant}/lr": lr,
        })

    t0 = time.time()
    history, best_val = trainer.fit(
        train_loader, val_loader, ckpt_path=ckpt_path,
        log_fn=_epoch_log if wandb_enabled else None,
    )
    elapsed = time.time() - t0

    test_res = trainer.evaluate(test_loader, return_per_class=True)

    return {
        "variant": variant,
        "rsl": use_rsl,
        "rac": use_rac,
        "best_val_acc": round(best_val, 6),
        "test_acc": round(test_res["accuracy"], 6),
        "test_f1": round(test_res["f1"], 6),
        "test_mean_class_acc": round(test_res["mean_class_acc"], 6),
        "per_class_acc": {
            cls: round(a, 4) for cls, a in zip(classes, test_res["per_class_acc"])
        },
        "train_time_s": round(elapsed, 1),
        "history": history,
    }


# -- Main ---------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=AVAILABLE_ARCHS, default="resnet18",
                   help="Backbone (paper's ablation uses resnet18).")
    p.add_argument("--dataset", choices=list(DATASETS), default="rafdb")
    p.add_argument("--data-root", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--face-weights", default=None,
                   help="MS-Celeb-1M face backbone (resnet18_msceleb.pth) — the paper's init. "
                        ".pth loads safely; falls back to ImageNet if missing.")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--wandb", action="store_true", help="Log run to Weights & Biases")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Arch: {args.arch}  |  Dataset: {args.dataset}")
    print(f"Backbone init: {'MS-Celeb-1M (' + args.face_weights + ')' if args.face_weights else 'ImageNet'}")
    print(f"Variants (RSL × RAC): {list(VARIANTS.keys())}\n")

    wandb_enabled = False
    if args.wandb:
        wandb_enabled = init_wandb(
            project="fer-emotion-recognition",
            entity=None,
            run_name=f"ablation_{args.arch}_{args.dataset}",
            config={
                "arch": args.arch,
                "dataset": args.dataset,
                "method": "ablation-RSL-RAC",
                "variants": list(VARIANTS.keys()),
                "face_weights": bool(args.face_weights),
                "epochs": args.epochs or make_train_cfg(args.arch, args.dataset).epochs,
                "batch_size": args.batch_size or make_train_cfg(args.arch, args.dataset).batch_size,
            },
        )

    results = []
    for name, cfg in VARIANTS.items():
        print(f"{'-'*60}")
        print(f"  [{name}]  RSL={cfg['rsl']}  RAC={cfg['rac']}")
        print(f"{'-'*60}")
        r = run_variant(
            variant=name, use_rsl=cfg["rsl"], use_rac=cfg["rac"],
            arch=args.arch, dataset=args.dataset,
            data_root=args.data_root,
            epochs=args.epochs, batch_size=args.batch_size,
            device=device, face_weights=args.face_weights,
            wandb_enabled=wandb_enabled,
        )
        results.append(r)
        print(f"  -> val_acc={r['best_val_acc']:.4f}  "
              f"test_acc={r['test_acc']:.4f}  "
              f"mean_cls_acc={r['test_mean_class_acc']:.4f}  "
              f"({r['train_time_s']}s)\n")

        if wandb_enabled:
            log_wandb({
                f"{name}/test_accuracy": r["test_acc"],
                f"{name}/test_f1": r["test_f1"],
                f"{name}/test_mean_class_acc": r["test_mean_class_acc"],
            })

    # -- Summary table (paper format: RSL / RAC toggles) ----------------
    _tick = lambda b: "✓" if b else "✗"
    print(f"\n{'='*72}")
    print(f"  Ablation results -- {args.arch} / {args.dataset}  (RSL × RAC)")
    print(f"{'='*72}")
    header = (f"{'Variant':<10} {'RSL':>4} {'RAC':>4} "
              f"{'Val Acc':>9} {'Test Acc':>9} {'Mean Cls':>10} {'Time':>8}")
    print(header)
    print(f"{'-'*72}")
    for r in results:
        print(f"{r['variant']:<10} {_tick(r['rsl']):>4} {_tick(r['rac']):>4} "
              f"{r['best_val_acc']:>9.4f} {r['test_acc']:>9.4f} "
              f"{r['test_mean_class_acc']:>10.4f} {r['train_time_s']:>7.1f}s")
    print(f"{'-'*72}")

    # Headline deltas vs baseline (the paper's narrative: each module helps; both best).
    by_name = {r["variant"]: r for r in results}
    if "baseline" in by_name:
        base = by_name["baseline"]["test_mean_class_acc"]
        print(f"\nMean-class-acc gain over baseline (= paper's headline metric):")
        for name in ("rsl", "rac", "rsl_rac"):
            if name in by_name:
                d = by_name[name]["test_mean_class_acc"] - base
                print(f"  {name:<10} {d:+.4f}")

    # Per-class accuracy table
    print(f"\nPer-class accuracy:")
    classes = list(results[0]["per_class_acc"].keys())
    print(f"  {'Variant':<10}", end="")
    for c in classes:
        print(f"  {c:>12}", end="")
    print()
    for r in results:
        print(f"  {r['variant']:<10}", end="")
        for c in classes:
            print(f"  {r['per_class_acc'][c]:>12.4f}", end="")
        print()

    # Save JSON
    os.makedirs("ablation/results", exist_ok=True)
    out_path = f"ablation/results/{args.arch}_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Combined training curve plot
    if not args.no_plot:
        histories = {r["variant"]: r["history"] for r in results}
        _plot_ablation(histories, args.arch, args.dataset)

    if wandb_enabled:
        finish_wandb()


def _plot_ablation(histories, arch, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("accuracy", "Accuracy"),
        ("loss", "Loss"),
        ("f1", "F1-score"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 4))
    fig.suptitle(f"Ablation (RSL × RAC) -- {arch} / {dataset}")
    colors = {"baseline": "#9467bd", "rsl": "#ff7f0e",
              "rac": "#1f77b4", "rsl_rac": "#2ca02c"}

    for ax, (k, title) in zip(axes, panels):
        for variant, history in histories.items():
            h = history.get(k, [])
            vh = history.get(f"val_{k}", [])
            ep = range(1, len(h) + 1)
            ax.plot(ep, h, color=colors.get(variant, "gray"),
                    lw=1.2, label=f"{variant} (train)")
            ax.plot(ep, vh, color=colors.get(variant, "gray"),
                    lw=1.8, ls="--", label=f"{variant} (val)")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend(fontsize=7)
    plt.tight_layout()
    out = f"ablation/results/{arch}_{dataset}_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Curves saved to {out}")


if __name__ == "__main__":
    main()
