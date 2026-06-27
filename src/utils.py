"""Reproducibility + plotting helpers."""
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def plot_history(history: dict, title: str = "Training curves") -> None:
    panels = [
        ("accuracy",  "Accuracy"),
        ("loss",      "Loss"),
        ("f1",        "F1-score"),
        ("precision", "Precision"),
        ("recall",    "Recall"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(20, 4))
    fig.suptitle(title)
    for ax, (k, t) in zip(axes, panels):
        ep = range(1, len(history[k]) + 1)
        ax.plot(ep, history[k],          label="train")
        ax.plot(ep, history[f"val_{k}"], label="val")
        ax.set_title(t)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(t)
        ax.legend()
    plt.tight_layout()
    plt.show()
