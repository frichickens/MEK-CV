"""Training loop: SGD-Nesterov + cosine LR + AMP + optional mixup + early stop."""
import copy
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score

from .config import TrainCfg, DatasetCfg


def _make_metrics(num_classes: int, device: torch.device) -> dict:
    return {
        "accuracy":  Accuracy( task="multiclass", num_classes=num_classes).to(device),
        "precision": Precision(task="multiclass", num_classes=num_classes, average="macro").to(device),
        "recall":    Recall(   task="multiclass", num_classes=num_classes, average="macro").to(device),
        "f1":        F1Score(  task="multiclass", num_classes=num_classes, average="macro").to(device),
    }


def _per_class_accuracy(confmat: np.ndarray) -> np.ndarray:
    """Diagonal / row-sum, with safe division. Same definition as mek/trainer.py."""
    rs = confmat.sum(axis=1)
    return np.where(rs > 0, np.diag(confmat) / np.maximum(rs, 1), 0.0)


def _mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[idx], y, y[idx], lam


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_cfg: TrainCfg,
        dataset_cfg: DatasetCfg,
        device: torch.device,
    ):
        self.model       = model.to(device)
        self.train_cfg   = train_cfg
        self.dataset_cfg = dataset_cfg
        self.device      = device

        self.criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smooth)
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=train_cfg.lr,
            momentum=train_cfg.momentum,
            weight_decay=train_cfg.weight_decay,
            nesterov=True,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=train_cfg.epochs
        )
        self.amp_on = device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_on)

    # ------------------------------------------------------------------ #
    def _train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        nc = self.dataset_cfg.num_classes
        metrics = _make_metrics(nc, self.device)
        total_loss, total_n = 0.0, 0
        alpha = self.train_cfg.mixup_alpha

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.amp_on):
                if alpha > 0:
                    x, y_a, y_b, lam = _mixup_batch(images, labels, alpha)
                    logits = self.model(x)
                    loss = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)
                else:
                    logits = self.model(images)
                    loss   = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * images.size(0)
            total_n    += images.size(0)
            for m in metrics.values():
                m.update(logits.softmax(1), labels)

        out = {k: v.compute().item() for k, v in metrics.items()}
        out["loss"] = total_loss / total_n
        return out

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, return_per_class: bool = False) -> dict:
        self.model.eval()
        nc = self.dataset_cfg.num_classes
        metrics = _make_metrics(nc, self.device)
        total_loss, total_n = 0.0, 0
        confmat = np.zeros((nc, nc), dtype=np.int64) if return_per_class else None

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits = self.model(images)
            loss   = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            total_n    += images.size(0)
            for m in metrics.values():
                m.update(logits.softmax(1), labels)
            if confmat is not None:
                preds = logits.argmax(1)
                for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                    confmat[t, p] += 1

        out = {k: v.compute().item() for k, v in metrics.items()}
        out["loss"] = total_loss / total_n
        if confmat is not None:
            pca = _per_class_accuracy(confmat)
            out["per_class_acc"]  = pca.tolist()
            out["mean_class_acc"] = float(pca.mean())
            out["confmat"]        = confmat.tolist()
        return out

    # ------------------------------------------------------------------ #
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        ckpt_path: str,
        log_fn=None,
    ) -> Tuple[dict, float]:
        keys = ["loss", "accuracy", "precision", "recall", "f1"]
        history = {k: [] for k in keys + [f"val_{k}" for k in keys]}

        best_val_acc, best_state, patience = 0.0, None, 0

        for epoch in range(1, self.train_cfg.epochs + 1):
            t0 = time.time()
            tr = self._train_epoch(train_loader)
            va = self.evaluate(val_loader)
            self.scheduler.step()

            if va["accuracy"] > best_val_acc:
                best_val_acc = va["accuracy"]
                best_state   = copy.deepcopy(self.model.state_dict())
                torch.save(best_state, ckpt_path)
                patience, flag = 0, "✓"
            else:
                patience += 1
                flag = " "

            for k in keys:
                history[k].append(tr[k])
                history[f"val_{k}"].append(va[k])

            print(
                f"Epoch {epoch:3d}/{self.train_cfg.epochs} {flag} | "
                f"loss={tr['loss']:.4f} acc={tr['accuracy']:.4f} f1={tr['f1']:.4f} | "
                f"val_loss={va['loss']:.4f} val_acc={va['accuracy']:.4f} val_f1={va['f1']:.4f} | "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e} | "
                f"{time.time()-t0:.0f}s"
            )

            if log_fn is not None:
                log_fn(epoch, tr, va, self.optimizer.param_groups[0]["lr"])

            if patience >= self.train_cfg.early_stop:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no val_acc improvement for {self.train_cfg.early_stop} epochs).")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"\nRestored best weights (val_acc={best_val_acc:.4f}).")

        return history, best_val_acc
