"""MEK training loop.

Each step does TWO forward passes per batch (original + horizontally flipped),
combines a re-balanced label-smoothing loss on logits with a re-balanced
flip-consistency loss on attention maps:

loss = LSR2(logits_orig, label) + λ · ACLoss(hm_orig, hm_flip, balance_w)

This is the exact formulation from train_exp.py in the original repo, ported
to (resnet18 | resnet34 | resnet50) via the CAM head in `mek/model.py`.
"""
import copy
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score

from .config import MEKTrainCfg, MEKDatasetCfg
from .losses import ReBalancedLabelSmoothing, ac_loss, make_flip_grid


def _make_metrics(num_classes: int, device: torch.device) -> dict:
    return {
        "accuracy": Accuracy(task="multiclass", num_classes=num_classes).to(device),
        "precision": Precision(task="multiclass", num_classes=num_classes, average="macro").to(device),
        "recall": Recall(task="multiclass", num_classes=num_classes, average="macro").to(device),
        "f1": F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device),
    }


def _per_class_accuracy(confmat: np.ndarray) -> np.ndarray:
    """Diagonal / row-sum, with safe division."""
    rs = confmat.sum(axis=1)
    return np.where(rs > 0, np.diag(confmat) / np.maximum(rs, 1), 0.0)


class MEKTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_cfg: MEKTrainCfg,
        dataset_cfg: MEKDatasetCfg,
        balance_weights: torch.Tensor,
        device: torch.device,
        criterion: nn.Module = None,
        use_ema: bool = False,
        ema_decay: float = 0.999,
    ):
        self.model = model.to(device)
        self.train_cfg = train_cfg
        self.dataset_cfg = dataset_cfg
        self.device = device

        # Exponential moving average of the weights. Off by default (existing
        # callers are unaffected); the webcam runner turns it on for a small,
        # reliable accuracy/stability gain. Evaluation and checkpointing use the
        # EMA weights when enabled.
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_model = None
        if use_ema:
            self.ema_model = copy.deepcopy(self.model).eval()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

        self.balance_weights = balance_weights.to(device)
        if criterion is not None:
            self.criterion = criterion.to(device)
        else:
            self.criterion = ReBalancedLabelSmoothing(
                epsilon=train_cfg.label_smooth,
                balance_weights=self.balance_weights,
            ).to(device)

        # Plain CE for eval (we don't need RSL on eval — just clean accuracy).
        self.eval_criterion = nn.CrossEntropyLoss()

        # Paper's ResNet recipe: Adam + ExponentialLR(gamma=0.9). (The Swin-T
        # reference repo used SGD/cosine; the paper's ResNet experiments use Adam.)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=train_cfg.sched_gamma
        )
        self.amp_on = device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_on)

        # 7x7 grid for ResNet-at-224 attention maps (precomputed once).
        self._flip_grid = None  # built lazily on first batch (feature size)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_ema(self):
        msd = self.model.state_dict()
        for k, v in self.ema_model.state_dict().items():
            mv = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(self.ema_decay).add_(mv.detach(), alpha=1.0 - self.ema_decay)
            else:
                v.copy_(mv)              # buffers like BN num_batches_tracked

    # ------------------------------------------------------------------
    def _train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        nc = self.dataset_cfg.num_classes
        metrics = _make_metrics(nc, self.device)
        total_loss, total_n = 0.0, 0
        sum_lsr, sum_flip = 0.0, 0.0

        for img, label, img_flip in loader:
            img = img.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            img_flip = img_flip.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.amp_on):
                logits_o, hm_o = self.model(img)
                logits_f, hm_f = self.model(img_flip)

                if self._flip_grid is None:
                    h, w = hm_o.shape[-2:]
                    self._flip_grid = make_flip_grid(h, w).to(self.device)

                lsr_loss = self.criterion(logits_o, label)
                fl_loss = ac_loss(hm_o, hm_f, self._flip_grid, self.balance_weights)
                loss = lsr_loss + self.train_cfg.flip_loss_weight * fl_loss

            self.scaler.scale(loss).backward()
            # Clip per the original repo (max_norm=5).
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.use_ema:
                self._update_ema()

            total_loss += loss.item() * img.size(0)
            sum_lsr += lsr_loss.item() * img.size(0)
            sum_flip += fl_loss.item() * img.size(0)
            total_n += img.size(0)
            for m in metrics.values():
                m.update(logits_o.softmax(1), label)

        out = {k: v.compute().item() for k, v in metrics.items()}
        out["loss"] = total_loss / total_n
        out["lsr_loss"] = sum_lsr / total_n
        out["flip_loss"] = sum_flip / total_n
        return out

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, return_per_class: bool = False) -> dict:
        eval_model = self.ema_model if self.use_ema else self.model
        eval_model.eval()
        nc = self.dataset_cfg.num_classes
        metrics = _make_metrics(nc, self.device)
        total_loss, total_n = 0.0, 0
        confmat = np.zeros((nc, nc), dtype=np.int64) if return_per_class else None

        for img, label in loader:
            img = img.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            logits, _ = eval_model(img)
            loss = self.eval_criterion(logits, label)

            total_loss += loss.item() * img.size(0)
            total_n += img.size(0)
            preds = logits.argmax(1)
            for m in metrics.values():
                m.update(logits.softmax(1), label)

            if confmat is not None:
                for t, p in zip(label.cpu().numpy(), preds.cpu().numpy()):
                    confmat[t, p] += 1

        out = {k: v.compute().item() for k, v in metrics.items()}
        out["loss"] = total_loss / total_n
        if confmat is not None:
            pca = _per_class_accuracy(confmat)
            out["per_class_acc"] = pca.tolist()
            out["mean_class_acc"] = float(pca.mean())
            out["confmat"] = confmat.tolist()
        return out

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        ckpt_path: str,
        log_fn=None,
    ) -> Tuple[dict, float]:
        keys = ["loss", "accuracy", "precision", "recall", "f1"]
        history = {k: [] for k in keys + [f"val_{k}" for k in keys]}
        history["lsr_loss"] = []
        history["flip_loss"] = []

        best_val_acc, best_state, patience = 0.0, None, 0

        for epoch in range(1, self.train_cfg.epochs + 1):
            t0 = time.time()
            tr = self._train_epoch(train_loader)
            va = self.evaluate(val_loader)
            self.scheduler.step()

            if va["accuracy"] > best_val_acc:
                best_val_acc = va["accuracy"]
                src_model = self.ema_model if self.use_ema else self.model
                best_state = copy.deepcopy(src_model.state_dict())
                torch.save(best_state, ckpt_path)
                patience, flag = 0, "+"
            else:
                patience += 1
                flag = " "

            for k in keys:
                history[k].append(tr[k])
                history[f"val_{k}"].append(va[k])
            history["lsr_loss"].append(tr["lsr_loss"])
            history["flip_loss"].append(tr["flip_loss"])

            print(
                f"Epoch {epoch:3d}/{self.train_cfg.epochs} {flag} | "
                f"loss={tr['loss']:.4f} (lsr={tr['lsr_loss']:.4f} flip={tr['flip_loss']:.4f}) "
                f"acc={tr['accuracy']:.4f} f1={tr['f1']:.4f} | "
                f"val_acc={va['accuracy']:.4f} val_f1={va['f1']:.4f} | "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e} | "
                f"{time.time()-t0:.0f}s"
            )

            if log_fn is not None:
                log_fn(epoch, tr, va, self.optimizer.param_groups[0]["lr"])

            if patience >= self.train_cfg.early_stop:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(no val_acc improvement for {self.train_cfg.early_stop} epochs)."
                )
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            if self.use_ema:
                self.ema_model.load_state_dict(best_state)
            print(f"\nRestored best weights (val_acc={best_val_acc:.4f}).")

        return history, best_val_acc
