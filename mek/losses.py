"""MEK losses.

Direct ports of `ACLoss` and `LSR2` from
zyh-uaiaaaa/Mine-Extra-Knowledge/code/train_exp.py, generalized so the
class-balance weight tensor can be computed from any dataset (instead of being
hard-coded for RAF-DB).
"""
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────
# Re-balanced attention map (RAM) — flip-consistency loss
# ────────────────────────────────────────────────────────────────────
def make_flip_grid(h: int, w: int) -> torch.Tensor:
    """Build the [1, 2, H, W] grid that horizontally flips a feature map via
    `F.grid_sample`. Same construction as the original repo."""
    x = torch.arange(w).view(1, -1).expand(h, -1)
    y = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x, y], dim=0).float().unsqueeze(0)   # [1, 2, H, W]
    grid[:, 0] = 2 * grid[:, 0] / (w - 1) - 1
    grid[:, 1] = 2 * grid[:, 1] / (h - 1) - 1
    grid[:, 0] = -grid[:, 0]                                  # horizontal flip
    return grid


def ac_loss(
    hm_orig: torch.Tensor,
    hm_flip: torch.Tensor,
    flip_grid: torch.Tensor,
    balance_w: torch.Tensor,
) -> torch.Tensor:
    """Flip-consistency loss with per-class re-balancing.

    Args:
        hm_orig:  attention map of the original image, [B, K, H, W]
        hm_flip:  attention map of the horizontally-flipped image, [B, K, H, W]
        flip_grid: precomputed flip grid, [1, 2, H, W]
        balance_w: per-class weight (≈ 1/freq, mean ≈ 1), [K]

    Returns:
        scalar loss.
    """
    B = hm_orig.size(0)
    grid = flip_grid.expand(B, -1, -1, -1).permute(0, 2, 3, 1)   # [B, H, W, 2]
    hm_flip_unflipped = F.grid_sample(
        hm_flip, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    flip_loss = F.mse_loss(hm_orig, hm_flip_unflipped, reduction="none")  # [B, K, H, W]
    flip_loss = flip_loss.mean(dim=[-1, -2])                              # [B, K]
    # Per-class weighting: pushes the model to learn flip-invariant attention
    # *especially* for under-represented classes.
    flip_loss = flip_loss @ balance_w                                     # [B]
    return flip_loss.mean()


# ────────────────────────────────────────────────────────────────────
# Re-balanced smooth label (RSL) — drop-in for nn.CrossEntropyLoss
# ────────────────────────────────────────────────────────────────────
class ReBalancedLabelSmoothing(nn.Module):
    """LSR2 from the MEK repo.

    Standard label smoothing puts (1-ε) on the target class and ε/(K-1) uniformly
    on the others. RSL instead distributes the ε mass *proportionally to the
    class-imbalance weights*, giving more soft probability to minor classes.
    """
    def __init__(self, epsilon: float, balance_weights: torch.Tensor):
        super().__init__()
        self.epsilon = float(epsilon)
        self.register_buffer("balance_weights", balance_weights.float())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        K = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)

        # Target slot gets (1 - ε); we'll fill the rest below.
        smooth = torch.zeros_like(logits)
        smooth.scatter_(1, target.view(-1, 1), 1.0 - self.epsilon)

        # Distribute ε across non-target classes proportionally to balance weights.
        mask = (smooth == 0)                                          # non-target positions
        bw = self.balance_weights.unsqueeze(0).expand_as(logits)      # [B, K]
        bw_masked = bw * mask
        bw_norm = bw_masked / bw_masked.sum(dim=1, keepdim=True).clamp_min(1e-8)
        smooth = smooth + bw_norm * self.epsilon

        loss = -(log_probs * smooth).sum(dim=1).mean()
        return loss


# ────────────────────────────────────────────────────────────────────
# Class-balance weight tensor
# ────────────────────────────────────────────────────────────────────
def compute_balance_weights(class_counts: Iterable[int]) -> torch.Tensor:
    """Inverse-frequency weights, normalized to mean=1.

    For RAF-DB this produces values close to the constants hard-coded in the
    original repo (e.g. ≈ 4.4 for 'fear', ≈ 0.26 for 'happy')."""
    counts = np.asarray(list(class_counts), dtype=np.float64)
    inv = 1.0 / np.maximum(counts, 1.0)
    weights = inv * (len(inv) / inv.sum())                            # mean = 1
    return torch.from_numpy(weights).float()
