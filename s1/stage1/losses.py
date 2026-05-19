"""
Loss functions for imbalanced binary classification.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class focal loss.

    Useful when malicious label=1 is rare.
    """

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.alpha,
            reduction="none",
        )
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce

        # print("[INFO] losses.py ------ forward")

        return loss.mean()


def compute_class_alpha(labels: List[int], num_classes: int = 2) -> torch.Tensor:
    """
    Inverse-frequency class weights.
    """
    counts = np.bincount(np.array(labels, dtype=int), minlength=num_classes)
    counts = np.maximum(counts, 1)

    inv = 1.0 / counts
    alpha = inv / inv.sum() * num_classes

    return torch.tensor(alpha, dtype=torch.float32)
