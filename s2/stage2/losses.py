from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_alpha(labels: Iterable[int], num_classes: int = 2) -> torch.Tensor:
    labels = np.asarray(list(labels), dtype=int)
    counts = np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1)
    total = counts.sum()
    alpha = total / (num_classes * counts)
    alpha = alpha / alpha.sum()
    print(f"[INFO] Stage2 class distribution: {counts.tolist()}")
    print(f"[INFO] Stage2 alpha weights: {alpha.tolist()}")
    return torch.tensor(alpha, dtype=torch.float32)


class FocalLossWithLabelSmoothing(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.alpha.to(logits.device) if self.alpha is not None else None,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce).clamp(min=1e-8, max=1.0)
        return (((1.0 - pt) ** self.gamma) * ce).mean()
