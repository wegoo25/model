"""Building blocks shared by the margin-softmax and focal losses."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss on top of per-sample cross entropy (alpha=1, gamma=2 by default)."""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


def aam_logits(
    x: torch.Tensor,
    weight: torch.Tensor,
    label: torch.Tensor,
    margin: float,
    scale: float,
    easy_margin: bool = False,
) -> torch.Tensor:
    """Additive angular margin logits (ArcFace) used by ``aamsoftmax`` and ``focal``."""
    cos_m, sin_m = math.cos(margin), math.sin(margin)
    # make cos(theta + m) monotonic decreasing for theta in [0, 180] degrees
    th = math.cos(math.pi - margin)
    mm = math.sin(math.pi - margin) * margin

    cosine = F.linear(F.normalize(x), F.normalize(weight))
    sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
    phi = cosine * cos_m - sine * sin_m
    if easy_margin:
        phi = torch.where(cosine > 0, phi, cosine)
    else:
        phi = torch.where((cosine - th) > 0, phi, cosine - mm)

    one_hot = torch.zeros_like(cosine)
    one_hot.scatter_(1, label.view(-1, 1), 1)
    return ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * scale


def am_logits(
    x: torch.Tensor, weight: torch.Tensor, label: torch.Tensor, margin: float, scale: float
) -> torch.Tensor:
    """Additive margin (CosFace) logits; ``weight`` has shape ``(nOut, nClasses)``."""
    x_norm = torch.div(x, torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12))
    w_norm = torch.div(weight, torch.norm(weight, p=2, dim=0, keepdim=True).clamp(min=1e-12))
    costh = torch.mm(x_norm, w_norm)
    delta = torch.zeros_like(costh).scatter_(1, label.view(-1, 1), margin)
    return scale * (costh - delta)
