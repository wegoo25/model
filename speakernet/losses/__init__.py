"""Loss registry. ``trainfunc`` in the configs is the module name, e.g. ``aamsoftmax``."""
from __future__ import annotations

import importlib
from typing import Tuple

import torch.nn as nn

LOSSES: Tuple[str, ...] = (
    "softmax",
    "amsoftmax",
    "aamsoftmax",
    "focal",
    "focal_am",
    "angleproto",
    "proto",
    "ge2e",
    "softmaxproto",
    "triplet",
)

# Losses that operate on groups of ``nPerSpeaker`` utterances rather than class labels.
METRIC_LEARNING_LOSSES = ("angleproto", "proto", "ge2e", "softmaxproto", "triplet")


def build_loss(name: str, **kwargs) -> nn.Module:
    """Instantiate ``speakernet.losses.<name>.LossFunction(**kwargs)``."""
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Available: {', '.join(LOSSES)}")
    return importlib.import_module(f"speakernet.losses.{name}").LossFunction(**kwargs)
