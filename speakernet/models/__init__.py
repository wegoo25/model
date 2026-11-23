"""Model registry.

``model`` in a config is the name of a module in this package that exposes
``MainModel(**kwargs)``. Modules are imported lazily so that optional
dependencies (einops, asteroid_filterbanks) are only needed by the models that
use them.
"""
from __future__ import annotations

import importlib
from typing import Dict, Tuple

import torch.nn as nn

MODELS: Tuple[str, ...] = (
    "AMCRN",
    "Att_BiMKRC",
    "BiMKRC",
    "MKRC",
    "MKRC_GAT",
    "MKRC_SE",
    "MKRC_SE2",
    "RawNet3",
    "TDNN",
    "VOT",
    "VOT_serial",
    "VOT_serial_Memory",
    "camp_plus",
    "ecapatdnn",
    "resnetse",
    "transformer",
)

# Names used by older configs/checkpoints. ``test_GAT`` was the runnable version of ``MKRC_GAT``.
ALIASES: Dict[str, str] = {"test_GAT": "MKRC_GAT"}

# Models that consume the raw waveform instead of 112-dim Kaldi fbank features.
WAVEFORM_MODELS = ("RawNet3",)


def canonical_name(name: str) -> str:
    return ALIASES.get(name, name)


def input_type(name: str) -> str:
    """``"waveform"`` or ``"fbank"`` — which front-end the dataset must provide."""
    return "waveform" if canonical_name(name) in WAVEFORM_MODELS else "fbank"


def build_model(name: str, **kwargs) -> nn.Module:
    """Instantiate ``speakernet.models.<name>.MainModel(**kwargs)``."""
    name = canonical_name(name)
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(MODELS)}")
    return importlib.import_module(f"speakernet.models.{name}").MainModel(**kwargs)
