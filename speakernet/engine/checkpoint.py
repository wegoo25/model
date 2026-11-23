"""Checkpoint saving / loading, compatible with the original ``model%09d.model`` files."""
from __future__ import annotations

import glob
import os
import re
from typing import Optional, Tuple

import torch
import torch.nn as nn

from speakernet.utils import get_logger

log = get_logger()


def save_checkpoint(model: nn.Module, path: str, eer: Optional[float] = None) -> None:
    """Save ``model.state_dict()`` (keys ``__S__.*`` / ``__L__.*``); optional ``.eer`` sidecar file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)
    if eer is not None:
        with open(os.path.splitext(path)[0] + ".eer", "w") as f:
            f.write("{:2.4f}".format(eer))


def load_checkpoint(model: nn.Module, path: str, device: torch.device | str = "cpu") -> Tuple[int, int]:
    """Leniently load weights; returns ``(loaded, skipped)`` parameter counts.

    Handles ``module.`` prefixes (DDP), the ``{"model": backbone_state}`` format of
    the original loader, and skips missing / size-mismatched entries with a warning.
    """
    state = torch.load(path, map_location=device, weights_only=True)
    if len(state) == 1 and "model" in state:
        state = {"__S__." + k: v for k, v in state["model"].items()}

    own = model.state_dict()
    loaded = skipped = 0
    for name, param in state.items():
        key = name
        if key not in own:
            key = key.replace("module.", "")
        if key not in own:
            log.warning("%s is not in the model.", name)
            skipped += 1
            continue
        if own[key].size() != param.size():
            log.warning("Wrong parameter size: %s, model: %s, loaded: %s", name, tuple(own[key].size()), tuple(param.size()))
            skipped += 1
            continue
        own[key].copy_(param)
        loaded += 1
    return loaded, skipped


def latest_checkpoint(model_dir: str) -> Tuple[Optional[str], int]:
    """Return ``(path, epoch)`` of the newest ``model<epoch>.model`` in ``model_dir`` (or ``(None, 0)``)."""
    best, best_epoch = None, 0
    for path in glob.glob(os.path.join(model_dir, "model0*.model")):
        m = re.fullmatch(r"model(\d+)\.model", os.path.basename(path))
        if m and int(m.group(1)) >= best_epoch:
            best, best_epoch = path, int(m.group(1))
    return best, best_epoch
