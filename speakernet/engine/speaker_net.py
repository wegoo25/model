"""Backbone + loss container.

The attribute names ``__S__`` (backbone) and ``__L__`` (loss) are deliberately kept:
they define the ``state_dict`` keys of every existing checkpoint.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from speakernet.losses import METRIC_LEARNING_LOSSES, build_loss
from speakernet.models import build_model


class SpeakerNet(nn.Module):
    def __init__(self, model: str, trainfunc: str, nPerSpeaker: int = 1, **kwargs):
        super().__init__()
        if trainfunc in METRIC_LEARNING_LOSSES and nPerSpeaker < 2:
            raise ValueError(f"Loss '{trainfunc}' needs nPerSpeaker >= 2 (got {nPerSpeaker}).")
        self.__S__ = build_model(model, **kwargs)
        self.__L__ = build_loss(trainfunc, **kwargs) if trainfunc else None
        self.nPerSpeaker = nPerSpeaker
        self.use_amp = False  # set by the trainer; only the backbone runs under autocast

    def embed(self, data: torch.Tensor) -> torch.Tensor:
        """Backbone forward under optional autocast; returns float32 embeddings."""
        with torch.autocast(device_type=data.device.type, enabled=self.use_amp):
            out = self.__S__(data)
        return out.float()

    def forward(
        self, data: torch.Tensor, label: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """``data``: ``(B, nPerSpeaker, ...)`` for training, ``(N, ...)`` for embedding extraction.

        Returns embeddings when ``label`` is None, else ``(loss, precision, embeddings)``.
        """
        if label is None:
            return self.embed(data)

        n = self.nPerSpeaker
        if n > 1:
            # (B, n, ...) -> (B*n, 1, ...) so each utterance goes through the backbone alone.
            data = data.reshape(-1, 1, *data.shape[2:])
        out = self.embed(data)
        if n > 1:
            out = out.reshape(-1, n, out.size(-1))
        loss, prec1 = self.__L__(out, label)
        return loss, prec1, out
