"""Training loop: optimizer, LR schedule, optional mixed precision."""
from __future__ import annotations

import time
import warnings
from typing import Tuple

import torch
import torch.nn as nn

from speakernet.utils import get_logger

log = get_logger()


class Trainer:
    """Wraps a :class:`SpeakerNet` (optionally inside DDP) with its optimizer and scheduler.

    Optimizers/scheduler match the original: Adam or SGD(momentum=0.9) with ``lr`` and
    ``weight_decay``; ``StepLR(step_size=test_interval, gamma=lr_decay)`` stepped per epoch.
    """

    def __init__(self, model: nn.Module, device: torch.device, /, optimizer: str = "adam", lr: float = 5e-4,
                 weight_decay: float = 0.0, test_interval: int = 1, lr_decay: float = 0.95,
                 mixedprec: bool = False, scheduler: str = "steplr", **kwargs):
        self.model = model  # possibly DDP-wrapped
        self.raw = model.module if hasattr(model, "module") else model
        self.device = device

        params = self.model.parameters()
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer '{optimizer}'")
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=test_interval, gamma=lr_decay)

        self.use_amp = bool(mixedprec) and device.type == "cuda"
        if mixedprec and not self.use_amp:
            log.warning("--mixedprec ignored: only supported on CUDA/ROCm devices.")
        self.raw.use_amp = self.use_amp
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def advance_scheduler(self, epochs: int) -> None:
        """Replay ``epochs`` scheduler steps (used when resuming)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # "scheduler.step() before optimizer.step()" is expected here
            for _ in range(epochs):
                self.scheduler.step()

    def current_lr(self) -> float:
        return max(g["lr"] for g in self.optimizer.param_groups)

    def train_epoch(self, loader, verbose: bool = True, log_interval: int = 20) -> Tuple[float, float]:
        """One epoch; returns ``(mean loss, mean precision %)``."""
        self.model.train()
        total_loss = total_prec = 0.0
        count = 0
        start = time.time()
        non_blocking = self.device.type == "cuda"

        for data, label in loader:
            data = data.to(self.device, non_blocking=non_blocking)
            label = label.to(self.device, non_blocking=non_blocking)

            self.optimizer.zero_grad(set_to_none=True)
            loss, prec1, _ = self.model(data, label)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # one host sync per step for logging
            total_loss += loss.item()
            total_prec += prec1.item()
            count += 1
            if verbose and count % log_interval == 0:
                rate = count * loader.batch_size / (time.time() - start)
                log.info("Batch %d/%d loss %.4f acc %.2f%% (%.1f utt/s)", count, len(loader),
                         total_loss / count, total_prec / count, rate)

        self.scheduler.step()
        if count == 0:
            raise RuntimeError("The training loader produced no batches (dataset smaller than batch_size?).")
        return total_loss / count, total_prec / count
