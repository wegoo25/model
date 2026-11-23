"""DET-curve plotting (headless-safe)."""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_det_curves(
    fnrs: Sequence[np.ndarray],
    fprs: Sequence[np.ndarray],
    names: Sequence[str],
    save_path: str,
    log_scale: bool = True,
    limit: Optional[float] = None,
) -> None:
    """Plot one or more DET curves (rates in [0, 1]) as percentages and save to ``save_path``."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for fnr, fpr, name in zip(fnrs, fprs, names):
        ax.plot(np.asarray(fpr) * 100, np.asarray(fnr) * 100, label=name, linewidth=1)
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    if limit is not None:
        ax.set_xlim(0, limit * 100)
        ax.set_ylim(0, limit * 100)
    ax.set_xlabel("False Alarm Rate (%)")
    ax.set_ylabel("Miss Detection Rate (%)")
    ax.set_title("Detection Error Tradeoff (DET)")
    ax.grid(True)
    ax.legend()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
