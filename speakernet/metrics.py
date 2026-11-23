"""Speaker-verification metrics: EER, DET error rates and minimum DCF.

Numerically equivalent to the original ``tuneThreshold.py`` implementation
(vectorised with NumPy instead of Python loops).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn import metrics as sk_metrics


def tune_threshold_from_score(
    scores: Sequence[float],
    labels: Sequence[int],
    target_fa: Sequence[float],
    target_fr: Optional[Sequence[float]] = None,
):
    """Return ``(tuned_thresholds, eer_percent, fpr, fnr)``.

    ``tuned_thresholds`` holds ``[threshold, fpr, fnr]`` entries: first one per
    target false-rejection rate, then one per target false-alarm rate.
    """
    fpr, tpr, thresholds = sk_metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    tuned = []
    if target_fr:
        for tfr in target_fr:
            idx = np.nanargmin(np.absolute(tfr - fnr))
            tuned.append([thresholds[idx], fpr[idx], fnr[idx]])

    for tfa in target_fa:
        idx = np.nanargmin(np.absolute(tfa - fpr))
        tuned.append([thresholds[idx], fpr[idx], fnr[idx]])

    idx_eer = np.nanargmin(np.absolute(fnr - fpr))
    eer = max(fpr[idx_eer], fnr[idx_eer]) * 100
    return tuned, eer, fpr, fnr


def compute_error_rates(
    scores: Sequence[float], labels: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """False-negative / false-positive rates at every score used as a threshold.

    Scores are sorted ascending (stable); ``fnrs[i]`` is the fraction of target
    trials rejected at threshold ``i`` and ``fprs[i]`` the fraction of
    non-target trials accepted.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    order = np.argsort(scores, kind="stable")
    thresholds = scores[order]
    sorted_labels = labels[order]

    n_target = sorted_labels.sum()
    n_nontarget = len(sorted_labels) - n_target
    if n_target == 0 or n_nontarget == 0:
        raise ValueError("Both target and non-target trials are required to compute error rates.")

    fnrs = np.cumsum(sorted_labels) / float(n_target)
    fprs = 1 - np.cumsum(1 - sorted_labels) / float(n_nontarget)
    return fnrs, fprs, thresholds


def compute_min_dcf(
    fnrs: np.ndarray,
    fprs: np.ndarray,
    thresholds: np.ndarray,
    p_target: float,
    c_miss: float,
    c_fa: float,
) -> Tuple[float, float]:
    """Minimum normalised detection cost (NIST SRE16 eqs. 2-4) and its threshold."""
    c_det = c_miss * np.asarray(fnrs) * p_target + c_fa * np.asarray(fprs) * (1 - p_target)
    best = int(np.argmin(c_det))
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    return float(c_det[best] / c_def), float(thresholds[best])
