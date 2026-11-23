"""Embedding extraction and trial scoring."""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from speakernet.data.dataset import EvalDataset, read_trial_list
from speakernet.models import input_type
from speakernet.utils import get_logger

log = get_logger()


@torch.no_grad()
def extract_embeddings(
    model,
    files: Sequence[str],
    root: str,
    device: torch.device,
    model_name: str,
    eval_frames: int,
    num_eval: int = 10,
    num_workers: int = 0,
    distributed: bool = False,
    print_interval: int = 100,
) -> Dict[str, torch.Tensor]:
    """Return ``{file: (num_eval, emb_dim) CPU tensor}`` for every file (all ranks gathered)."""
    feature = input_type(model_name)
    dataset = EvalDataset(files, root, eval_frames, num_eval=num_eval, feature=feature)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if distributed else None
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=num_workers,
        sampler=sampler, pin_memory=device.type == "cuda",
    )
    rank = dist.get_rank() if distributed else 0

    was_training = model.training
    model.eval()
    feats: Dict[str, torch.Tensor] = {}
    start = time.time()
    for idx, (data, name) in enumerate(loader):
        data = data[0].to(device, non_blocking=True)
        if feature == "fbank":
            data = data.unsqueeze(1)  # (N, 1, T, F): some backbones need a channel axis; the rest squeeze it
        emb = model(data).detach().reshape(data.shape[0], -1).cpu()
        feats[name[0]] = emb
        if rank == 0 and idx % print_interval == 0:
            log.info("Embedding %d/%d (%.2f Hz), size %d", idx, len(loader), (idx + 1) / (time.time() - start), emb.shape[1])
    model.train(was_training)

    if distributed:
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, feats)
        feats = {k: v for part in gathered for k, v in part.items()}
    return feats


def score_trials(
    feats: Dict[str, torch.Tensor],
    pairs: Sequence[Tuple[str, str]],
    normalize: bool = True,
) -> np.ndarray:
    """Score = mean over crops of the per-crop cosine similarity (the original protocol)."""
    scores = np.empty(len(pairs), dtype=np.float64)
    cache: Dict[str, torch.Tensor] = {}

    def get(name: str) -> torch.Tensor:
        if name not in cache:
            e = feats[name]
            cache[name] = F.normalize(e, p=2, dim=1) if normalize else e
        return cache[name]

    for i, (a, b) in enumerate(pairs):
        scores[i] = F.cosine_similarity(get(a), get(b), dim=1).mean().item()
    return scores


def evaluate_from_list(
    model,
    test_list: str,
    test_path: str,
    device: torch.device,
    model_name: str,
    eval_frames: int,
    num_eval: int = 10,
    num_workers: int = 0,
    distributed: bool = False,
    **kwargs,
) -> Tuple[Optional[np.ndarray], List[Optional[int]], List[str]]:
    """Extract embeddings for all unique files of a trial list and score every trial.

    Returns ``(scores, labels, trials)``; on non-zero ranks ``scores`` is None.
    ``labels`` entries are None when the list has no label column.
    """
    labels, pairs = read_trial_list(test_list)
    files = sorted({f for pair in pairs for f in pair})
    feats = extract_embeddings(model, files, test_path, device, model_name, eval_frames,
                               num_eval, num_workers, distributed)
    if distributed and dist.get_rank() != 0:
        return None, labels, []
    inner = model.module if hasattr(model, "module") else model
    normalize = getattr(inner.__L__, "test_normalize", True) if inner.__L__ is not None else True
    scores = score_trials(feats, pairs, normalize)
    return scores, labels, [f"{a} {b}" for a, b in pairs]


def compute_metrics(scores, labels, dcf_p_target=0.05, dcf_c_miss=1.0, dcf_c_fa=1.0):
    """Return ``(eer_percent, min_dcf, threshold)`` or None when labels are missing / single-class."""
    from speakernet.metrics import compute_error_rates, compute_min_dcf, tune_threshold_from_score

    if any(l is None for l in labels):
        return None
    if len(set(labels)) < 2:
        log.warning("Trial list has a single class; EER/MinDCF undefined.")
        return None
    _, eer, _, _ = tune_threshold_from_score(scores, labels, [1, 0.1])
    fnrs, fprs, thresholds = compute_error_rates(scores, labels)
    mindcf, threshold = compute_min_dcf(fnrs, fprs, thresholds, dcf_p_target, dcf_c_miss, dcf_c_fa)
    return eer, mindcf, threshold
