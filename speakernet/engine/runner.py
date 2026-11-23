"""High-level routines shared by the command-line scripts."""
from __future__ import annotations

import argparse
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from speakernet.data.dataset import TrainDataset, worker_init_fn
from speakernet.data.sampler import TrainSampler
from speakernet.engine.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from speakernet.engine.evaluation import compute_metrics, evaluate_from_list
from speakernet.engine.speaker_net import SpeakerNet
from speakernet.engine.trainer import Trainer
from speakernet.losses import LOSSES
from speakernet.models import canonical_name, input_type
from speakernet.utils import get_logger, resolve_device, set_seed

log = get_logger()


def validate_args(args: argparse.Namespace, training: bool) -> None:
    if not args.model:
        raise ValueError("--model is required (see speakernet/models/__init__.py for the list).")
    args.model = canonical_name(args.model)
    if not args.trainfunc:
        raise ValueError(f"--trainfunc is required, one of: {', '.join(LOSSES)}")
    if training and not (args.train_list and args.train_path):
        raise ValueError("--train_list and --train_path are required for training.")
    if (args.eval or not training) and not (args.test_list and args.test_path):
        raise ValueError("--test_list and --test_path are required for evaluation.")
    if input_type(args.model) == "waveform" and args.eval_frames < 0:
        raise ValueError("--eval_frames must be >= 0")
    if input_type(args.model) == "fbank" and args.eval_frames <= 0:
        raise ValueError("--eval_frames must be > 0 for feature-based models.")


def build_speaker_net(args: argparse.Namespace, device: torch.device) -> SpeakerNet:
    return SpeakerNet(**vars(args)).to(device)


def run_evaluation(model, args, device, rank: int = 0, save_dir: Optional[str] = None) -> Optional[Tuple[float, float, float]]:
    """Score ``args.test_list``; on rank 0 log EER / MinDCF and save scores/labels/trials."""
    scores, labels, trials = evaluate_from_list(
        model, args.test_list, args.test_path, device, args.model, args.eval_frames,
        num_eval=args.num_eval, num_workers=args.nDataLoaderThread, distributed=args.distributed,
    )
    if rank != 0:
        return None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "scores.npy"), scores)  # raw scores (the original overwrote these with booleans)
        if all(l is not None for l in labels):
            np.save(os.path.join(save_dir, "labels.npy"), np.asarray(labels))
        with open(os.path.join(save_dir, "trials_scored.txt"), "w") as f:
            f.writelines(f"{s:.6f} {t}\n" for s, t in zip(scores, trials))
    result = compute_metrics(scores, labels, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)
    if result is None:
        log.info("Scored %d trials (no labels, EER/MinDCF skipped).", len(scores))
    return result


def _make_loader(dataset, sampler, args, device):
    workers = args.nDataLoaderThread
    return torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=workers, sampler=sampler,
        pin_memory=device.type == "cuda", worker_init_fn=worker_init_fn,
        drop_last=True, persistent_workers=workers > 0,
    )


def main_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    """Training (or ``--eval``) on one process; ``rank`` is the GPU index when distributed."""
    set_seed(args.seed)
    device = resolve_device(args.device, rank if args.distributed else 0)
    is_main = rank == 0

    model_dir = os.path.join(args.save_path, "model")
    result_dir = os.path.join(args.save_path, "result")
    if is_main:
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

    net = build_speaker_net(args, device)
    if args.distributed:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = args.port
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", world_size=world_size, rank=rank)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=True)
        log.info("Rank %d/%d ready on %s", rank, world_size, device)

    trainer = Trainer(net, device, **vars(args))
    raw = trainer.raw

    epoch = 1
    ckpt, last_epoch = latest_checkpoint(model_dir)
    if args.initial_model:
        load_checkpoint(raw, args.initial_model, device)
        log.info("Model %s loaded!", args.initial_model)
    elif ckpt:
        load_checkpoint(raw, ckpt, device)
        log.info("Model %s loaded from previous state!", ckpt)
        epoch = last_epoch + 1
    trainer.advance_scheduler(epoch - 1)

    log.info("Backbone parameters: %d", sum(p.numel() for p in raw.__S__.parameters()))

    if args.eval:
        result = run_evaluation(raw, args, device, rank, save_dir=model_dir)
        if result:
            log.info("VEER %2.4f MinDCF %2.5f Threshold %2.5f", *result)
        return

    dataset = TrainDataset(feature=input_type(args.model), **vars(args))
    if dataset.num_speakers > args.nClasses and args.trainfunc not in ("angleproto", "proto", "ge2e", "triplet"):
        raise ValueError(f"Training list has {dataset.num_speakers} speakers but --nClasses={args.nClasses}.")
    sampler = TrainSampler(dataset, **vars(args))
    loader = _make_loader(dataset, sampler, args, device)

    score_file = open(os.path.join(args.save_path, "scores.txt"), "a+") if is_main else None
    try:
        for it in range(epoch, args.max_epoch + 1):
            sampler.set_epoch(it)
            clr_before = trainer.current_lr()
            t0 = time.time()
            loss, prec = trainer.train_epoch(loader, verbose=is_main)
            if is_main:
                msg = "Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f} ({:.0f}s)".format(
                    it, prec, loss, clr_before, time.time() - t0)
                log.info(msg)
                score_file.write(msg + "\n")

            if it % args.eval_interval == 0 or it == args.max_epoch:
                eer = None
                if args.test_list:
                    result = run_evaluation(raw, args, device, rank, save_dir=None)
                    if is_main and result:
                        eer = result[0]
                        msg = "Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}".format(it, result[0], result[1])
                        log.info(msg)
                        score_file.write(msg + "\n")
                if is_main:
                    save_checkpoint(raw, os.path.join(model_dir, "model%09d.model" % it), eer)
                    score_file.flush()
    finally:
        if score_file:
            score_file.close()
        if args.distributed:
            dist.destroy_process_group()


def launch(args: argparse.Namespace) -> None:
    """Spawn ``--world_size`` processes (default: one per GPU) when ``--distributed``; otherwise run in-process.

    Without GPUs, ``--world_size N --device cpu`` runs DDP over gloo (useful for testing).
    """
    if args.distributed:
        world = args.world_size or torch.cuda.device_count()
        if world < 1:
            raise RuntimeError("--distributed needs a CUDA/ROCm device, or --world_size N with --device cpu.")
        mp.spawn(main_worker, nprocs=world, args=(world, args))
    else:
        main_worker(0, 1, args)
