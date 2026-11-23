"""Command-line + YAML configuration.

Precedence (highest first): explicit command-line flags, YAML ``--config`` file, defaults.
(The original script let YAML silently override command-line flags.)
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, List, Optional

import yaml


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "1", "yes", "y"):
        return True
    if str(value).lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SpeakerNet: speaker recognition / verification")
    p.add_argument("--config", type=str, default=None, help="YAML config file (flags override it)")

    g = p.add_argument_group("data loader")
    g.add_argument("--max_frames", type=int, default=200, help="Training crop length in frames (10 ms each)")
    g.add_argument("--eval_frames", type=int, default=300, help="Test crop length in frames (0 = whole file, waveform models only)")
    g.add_argument("--num_eval", type=int, default=10, help="Number of crops per test utterance")
    g.add_argument("--batch_size", type=int, default=200, help="Batch size (number of speakers per batch)")
    g.add_argument("--max_seg_per_spk", type=int, default=500, help="Maximum utterances per speaker per epoch")
    g.add_argument("--nDataLoaderThread", type=int, default=8, help="DataLoader worker processes")
    g.add_argument("--augment", type=str2bool, default=False, help="MUSAN/RIR augmentation")
    g.add_argument("--seed", type=int, default=3407, help="Random seed")

    g = p.add_argument_group("training")
    g.add_argument("--test_interval", type=int, default=2, help="LR-scheduler step size in epochs")
    g.add_argument("--eval_interval", type=int, default=3, help="Evaluate and checkpoint every N epochs (and at the last epoch)")
    g.add_argument("--max_epoch", type=int, default=500, help="Maximum number of epochs")
    g.add_argument("--trainfunc", type=str, default="", help="Loss function (module name in speakernet/losses)")
    g.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    g.add_argument("--scheduler", type=str, default="steplr", choices=["steplr"])
    g.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    g.add_argument("--lr_decay", type=float, default=0.95, help="LR decay factor every [test_interval] epochs")
    g.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")

    g = p.add_argument_group("loss functions")
    g.add_argument("--hard_prob", type=float, default=0.5, help="Hard negative mining probability (triplet)")
    g.add_argument("--hard_rank", type=int, default=10, help="Hard negative mining rank (triplet)")
    g.add_argument("--margin", type=float, default=0.3, help="Loss margin")
    g.add_argument("--scale", type=float, default=30, help="Loss scale")
    g.add_argument("--nPerSpeaker", type=int, default=1, help="Utterances per speaker per batch (metric-learning losses need >= 2)")
    g.add_argument("--nClasses", type=int, default=1211, help="Number of training speakers (classifier size)")

    g = p.add_argument_group("evaluation")
    g.add_argument("--dcf_p_target", type=float, default=0.05)
    g.add_argument("--dcf_c_miss", type=float, default=1)
    g.add_argument("--dcf_c_fa", type=float, default=1)

    g = p.add_argument_group("load and save")
    g.add_argument("--initial_model", type=str, default="", help="Checkpoint to initialise from")
    g.add_argument("--save_path", type=str, default="exps/exp1", help="Output directory for models and logs")

    g = p.add_argument_group("paths")
    g.add_argument("--train_list", type=str, default="", help="Lines: <speaker> <relative wav path>")
    g.add_argument("--train_path", type=str, default="", help="Root directory of the training wavs")
    g.add_argument("--test_list", type=str, default="", help="Lines: <label> <enroll wav> <test wav>")
    g.add_argument("--test_path", type=str, default="", help="Root directory of the test wavs")
    g.add_argument("--musan_path", type=str, default="", help="MUSAN root (for --augment)")
    g.add_argument("--rir_path", type=str, default="", help="RIRS_NOISES/simulated_rirs root (for --augment)")

    g = p.add_argument_group("model")
    g.add_argument("--model", type=str, default="", help="Model name (module in speakernet/models)")
    g.add_argument("--n_mels", type=int, default=40)
    g.add_argument("--log_input", type=str2bool, default=False)
    g.add_argument("--encoder_type", type=str, default="SAP")
    g.add_argument("--nOut", type=int, default=512, help="Embedding size")
    g.add_argument("--sinc_stride", type=int, default=10, help="First-layer stride of RawNet3")

    g = p.add_argument_group("run mode / hardware")
    g.add_argument("--eval", dest="eval", action="store_true", help="Evaluate only (no training)")
    g.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="'cuda' also selects AMD ROCm builds")
    g.add_argument("--port", type=str, default="8888", help="Master port for distributed training")
    g.add_argument("--world_size", type=int, default=0, help="Processes for --distributed (0 = number of GPUs)")
    g.add_argument("--distributed", action="store_true", help="Multi-GPU DistributedDataParallel")
    g.add_argument("--mixedprec", action="store_true", help="Automatic mixed precision (GPU only)")
    return p


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)

    if pre.config:
        with open(pre.config, "r") as f:
            yml = yaml.safe_load(f) or {}
        actions = {a.dest: a for a in parser._actions}
        defaults = {}
        for key, value in yml.items():
            action = actions.get(key)
            if action is None or key == "config":
                sys.stderr.write(f"Ignored unknown parameter {key} in yaml.\n")
                continue
            # YAML 1.1 reads e.g. 5e-05 as a string, so coerce through the argparse type.
            defaults[key] = action.type(value) if action.type is not None and value is not None else value
        parser.set_defaults(**defaults)

    return parser.parse_args(argv)
