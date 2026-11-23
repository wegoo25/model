#!/usr/bin/env python
"""Evaluate a checkpoint on a trial list (EER / MinDCF) and save scores.  Example:

    python scripts/evaluate.py --config configs/MKRC_AAM.yaml --initial_model exps/MKRC_AAM/model/model000000030.model \
        --test_list lists/veri_test2.txt --test_path /data/voxceleb1 --save_path exps/MKRC_AAM
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speakernet.config import parse_args  # noqa: E402
from speakernet.engine.runner import launch, validate_args  # noqa: E402
from speakernet.utils import get_logger  # noqa: E402


def main(argv=None):
    args = parse_args(argv)
    args.eval = True
    validate_args(args, training=False)
    if not args.initial_model:
        get_logger().warning("No --initial_model given: the newest checkpoint in <save_path>/model is used, if any.")
    launch(args)


if __name__ == "__main__":
    main()
