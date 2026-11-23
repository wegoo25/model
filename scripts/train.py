#!/usr/bin/env python
"""Train a speaker model.  Example:

    python scripts/train.py --config configs/MKRC_AAM.yaml \
        --train_list lists/train_list.txt --train_path /data/voxceleb2 \
        --test_list lists/veri_test2.txt --test_path /data/voxceleb1 --save_path exps/MKRC_AAM
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speakernet.config import parse_args  # noqa: E402
from speakernet.engine.runner import launch, validate_args  # noqa: E402


def main(argv=None):
    args = parse_args(argv)
    validate_args(args, training=not args.eval)
    launch(args)


if __name__ == "__main__":
    main()
