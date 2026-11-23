#!/usr/bin/env python
"""Plot DET curves of several models from saved ``scores.npy`` / ``labels.npy`` files.

    python scripts/plot_det.py --run "K=2:exps/a/model" --run "K=3:exps/b/model" --output det_curve.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from speakernet.metrics import compute_error_rates  # noqa: E402
from speakernet.plotting import plot_det_curves  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, metavar="NAME:DIR",
                   help="Legend name and a directory holding scores.npy and labels.npy")
    p.add_argument("--output", default="det_curve.png")
    p.add_argument("--linear", action="store_true", help="Linear instead of log axes")
    args = p.parse_args(argv)

    fnrs, fprs, names = [], [], []
    for item in args.run:
        name, _, directory = item.partition(":")
        scores = np.load(os.path.join(directory, "scores.npy"))
        labels = np.load(os.path.join(directory, "labels.npy"))
        fnr, fpr, _ = compute_error_rates(scores, labels)
        fnrs.append(fnr), fprs.append(fpr), names.append(name)
    plot_det_curves(fnrs, fprs, names, args.output, log_scale=not args.linear)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
