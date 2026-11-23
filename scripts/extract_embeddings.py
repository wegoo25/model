#!/usr/bin/env python
"""Extract speaker embeddings for a list of wav files into a ``.pt`` file.

    python scripts/extract_embeddings.py --config configs/MKRC_AAM.yaml --initial_model <ckpt> \
        --wav_list files.txt --wav_root /data/wavs --output embeddings.pt

``files.txt`` holds one relative wav path per line. The output is a dict
``{file: tensor(num_eval, emb_dim)}``; average over dim 0 for one vector per utterance.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from speakernet.config import parse_args  # noqa: E402
from speakernet.engine import SpeakerNet, extract_embeddings, latest_checkpoint, load_checkpoint  # noqa: E402
from speakernet.models import canonical_name  # noqa: E402
from speakernet.utils import get_logger, resolve_device  # noqa: E402


def main(argv=None):
    extra = argparse.ArgumentParser(add_help=False, description="Extract embeddings; accepts all train.py model/config flags too.")
    extra.add_argument("--wav_list", required=True)
    extra.add_argument("--wav_root", required=True)
    extra.add_argument("--output", required=True)
    if any(a in ("-h", "--help") for a in (argv if argv is not None else sys.argv[1:])):
        extra.print_help()
        parse_args(["--help"])
    ex, rest = extra.parse_known_args(argv)
    args = parse_args(rest)
    args.model = canonical_name(args.model)
    log = get_logger()

    device = resolve_device(args.device)
    args.trainfunc = args.trainfunc or "softmax"  # loss head is only needed to load a full checkpoint
    net = SpeakerNet(**vars(args)).to(device)
    ckpt = args.initial_model or latest_checkpoint(os.path.join(args.save_path, "model"))[0]
    if not ckpt:
        raise SystemExit("No checkpoint: pass --initial_model.")
    load_checkpoint(net, ckpt, device)

    with open(ex.wav_list) as f:
        files = [l.strip() for l in f if l.strip()]
    feats = extract_embeddings(net, files, ex.wav_root, device, args.model, args.eval_frames,
                               args.num_eval, args.nDataLoaderThread)
    torch.save(feats, ex.output)
    log.info("Saved %d embeddings to %s", len(feats), ex.output)


if __name__ == "__main__":
    main()
