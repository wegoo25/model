"""End-to-end smoke tests on synthetic data: train -> checkpoint -> resume -> eval -> embeddings."""
import os
import subprocess
import sys

import numpy as np
import torch

from speakernet.engine import SpeakerNet, load_checkpoint
from speakernet.engine.runner import launch, validate_args
from speakernet.config import parse_args

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def common(toy, save, model="TDNN", loss="aamsoftmax", extra=()):
    return [
        "--model", model, "--trainfunc", loss, "--train_list", str(toy / "train.txt"),
        "--train_path", str(toy), "--test_list", str(toy / "trials.txt"), "--test_path", str(toy),
        "--save_path", str(save), "--max_frames", "100", "--eval_frames", "150", "--batch_size", "3",
        "--nClasses", "6", "--nDataLoaderThread", "0", "--max_epoch", "2", "--eval_interval", "1",
        "--device", "cpu", *extra,
    ]


def run_train(argv):
    args = parse_args(argv)
    validate_args(args, training=not args.eval)
    launch(args)


def test_train_resume_eval(toy_data, tmp_path):
    save = tmp_path / "exp"
    run_train(common(toy_data, save))
    models = sorted(os.listdir(save / "model"))
    assert "model000000002.model" in models and "model000000002.eer" in models
    assert (save / "scores.txt").read_text().count("Epoch") >= 4

    state = torch.load(save / "model" / "model000000002.model", weights_only=True)
    assert all(k.startswith(("__S__.", "__L__.")) for k in state)

    # resume: asking for 3 epochs continues from epoch 3
    run_train(common(toy_data, save, extra=["--max_epoch", "3"]))
    assert os.path.exists(save / "model" / "model000000003.model")

    # eval-only
    run_train(common(toy_data, save, extra=["--eval", "--initial_model", str(save / "model" / "model000000003.model")]))
    scores, labels = np.load(save / "model" / "scores.npy"), np.load(save / "model" / "labels.npy")
    assert scores.dtype == np.float64 and len(scores) == len(labels) == 12


def test_scripts_cli(toy_data, tmp_path):
    save = tmp_path / "exp"
    env = dict(os.environ, PYTHONPATH=ROOT)
    argv = common(toy_data, save, model="MKRC", loss="softmax")
    subprocess.run([sys.executable, f"{ROOT}/scripts/train.py", *argv], check=True, env=env, capture_output=True)
    ckpt = str(save / "model" / "model000000002.model")
    out = tmp_path / "emb.pt"
    subprocess.run([sys.executable, f"{ROOT}/scripts/extract_embeddings.py", *argv, "--initial_model", ckpt,
                    "--wav_list", str(toy_data / "files.txt"), "--wav_root", str(toy_data), "--output", str(out)],
                   check=True, env=env, capture_output=True)
    feats = torch.load(out, weights_only=True)
    assert len(feats) == 6 and next(iter(feats.values())).shape == (10, 512)
    subprocess.run([sys.executable, f"{ROOT}/scripts/evaluate.py", *argv, "--initial_model", ckpt],
                   check=True, env=env, capture_output=True)
    subprocess.run([sys.executable, f"{ROOT}/scripts/plot_det.py", "--run", f"a:{save}/model", "--output", str(tmp_path / "d.png")],
                   check=True, env=env, capture_output=True)
    assert (tmp_path / "d.png").exists()


def test_metric_learning_loss_trains(toy_data, tmp_path):
    run_train(common(toy_data, tmp_path / "e", loss="angleproto", extra=["--nPerSpeaker", "2", "--batch_size", "2", "--max_epoch", "1"]))
    assert (tmp_path / "e" / "model" / "model000000001.model").exists()
