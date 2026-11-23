"""DDP (gloo, CPU) and MUSAN/RIR augmentation, exercised with synthetic data."""
import os
import random

import numpy as np
import pytest
import soundfile as sf
import torch

from speakernet.config import parse_args
from speakernet.data.augment import AugmentWAV
from speakernet.engine.runner import launch, validate_args

from test_pipeline import common

SR = 16000


@pytest.fixture(scope="module")
def aug_dirs(tmp_path_factory):
    """musan/{noise,speech,music}/<sub>/*.wav and rirs/<room>/<sub>/*.wav (one RIR = 5-sample delay)."""
    root = tmp_path_factory.mktemp("aug")
    rng = np.random.RandomState(1)
    for cat in ("noise", "speech", "music"):
        d = root / "musan" / cat / "sub"
        d.mkdir(parents=True)
        for i in range(8):
            sf.write(d / f"{i}.wav", (0.1 * rng.randn(SR)).astype(np.float32), SR)
    d = root / "rirs" / "smallroom" / "Room001"
    d.mkdir(parents=True)
    rir = np.zeros(200, dtype=np.float32)
    rir[5] = 1.0
    sf.write(d / "rir.wav", rir, SR)
    return root / "musan", root / "rirs"


def test_augmentation_branches(aug_dirs):
    aug = AugmentWAV(*map(str, aug_dirs))
    audio = 0.3 * torch.randn(1, 8000)

    # reverb with a pure 5-sample delay is an exact shift
    rev = aug.reverberate(audio)
    assert rev.shape == audio.shape
    torch.testing.assert_close(rev[:, 5:], audio[:, :-5], rtol=1e-4, atol=1e-5)
    assert rev[:, :5].abs().max() < 1e-5

    # a single noise at a fixed SNR lands on that SNR
    aug.noise_snr["noise"], aug.num_noise["noise"] = [10, 10], [1, 1]
    noisy = aug.additive_noise("noise", audio)
    added = noisy - audio
    snr = 10 * torch.log10(audio.pow(2).mean() / added.pow(2).mean())
    assert snr.item() == pytest.approx(10, abs=0.5)
    for cat in ("music", "speech"):
        assert aug.additive_noise(cat, audio).shape == audio.shape

    # the random chooser covers clean and augmented outputs, always with the input's shape
    random.seed(0)
    outs = [aug(audio) for _ in range(40)]
    assert all(o.shape == audio.shape for o in outs)
    assert any(torch.equal(o, audio) for o in outs) and any(not torch.equal(o, audio) for o in outs)


def test_augmentation_requires_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        AugmentWAV(str(tmp_path), str(tmp_path))


def test_training_with_augmentation(toy_data, aug_dirs, tmp_path):
    musan, rirs = aug_dirs
    args = parse_args(common(toy_data, tmp_path / "exp", extra=[
        "--augment", "true", "--musan_path", str(musan), "--rir_path", str(rirs), "--max_epoch", "1"]))
    validate_args(args, training=True)
    launch(args)
    assert os.path.exists(tmp_path / "exp" / "model" / "model000000001.model")


def test_ddp_two_processes_cpu(toy_data, tmp_path):
    """DistributedDataParallel over gloo with 2 spawned CPU processes: train, evaluate, checkpoint."""
    save = tmp_path / "ddp"
    port = str(20000 + os.getpid() % 20000)
    args = parse_args(common(toy_data, save, extra=["--distributed", "--world_size", "2", "--port", port, "--max_epoch", "2"]))
    validate_args(args, training=True)
    launch(args)
    assert os.path.exists(save / "model" / "model000000002.model")
    assert "VEER" in (save / "scores.txt").read_text()
    state = torch.load(save / "model" / "model000000002.model", weights_only=True)
    assert all(k.startswith(("__S__.", "__L__.")) for k in state)  # saved from the unwrapped module

    # distributed eval-only reproduces single-process scores for the same checkpoint
    ckpt = str(save / "model" / "model000000002.model")
    for name, extra in (("single", []), ("dist", ["--distributed", "--world_size", "2", "--port", str(int(port) + 1)])):
        a = parse_args(common(toy_data, tmp_path / name, extra=["--eval", "--initial_model", ckpt, *extra]))
        validate_args(a, training=False)
        launch(a)
    s = np.load(tmp_path / "single" / "model" / "scores.npy")
    d = np.load(tmp_path / "dist" / "model" / "scores.npy")
    np.testing.assert_allclose(s, d, atol=1e-5)
