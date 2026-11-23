"""Losses, metrics, features, dataset and sampler."""
import os

import numpy as np
import pytest
import torch
from sklearn import metrics as skm

from speakernet.data.dataset import EvalDataset, TrainDataset, read_trial_list
from speakernet.data.features import SAMPLE_RATE, compute_fbank, load_audio, num_fbank_frames, samples_for_frames, wrap_pad
from speakernet.data.sampler import TrainSampler
from speakernet.losses import LOSSES, METRIC_LEARNING_LOSSES, build_loss
from speakernet.metrics import compute_error_rates, compute_min_dcf, tune_threshold_from_score


@pytest.mark.parametrize("name", [l for l in LOSSES if l not in METRIC_LEARNING_LOSSES])
def test_classification_losses(name):
    loss = build_loss(name, nOut=32, nClasses=6, margin=0.2, scale=30)
    x = torch.randn(8, 32, requires_grad=True)
    y = torch.randint(0, 5, (8,))
    nloss, prec = loss(x, y)
    nloss.backward()
    assert torch.isfinite(nloss) and x.grad is not None and 0 <= prec.item() <= 100


@pytest.mark.parametrize("name", METRIC_LEARNING_LOSSES)
def test_metric_learning_losses(name):
    loss = build_loss(name, nOut=32, nClasses=6, margin=0.2, scale=30, hard_prob=0.5, hard_rank=1)
    x = torch.randn(12, 2, 32, requires_grad=True)
    y = torch.arange(12) % 6
    nloss, _ = loss(x, y)
    nloss.backward()
    assert torch.isfinite(nloss) and x.grad is not None


def test_metrics_match_bruteforce():
    rng = np.random.RandomState(0)
    labels = rng.randint(0, 2, 300)
    scores = rng.randn(300) + labels
    fnrs, fprs, thr = compute_error_rates(scores, labels)
    for i in (0, 57, 150, 299):  # threshold = score i, "accept" if score > threshold
        accept = scores > thr[i]
        assert fnrs[i] == pytest.approx(((~accept) & (labels == 1)).sum() / (labels == 1).sum())
        assert fprs[i] == pytest.approx((accept & (labels == 0)).sum() / (labels == 0).sum())
    _, eer, fpr, fnr = tune_threshold_from_score(scores, labels, [1, 0.1])
    fpr_s, tpr_s, _ = skm.roc_curve(labels, scores)
    assert eer / 100 == pytest.approx(max(fpr_s[np.argmin(abs((1 - tpr_s) - fpr_s))], 1 - tpr_s[np.argmin(abs((1 - tpr_s) - fpr_s))]))
    dcf, _ = compute_min_dcf(fnrs, fprs, thr, 0.05, 1, 1)
    assert 0 < dcf


def test_fbank_shapes_and_crop_equivalence():
    wav = torch.randn(1, SAMPLE_RATE * 2)
    full = compute_fbank(wav)
    assert full.shape == (num_fbank_frames(wav.shape[-1]), 112)
    start, n = 37, 50
    crop = compute_fbank(wav[:, start * 160:start * 160 + samples_for_frames(n)])
    torch.testing.assert_close(crop, full[start:start + n], atol=1e-4, rtol=1e-4)
    assert wrap_pad(torch.randn(1, 100), 250).shape[-1] == 250


def test_dataset_and_sampler(toy_data):
    ds = TrainDataset(str(toy_data / "train.txt"), str(toy_data), max_frames=100)
    assert ds.num_speakers == 6 and len(ds) == 24
    x, y = ds[[0, 1]]
    assert x.shape == (2, 100, 112) and y == ds.data_label[1]

    sampler = TrainSampler(ds, nPerSpeaker=1, max_seg_per_spk=10, batch_size=3, distributed=False, seed=1)
    groups = list(sampler)
    assert len(groups) == len(sampler) and len(groups) % 3 == 0
    for i in range(0, len(groups), 3):  # no repeated speaker inside a batch
        labels = [ds.data_label[g[0]] for g in groups[i:i + 3]]
        assert len(set(labels)) == 3
    sampler.set_epoch(1)
    assert list(sampler) != groups


def test_eval_dataset_and_trials(toy_data):
    labels, pairs = read_trial_list(str(toy_data / "trials.txt"))
    assert labels[:2] == [1, 0] and len(pairs) == 12
    ds = EvalDataset([pairs[0][0]], str(toy_data), eval_frames=150, num_eval=10)
    x, name = ds[0]
    assert x.shape == (10, 150, 112)
    wav = EvalDataset([pairs[0][0]], str(toy_data), eval_frames=0, feature="waveform")[0][0]
    assert wav.shape[0] == 1 and wav.shape[1] == load_audio(str(toy_data / pairs[0][0])).shape[1]


def test_every_config_is_valid_and_builds():
    """Each shipped YAML parses, names a registered model/loss, and its nOut matches the model's embedding size."""
    import glob

    from speakernet.config import parse_args
    from speakernet.models import build_model, canonical_name, input_type

    emb_dim = {"MKRC_GAT": 256}
    paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "configs", "*.yaml")))
    assert len(paths) >= 28
    for path in paths:
        args = parse_args(["--config", path])
        assert args.trainfunc in LOSSES, path
        name = canonical_name(args.model)
        model = build_model(name, **{k: getattr(args, k) for k in ("nOut", "encoder_type", "sinc_stride")})
        if input_type(name) != "waveform":  # only RawNet3 honours nOut
            assert args.nOut == emb_dim.get(name, 512), f"{path}: nOut={args.nOut}"
        del model
