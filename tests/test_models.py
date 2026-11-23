"""Every registered model: forward shape, backward gradients and checkpoint round-trip."""
import pytest
import torch

from speakernet.engine import SpeakerNet, load_checkpoint, save_checkpoint
from speakernet.models import MODELS, build_model, input_type

KW = dict(nOut=512, encoder_type="ASP", sinc_stride=10, n_mels=40, log_input=True)


# These models hard-code a sequence length of 301 inside their attention blocks.
FIXED_300 = ("VOT", "VOT_serial", "VOT_serial_Memory", "transformer")
EMB_DIM = {"MKRC_GAT": 256}


def make_input(name, batch=2, frames=None):
    frames = frames or (300 if name in FIXED_300 else 100)
    if input_type(name) == "waveform":
        return torch.randn(batch, 1, frames * 160)
    return torch.randn(batch, 1, frames, 112)


@pytest.mark.parametrize("name", MODELS)
def test_forward_backward(name):
    model = build_model(name, **KW).train()
    out = model(make_input(name))
    out = out.reshape(out.shape[0], -1)
    assert out.shape == (2, EMB_DIM.get(name, 512)), out.shape
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("name", MODELS)
def test_eval_accepts_channel_axis(name):
    """The evaluator feeds (N, 1, T, F); the 3D (N, T, F) layout of the original must agree."""
    model = build_model(name, **KW).eval()
    x = make_input(name, batch=3)
    with torch.no_grad():
        a = model(x).reshape(3, -1)
        if input_type(name) == "fbank" and name != "MKRC_SE2":
            b = model(x.squeeze(1)).reshape(3, -1)
            torch.testing.assert_close(a, b)


def test_checkpoint_roundtrip_and_legacy_formats(tmp_path):
    kw = dict(model="TDNN", trainfunc="aamsoftmax", nPerSpeaker=1, nClasses=7, **KW)
    a, b = SpeakerNet(**kw), SpeakerNet(**kw)
    save_checkpoint(a, str(tmp_path / "model000000001.model"), eer=3.0)
    assert (tmp_path / "model000000001.eer").read_text() == "3.0000"
    loaded, skipped = load_checkpoint(b, str(tmp_path / "model000000001.model"))
    assert skipped == 0 and loaded == len(a.state_dict())
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k])

    # DDP-style "module." prefix and the {"model": backbone} format of the original loader
    torch.save({"module." + k: v for k, v in a.state_dict().items()}, tmp_path / "ddp.model")
    c = SpeakerNet(**kw)
    assert load_checkpoint(c, str(tmp_path / "ddp.model"))[1] == 0
    torch.save({"model": a.__S__.state_dict()}, tmp_path / "bb.model")
    d = SpeakerNet(**kw)
    loaded, _ = load_checkpoint(d, str(tmp_path / "bb.model"))
    assert loaded == len(a.__S__.state_dict())
    assert torch.equal(next(iter(d.__S__.state_dict().values())), next(iter(a.__S__.state_dict().values())))


def test_checkpoint_skips_mismatched_shapes(tmp_path):
    kw = dict(model="TDNN", trainfunc="aamsoftmax", nPerSpeaker=1, **KW)
    a, b = SpeakerNet(nClasses=7, **kw), SpeakerNet(nClasses=9, **kw)
    save_checkpoint(a, str(tmp_path / "m.model"))
    loaded, skipped = load_checkpoint(b, str(tmp_path / "m.model"))
    assert skipped == 1  # the classifier weight only
