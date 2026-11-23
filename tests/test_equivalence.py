"""Numerical equivalence of the refactored models/losses with the original code.

The original tree is exported from the git tag ``pre-refactor`` into a temp dir and executed in a
subprocess (its top-level ``models``/``loss`` packages would otherwise clash). Skipped when the tag
or git is unavailable.
"""
import random
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import torch

from speakernet.losses import LOSSES, build_loss
from speakernet.models import MODELS, ALIASES, build_model

ROOT = Path(__file__).resolve().parents[1]
HELPER = Path(__file__).with_name("_orig_reference.py")
ORIG_NAME = {v: k for k, v in ALIASES.items()}  # MKRC_GAT -> test_GAT
KW = dict(nOut=512, encoder_type="ASP", sinc_stride=10)


@pytest.fixture(scope="module")
def orig_tree(tmp_path_factory):
    d = tmp_path_factory.mktemp("orig")
    try:
        tar = subprocess.run(["git", "archive", "pre-refactor"], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git tag 'pre-refactor' not available")
    (d / "o.tar").write_bytes(tar)
    with tarfile.open(d / "o.tar") as t:
        t.extractall(d)
    # MKRC: the original sums in place into views of x; apply only the clone() fix to the original
    mkrc = d / "models" / "MKRC.py"
    mkrc.write_text(mkrc.read_text().replace("combined = subfeatures[i]\n", "combined = subfeatures[i].clone()\n"))
    return d


def run_reference(orig_tree, kind, name):
    out = orig_tree / f"{kind}_{name}.pt"
    subprocess.run([sys.executable, str(HELPER), str(orig_tree), str(out), kind, name], cwd=orig_tree, check=True, capture_output=True)
    return torch.load(out, weights_only=True)


@pytest.mark.parametrize("name", MODELS)
def test_model_matches_original(orig_tree, name):
    ref = run_reference(orig_tree, "model", ORIG_NAME.get(name, name))
    torch.manual_seed(0)
    model = build_model(name, **KW).eval()
    # identical parameter names/shapes, so original checkpoints load strictly
    assert list(model.state_dict()) == list(ref["state"])
    model.load_state_dict(ref["state"], strict=True)
    with torch.no_grad():
        y = model(ref["x"])
    torch.testing.assert_close(y, ref["y"], rtol=0, atol=1e-6)


@pytest.mark.parametrize("name", LOSSES)
def test_loss_matches_original(orig_tree, name):
    if name == "triplet":
        pytest.skip("original relies on F.pairwise_distance broadcasting removed in current torch")
    ref = run_reference(orig_tree, "loss", name)
    torch.manual_seed(0)
    random.seed(0)
    loss = build_loss(name, nOut=32, nClasses=6, margin=0.2, scale=30, hard_prob=0.5, hard_rank=1)
    loss.load_state_dict(ref["state"], strict=True)
    random.seed(2)
    nloss, prec = loss(ref["x"], ref["y"])
    torch.testing.assert_close(nloss.detach(), ref["loss"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(torch.as_tensor(prec).float(), ref["prec"], rtol=1e-5, atol=1e-4)
