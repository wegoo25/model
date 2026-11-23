"""Run inside an export of the ORIGINAL (pre-refactor) tree and dump reference tensors.

Usage: python _orig_reference.py <orig_dir> <out.pt> model <name> | loss <name>
CUDA calls and the IPython import of the original code are neutralised so it runs on CPU.
"""
import importlib
import random
import sys
import types

import torch

orig_dir, out_path, kind, name = sys.argv[1:5]
sys.path.insert(0, orig_dir)
sys.modules["IPython"] = types.SimpleNamespace(embed=lambda: None)
_to = torch.Tensor.to
torch.Tensor.to = lambda self, *a, **k: self if a and a[0] == "cuda" else _to(self, *a, **k)

torch.Tensor.cuda = lambda self, *a, **k: self
KW = dict(nOut=512, encoder_type="ASP", sinc_stride=10)

if kind == "model":
    torch.manual_seed(0)
    m = importlib.import_module("models." + name).MainModel(**(KW if name == "RawNet3" else {})).eval()
    torch.manual_seed(1)
    x = torch.randn(2, 1, 48240) if name == "RawNet3" else torch.randn(2, 1, 300, 112)
    with torch.no_grad():
        y = m(x)
    torch.save(dict(state=m.state_dict(), x=x, y=y), out_path)
else:
    torch.manual_seed(0)
    random.seed(0)
    module = importlib.import_module("loss." + name)
    kw = dict(nOut=32, nClasses=6, margin=0.2, scale=30, hard_prob=0.5, hard_rank=1)
    metric = name in ("angleproto", "proto", "ge2e", "softmaxproto", "triplet")
    loss = module.LossFunction(**kw)
    torch.manual_seed(1)
    x = torch.randn(12, 2, 32) if metric else torch.randn(8, 32)
    y = torch.arange(12) % 6 if metric else torch.randint(0, 5, (8,))
    random.seed(2)
    nloss, prec = loss(x, y)
    torch.save(dict(state=loss.state_dict(), x=x, y=y, loss=nloss.detach(), prec=torch.as_tensor(prec).float()), out_path)
