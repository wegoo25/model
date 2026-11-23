"""AM-Softmax (CosFace)."""
import torch
import torch.nn as nn

from speakernet.losses.common import am_logits
from speakernet.utils import accuracy


class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.3, scale=15, **kwargs):
        super().__init__()
        self.test_normalize = True
        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.W = nn.Parameter(torch.randn(nOut, nClasses), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, label=None):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        logits = am_logits(x, self.W, label, self.m, self.s)
        loss = self.ce(logits, label)
        prec1 = accuracy(logits.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1
