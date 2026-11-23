"""AAM-Softmax (ArcFace). Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)."""
import torch
import torch.nn as nn

from speakernet.losses.common import aam_logits
from speakernet.utils import accuracy


class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.3, scale=15, easy_margin=False, **kwargs):
        super().__init__()
        self.test_normalize = True
        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(nClasses, nOut), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

    def forward(self, x, label=None):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        output = aam_logits(x, self.weight, label, self.m, self.s, self.easy_margin)
        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1
