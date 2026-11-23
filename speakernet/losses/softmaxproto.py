"""Softmax + angular prototypical loss. Expects exactly two utterances per speaker."""
import torch.nn as nn

from speakernet.losses import angleproto, softmax


class LossFunction(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.test_normalize = True
        self.softmax = softmax.LossFunction(**kwargs)
        self.angleproto = angleproto.LossFunction(**kwargs)

    def forward(self, x, label=None):
        assert x.size()[1] == 2
        nloss_s, prec1 = self.softmax(x.reshape(-1, x.size()[-1]), label.repeat_interleave(2))
        nloss_p, _ = self.angleproto(x, None)
        return nloss_s + nloss_p, prec1
