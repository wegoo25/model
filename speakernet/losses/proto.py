"""Prototypical loss. Expects embeddings of shape ``(batch, nPerSpeaker >= 2, dim)``."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from speakernet.utils import accuracy


class LossFunction(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.test_normalize = False
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        assert x.size()[1] >= 2
        out_anchor = torch.mean(x[:, 1:, :], 1)
        out_positive = x[:, 0, :]
        stepsize = out_anchor.size()[0]

        output = -1 * (
            F.pairwise_distance(out_positive.unsqueeze(-1), out_anchor.unsqueeze(-1).transpose(0, 2)) ** 2
        )
        label = torch.arange(stepsize, device=x.device)
        nloss = self.criterion(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return nloss, prec1
