"""Plain softmax classification head."""
import torch.nn as nn

from speakernet.utils import accuracy


class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, **kwargs):
        super().__init__()
        self.test_normalize = True
        self.criterion = nn.CrossEntropyLoss()
        self.fc = nn.Linear(nOut, nClasses)

    def forward(self, x, label=None):
        x = self.fc(x)
        nloss = self.criterion(x, label)
        prec1 = accuracy(x.detach(), label.detach(), topk=(1,))[0]
        return nloss, prec1
