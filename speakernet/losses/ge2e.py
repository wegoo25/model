"""Generalised end-to-end (GE2E) loss. Expects embeddings of shape ``(batch, nPerSpeaker >= 2, dim)``."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from speakernet.utils import accuracy


class LossFunction(nn.Module):
    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
        super().__init__()
        self.test_normalize = True
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        assert x.size()[1] >= 2
        gsize = x.size()[1]
        centroids = torch.mean(x, 1)
        stepsize = x.size()[0]

        cos_sim_matrix = []
        for ii in range(gsize):
            idx = [*range(gsize)]
            idx.remove(ii)
            exc_centroids = torch.mean(x[:, idx, :], 1)
            cos_sim_diag = F.cosine_similarity(x[:, ii, :], exc_centroids)
            cos_sim = F.cosine_similarity(x[:, ii, :].unsqueeze(-1), centroids.unsqueeze(-1).transpose(0, 2))
            cos_sim[range(stepsize), range(stepsize)] = cos_sim_diag
            cos_sim_matrix.append(torch.clamp(cos_sim, 1e-6))

        cos_sim_matrix = torch.stack(cos_sim_matrix, dim=1)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b

        label = torch.repeat_interleave(torch.arange(stepsize, device=x.device), repeats=gsize, dim=0)
        logits = cos_sim_matrix.view(-1, stepsize)
        nloss = self.criterion(logits, label)
        prec1 = accuracy(logits.detach(), label.detach(), topk=(1,))[0]
        return nloss, prec1
