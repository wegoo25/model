"""Triplet loss with (semi-)hard negative mining. Expects exactly two utterances per speaker."""
import random

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from speakernet.metrics import tune_threshold_from_score


class LossFunction(nn.Module):
    def __init__(self, hard_rank=0, hard_prob=0, margin=0, **kwargs):
        super().__init__()
        self.test_normalize = True
        self.hard_rank = hard_rank
        self.hard_prob = hard_prob
        self.margin = margin

    def forward(self, x, label=None):
        assert x.size()[1] == 2
        out_anchor = F.normalize(x[:, 0, :], p=2, dim=1)
        out_positive = F.normalize(x[:, 1, :], p=2, dim=1)

        # (N, N) matrix of negative squared distances between every anchor and every positive
        diff = out_anchor.unsqueeze(1) - out_positive.unsqueeze(0) + 1e-6
        output = -diff.pow(2).sum(-1)
        negidx = self.mine_hard_negative(output.detach())
        out_negative = out_positive[negidx, :]

        labelnp = numpy.array([1] * len(out_positive) + [0] * len(out_negative))

        pos_dist = F.pairwise_distance(out_anchor, out_positive)
        neg_dist = F.pairwise_distance(out_anchor, out_negative)
        nloss = torch.mean(F.relu(torch.pow(pos_dist, 2) - torch.pow(neg_dist, 2) + self.margin))

        scores = -1 * torch.cat([pos_dist, neg_dist], dim=0).detach().cpu().numpy()
        errors = tune_threshold_from_score(scores, labelnp, [])
        return nloss, errors[1]

    def mine_hard_negative(self, output):
        negidx = []
        for idx, similarity in enumerate(output):
            simval, simidx = torch.sort(similarity, descending=True)
            if self.hard_rank < 0:
                semihardidx = simidx[(similarity[idx] - self.margin < simval) & (simval < similarity[idx])]
                if len(semihardidx) == 0:
                    negidx.append(random.choice(simidx))
                else:
                    negidx.append(random.choice(semihardidx))
            else:
                simidx = simidx[simidx != idx]
                if random.random() < self.hard_prob:
                    negidx.append(simidx[random.randint(0, self.hard_rank)])
                else:
                    negidx.append(random.choice(simidx))
        return negidx
