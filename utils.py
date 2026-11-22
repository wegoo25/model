#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F \



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert len(input.size()) == 2, 'The number of dimensions of input tensor must be 2!'
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

def plot_det_curve(fnr, fpr, save_path):

    plt.figure()
    plt.plot(fpr, fnr, label="DET Curve", color='red', linestyle='dashed')
    plt.plot([0, 1], [0, 1], 'r--', label="EER Line (y=x)")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("False Negative Rate (FNR)")
    plt.title("DET Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    #plt.show()


def plot_multiple_det_curves(fnr_list, fpr_list, labels, save_path):
    plt.figure()
    for fnr, fpr, label in zip(fnr_list, fpr_list, labels):
        plt.plot(fpr, fnr, label=label)
    plt.plot([0, 1], [0, 1], 'r--', label="EER Line (y=x)")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("False Negative Rate (FNR)")
    plt.title("DET Curve")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 0.4)  # Set x-axis limit
    plt.ylim(0, 0.4)  # Set y-axis limit
    plt.savefig(save_path)
    # plt.show()

if __name__ == "__main__":
    # Define the file name
    file_names = ["TDNN.txt", "BiMKRC.txt", "AMCRN.txt"]  # Replace with your actual file names
    labels = ["TDNN", "BiMKRC", "AMCRN"]  # Labels for the curves

    # Initialize lists to store all fnrs and fprs
    all_fprs = []
    all_fnrs = []

    # Read each file and extract values
    for file_name in file_names:
        fprs = []
        fnrs = []
        with open(file_name, "r") as file:
            for line in file:
                if line.startswith("fprs:"):
                    fprs = list(map(float, line.split("fprs:")[1].strip().split(", ")))
                elif line.startswith("fnrs:"):
                    fnrs = list(map(float, line.split("fnrs:")[1].strip().split(", ")))
        all_fprs.append(fprs)
        all_fnrs.append(fnrs)

    # Plot the DET curves
    save_path = "./det_curves/Combined_DET.png"
    plot_multiple_det_curves(all_fnrs, all_fprs, labels, save_path)
