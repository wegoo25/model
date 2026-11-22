import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from tuneThreshold import ComputeErrorRates

model_paths = {
    "K = 2": "exps/MKRC4_AAM/model/scores.npy",
    "K = 3": "exps/MKRC5_AAM/model/scores.npy",
    "K = 4": "exps/MKRC6_AAM/model/scores.npy"
}

plt.figure(figsize=(6,6))

# Loop through models and plot DET for each
for model_name, score_path in model_paths.items():
    scores = np.load(score_path)
    labels = np.load(score_path.replace("scores.npy", "labels.npy"))  # Load corresponding labels

    fnrs, fprs, _ = ComputeErrorRates(scores, labels)

    plt.plot(fprs * 100, fnrs * 100, label=f"{model_name}", linewidth=0.2)  # Convert to percentages

# Configure plot
plt.xscale("log")
plt.yscale("log")
plt.xlabel("False Alarm Rate (%)")
plt.ylabel("Miss Detection Rate (%)")
plt.title("Detection Error Tradeoff (DET) Curve - Model Comparison")
plt.grid()
plt.legend()

plt.savefig("det_curve.png")
