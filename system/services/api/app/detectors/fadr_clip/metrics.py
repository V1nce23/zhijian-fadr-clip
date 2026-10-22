from collections import defaultdict

import numpy as np
import torch
from sklearn import metrics


def calculate_metrics_for_train(label, output):
    """Compute batch metrics used by the trainer progress bar."""
    if output.size(1) == 2:
        prob = torch.softmax(output, dim=1)[:, 1]
    else:
        prob = output

    _, prediction = torch.max(output, 1)
    correct = (prediction == label).sum().item()
    accuracy = correct / prediction.size(0)

    y_true = label.detach().cpu().numpy()
    y_pred = prob.detach().cpu().numpy()
    ap = metrics.average_precision_score(y_true, y_pred)

    try:
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
    except ValueError:
        return None, None, accuracy, ap

    if np.isnan(fpr[0]) or np.isnan(tpr[0]):
        auc, eer = None, None
    else:
        auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

    return auc, eer, accuracy, ap


class Recorder:
    """Track running means for scalar values."""

    def __init__(self):
        self.sum = 0.0
        self.num = 0

    def update(self, item, num=1):
        if item is None:
            return
        self.sum += float(item) * num
        self.num += num

    def average(self):
        if self.num == 0:
            return None
        return self.sum / self.num

    def clear(self):
        self.sum = 0.0
        self.num = 0
