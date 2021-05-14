import torch
import torch.nn as nn
import torch.nn.functional as F


class Accuracy:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, dist, target, device=None):
        dist = dist.cuda(device)
        target = target.cuda(device)
        assert target.ndim == 1 and target.size() == dist.size()
        preds = (dist < self.threshold)
        accuracy = (preds == target).sum().item() / target.size(0)
        return accuracy, preds
