import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLossAccuracy:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, dist, target, device=None):
        dist = dist.cuda(device)
        target = target.cuda(device)
        assert target.ndim == 1 and target.size() == dist.size()
        preds = (dist < self.threshold)
        accuracy = (preds == target).sum().item() / target.size(0)
        return accuracy, preds


def multi_class_accuracy(probs, target, device=None):
    probs = probs.cuda(device)
    target = target.cuda(device)
    assert target.ndim == 1 and target.size(0) == probs.size(0)
    preds = probs.argmax(dim=1)
    accuracy = (preds == target).sum().item() / target.size(0)
    return accuracy, preds
