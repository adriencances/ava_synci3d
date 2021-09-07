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


def two_class_simple_accuracy(out, target, device=None):
    out = out.cuda(device)
    target = target.cuda(device)
    assert target.ndim == 1 and target.size(0) == out.size(0)
    preds = out > 0
    accuracy = (preds.squeeze() == target).sum().item() / target.size(0)
    return accuracy, preds


class CosineLossAccuracy:
    def __init__(self, pred_margin, eps=1e-08):
        self.pred_margin = pred_margin
        self.eps = eps
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=self.eps)

    def __call__(self, out, target, device=None):
        x1, x2 = out
        x1 = x1.cuda(device)
        x2 = x2.cuda(device)

        sims = self.cosine_similarity(x1, x2)

        preds = (sims > self.pred_margin)
        accuracy = (preds.squeeze() == target).sum().item() / target.size(0)
        return accuracy, preds
