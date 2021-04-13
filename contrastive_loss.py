import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = 1.5
        self.eps = 1e-9

    def forward(self, x1, x2, target, size_average=True):
        dist = F.pairwise_distance(x1, x2)
        loss = 0.5*torch.mean((1-target) * torch.pow(dist, 2) + 
                           ((target)) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        sigmoid = torch.nn.Sigmoid()
        sigm = sigmoid(dist)
        return loss, sigm
