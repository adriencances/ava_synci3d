import os
import sys
import numpy as np
import cv2
import glob
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from dataset_aux import FrameProcessor
from dataset import AvaPairs
from synci3d import SyncI3d
from train import train_epoch
from contrastive_loss import ContrastiveLoss


def BinaryClassificationAccuracy(y_prob, y_true):
    y_prob = y_prob.cuda()
    y_true = y_true.cuda()
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)


def main():
    dataset = AvaPairs("train")
    dataloader_train = data.DataLoader(dataset, batch_size= 10, shuffle=True)

    model = SyncI3d()
    model.cuda()

    loss_fn = ContrastiveLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.000001)
    lr_sched = optim.lr_scheduler.StepLR(optimizer, 10)

    cuda = torch.cuda.is_available()

    train_epoch(dataloader_train, model, 0, loss_fn, optimizer, lr_sched, cuda, log_interval=0, metric=BinaryClassificationAccuracy)


if __name__ == "__main__":
    main()

