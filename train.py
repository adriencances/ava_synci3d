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

from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset_aux import FrameProcessor
from dataset import AvaPairs
from synci3d import SyncI3d
from contrastive_loss import ContrastiveLoss


def BinaryClassificationAccuracy(y_prob, y_true):
    y_prob = y_prob.cuda()
    y_true = y_true.cuda()
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)


def train(epochs, batch_size):

    # writer = SummaryWriter()

    # Model
    model = SyncI3d()
    model.cuda()

    # Loss function, optimizer, learning rate schedule
    loss_fn = ContrastiveLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.000001)

    # Dataset
    dataset = AvaPairs("train")

    # Data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    for epoch in range(epochs):
        print("EPOCH {}".format(epoch))
        nb_batches = 0
        total_loss = 0
        total_acc = []
        for batch_id, (segment1, segment2, target) in enumerate(tqdm.tqdm(train_loader)):
            # Pass inputs to GPU
            segment1 = segment1.cuda()
            segment2 = segment2.cuda()

            # Pass inputs to the model
            features1, features2 = model(segment1, segment2) # shape : bx1024

            # Normalize each feature vector (separately)
            features1_norm = F.normalize(features1, p=2, dim=1)
            features2_norm = F.normalize(features2, p=2, dim=1)

            # Compute loss
            loss, sigm = loss_fn(features1_norm, features2_norm, target.cuda())
            total_loss += loss.item()

            # Update weights
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            # Compute accuracy
            value = BinaryClassificationAccuracy(sigm, target)
            total_acc.append(value)

            nb_batches += 1

        total_loss /= nb_batches
        acc = sum(total_acc)/len(total_acc)
        # return total_loss, acc
