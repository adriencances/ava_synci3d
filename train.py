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


def train_epoch(train_loader, model, epoch, loss_fn, optimizer, lr_sched, cuda, log_interval, metric):
    model.train()

    nb_batches = 0
    total_loss = 0 
    total_acc = []

    for batch_idx, (segment1, segment2, target) in enumerate(train_loader):
        # Pass inputs to GPU
        segment1 = segment1.cuda()
        segment2 = segment2.cuda()

        # Pass inputs to the model
        features1, features2 = model(segment1, segment2) # shape : bx1024

        # Normalize each feature vector (separately)
        features1_norm = F.normalize(v1, p=2, dim=1)
        features2_norm = F.normalize(v2, p=2, dim=1)

        # Compute loss
        loss, sigm = loss_fn(v1_norm, v2_norm, target.cuda())
        total_loss += loss.item()

        # Update weights
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        lr_sched.step()

        # Compute accuracy
        value = metric(sigm, target)
        total_acc.append(value)

        nb_batches += 1

    total_loss /= nb_batches
    acc = sum(total_acc)/len(total_acc)
    return total_loss, acc
