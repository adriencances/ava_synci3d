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

import matplotlib.pyplot as plt


def evaluate(model, dataloader, loss_fn, accuracy_fn):
    nb_positive_preds = 0
    nb_negative_preds = 0
    probs = []
    with torch.no_grad():
        for batch_id, (segment1, segment2, target) in enumerate(tqdm.tqdm(dataloader)):
            # Pass inputs to GPU
            segment1 = segment1.cuda()
            segment2 = segment2.cuda()
            target = target.cuda()

            # Pass inputs to the model
            features1, features2 = model(segment1, segment2) # shape : bx1024

            # Normalize each feature vector (separately)
            features1_norm = F.normalize(features1, p=2, dim=1)
            features2_norm = F.normalize(features2, p=2, dim=1)

            # Compute loss and distances
            loss, dist = loss_fn(features1_norm, features2_norm, target)
            
            # Compute accuracy and predictions
            acc, preds = accuracy_fn(dist, target)

            nb_positive_preds += preds.sum()
            nb_negative_preds += len(preds) - nb_positive_preds

            probs += sigm.tolist()

    print("Nb pairs: \t{}".format(len(probs)))
    print("Min prob: \t{}".format(min(probs)))
    print("Mean prob: \t{}".format(np.mean(probs)))
    print("Max prob: \t{}".format(max(probs)))
    print("Positive preds: \t{}".format(nb_positive_preds))
    print("Negative preds: \t{}".format(nb_negative_preds))


def plot_distance_distribution(distances, figure_file):
    # Plot and save histogram
    plt.clf()
    plt.hist(distances, density=True, bins=30)
    plt.ylabel("Probability")
    plt.xlabel("Distance between feature vectors")
    plt.savefig(figure_file)


def compute_and_plot_distance_distribution(model, dataloader, figure_file):
    distances = []
    with torch.no_grad():
        for batch_id, (segment1, segment2, target) in enumerate(tqdm.tqdm(dataloader)):
            # Pass inputs to GPU
            segment1 = segment1.cuda()
            segment2 = segment2.cuda()
            target = target.cuda()

            # Pass inputs to the model
            features1, features2 = model(segment1, segment2) # shape : bx1024

            # Normalize each feature vector (separately)
            features1_norm = F.normalize(features1, p=2, dim=1)
            features2_norm = F.normalize(features2, p=2, dim=1)

            # Compute distances
            dist = F.pairwise_distance(features1_norm, features2_norm)
            distances += dist.tolist()

    # Plot and save histogram
    plot_distance_distribution(distances, figure_file)


def read_summary_file(file):
    epochs = []
    losses = []
    accs = []
    with open(file, "r") as f:
        for line in f:
            entries = line.strip().split(",")
            epoch = int(entries[0])
            loss = float(entries[1])
            acc = float(entries[2])
            epochs.append(epoch)
            losses.append(loss)
            accs.append(acc)
    epochs = np.array(epochs)
    losses = np.array(losses)
    accs = np.array(accs)

    return epochs, losses, accs


def plot_loss_and_accuracy_graph(summary_file, figure_file, nb_epochs):
    epochs, losses, accs = read_summary_file(summary_file)

    plt.clf()
    plt.plot(epochs[:nb_epochs], losses[:nb_epochs])
    plt.plot(epochs[:nb_epochs], accs[:nb_epochs])
    plt.legend(["training loss", "training accuracy"])
    plt.xlabel("epoch")
    plt.savefig(figure_file)
