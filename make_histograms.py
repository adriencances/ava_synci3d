import os
import sys
import numpy as np
import cv2
import glob
import tqdm
import pickle
from pathlib import Path

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
from contrastive_loss import ContrastiveLoss
from accuracy import Accuracy

import matplotlib.pyplot as plt


def do_epoch(dataloader, model, loss_fn, accuracy_fn):
    nb_batches = 0
    loss = 0
    acc = 0
    pos_distances = []
    neg_distances = []

    with torch.no_grad():
        model.eval()
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
            loss += loss.item()

            # Compute accuracy and predictions
            value, preds = accuracy_fn(dist, target)
            acc += value

            # Compute distances
            dist = dist.tolist()
            for i in range(len(dist)):
                if target[i] == 1:
                    pos_distances.append(dist[i])
                else:
                    neg_distances.append(dist[i])

            nb_batches += 1

        loss /= nb_batches
        acc /= nb_batches
    
    return loss, acc, pos_distances, neg_distances


def print_pair(segment1, segment2, index, high=True):
    output_dir = "/home/acances/Code/human_interaction_SyncI3d/interesting_positive_pairs"
    output_subdir = "{}/{}/pair_{}".format(output_dir, "high_distance" if high else "low_distance", index)

    Path(output_subdir).mkdir(parents=True, exist_ok=True)

    for i in range(segment1.shape[1]):
        filename1 = "{}/tensor1_frame_{}.jpg".format(output_subdir, i + 1)
        frame1 = segment1[:,i,:,:].cpu().numpy().transpose(2, 1, 0)
        cv2.imwrite(filename1, frame1)

        filename2 = "{}/tensor2_frame_{}.jpg".format(output_subdir, i + 1)
        frame2 = segment2[:,i,:,:].cpu().numpy().transpose(2, 1, 0)
        cv2.imwrite(filename2, frame2)


def do_epoch_and_print_out_interesting_pairs(dataloader, model, loss_fn, accuracy_fn, nb_high_dist_pos, nb_low_dist_pos):
    nb_batches = 0
    loss = 0
    acc = 0
    pos_distances = []
    neg_distances = []

    cnt_high_dist_pos = 0
    cnt_low_dist_pos = 0
    high_threshold = 1.2
    low_threshold = 0.2

    with torch.no_grad():
        model.eval()
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
            loss += loss.item()

            # Compute accuracy and predictions
            value, preds = accuracy_fn(dist, target)
            acc += value

            # Compute distances
            dist = dist.tolist()
            for i in range(len(dist)):
                if target[i] == 1:
                    pos_distances.append(dist[i])
                else:
                    neg_distances.append(dist[i])
                
                # Print out pair if interesting
                if target[i] == 1:
                    if cnt_high_dist_pos < nb_high_dist_pos and dist[i] > high_threshold:
                        cnt_high_dist_pos += 1
                        print_pair(segment1[i], segment2[i], index=cnt_high_dist_pos, high=True)
                    if cnt_low_dist_pos < nb_low_dist_pos and dist[i] < low_threshold:
                        cnt_low_dist_pos += 1
                        print_pair(segment1[i], segment2[i], index=cnt_low_dist_pos, high=False)

            nb_batches += 1

        loss /= nb_batches
        acc /= nb_batches
    
    return loss, acc, pos_distances, neg_distances


def plot_histogram(pos_distances, neg_distances, margin, figure_file):
    plt.clf()
    _, bins, _ = plt.hist(pos_distances, bins=50, alpha=0.5, range=[0, margin], label="positive pairs")
    _ = plt.hist(neg_distances, bins=bins, alpha=0.5, label="negative pairs")
    plt.ylabel("Probability")
    plt.xlabel("Distance between feature vectors")
    plt.savefig(figure_file)


def make_histogram(checkpoint_file, phase, data_size, figure_file):
    # Parameters
    margin = 1.5
    threshold = 0.5

    # Dataset
    nb_positives = data_size //4
    dataset = AvaPairs(phase=phase, nb_positives=nb_positives)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # Model
    model = SyncI3d(num_in_frames=16)
    model.cuda()

    # Loss function, optimizer
    loss_fn = ContrastiveLoss(margin=margin)

    # Accuracy function
    accuracy_fn = Accuracy(threshold=threshold)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["model"])

    # Get loss, accuracy and distances for positive and negative pairs
    loss, acc, pos_distances, neg_distances = do_epoch_and_print_out_interesting_pairs(
        dataloader, model, loss_fn, accuracy_fn,
        nb_high_dist_pos=100, nb_low_dist_pos=100)
    print("Loss: \t\t{}".format(loss))
    print("Accuracy: \t{}".format(acc))

    # Save distances
    with open("histograms/pos_neg_distances.pkl", "wb") as f:
        pickle.dump((pos_distances, neg_distances), f)

    # Plot and save histogram
    plot_histogram(pos_distances, neg_distances, margin, figure_file)


if __name__ == "__main__":
    checkpoint_file = sys.argv[1]
    phase = sys.argv[2]
    data_size = int(sys.argv[3])

    epoch = int(checkpoint_file.split("/")[-1].split(".")[-2].split("_")[-1][5:])
    figure_file = "histograms/hist_size{}_{}_epoch{}.png".format(data_size, phase, epoch)

    make_histogram(checkpoint_file, phase, data_size, figure_file)
    
