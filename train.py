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

from utils import plot_distance_distribution


class Accuracy:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, dist, target):
        dist = dist.cuda()
        target = target.cuda()
        assert target.ndim == 1 and target.size() == dist.size()
        preds = (dist < self.threshold)
        accuracy = (preds == target).sum().item() / target.size(0)
        return accuracy, preds


def train_epoch(dataloader_train, model, epoch, loss_fn, optimizer, accuracy_fn):
    nb_batches = 0
    train_loss = 0
    train_acc = 0
    train_distances = []

    model.train()
    for batch_id, (segment1, segment2, target) in enumerate(tqdm.tqdm(dataloader_train)):
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
        train_loss += loss.item()

        # Update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy and predictions
        value, preds = accuracy_fn(dist, target)
        train_acc += value

        # Compute distances
        train_distances += dist.tolist()

        nb_batches += 1

    train_loss /= nb_batches
    train_acc /= nb_batches

    return train_loss, train_acc, train_distances


def test_epoch(dataloader_val, model, epoch, loss_fn, optimizer, accuracy_fn):
    nb_batches = 0
    val_loss = 0
    val_acc = 0
    val_distances = []

    with torch.no_grad():
        model.eval()
        for batch_id, (segment1, segment2, target) in enumerate(tqdm.tqdm(dataloader_val)):
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
            val_loss += loss.item()

            # Compute accuracy and predictions
            value, preds = accuracy_fn(dist, target)
            val_acc += value

            # Compute distances
            val_distances += dist.tolist()

            nb_batches += 1

        val_loss /= nb_batches
        val_acc /= nb_batches

    return val_loss, val_acc, val_distances


def train_model(epochs, train_data_size, batch_size, lr=0.01, margin=1.5, threshold=0.5):
    summary_file = "summaries/summary_lr{}_marg{}_thre{}_epochs{}.csv".format(lr, margin, threshold, epochs)

    # Tensorboard writer
    writer = SummaryWriter("essai")

    # Model
    model = SyncI3d(num_in_frames=16)
    model.cuda()

    # Loss function, optimizer
    loss_fn = ContrastiveLoss(margin=margin)
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=0.000001
    )

    # Accuracy function
    accuracy_fn = Accuracy(threshold)

    # Datasets
    val_data_size = train_data_size // 4
    nb_positives_train = train_data_size // 4
    nb_positives_val = val_data_size // 4
    dataset_train = AvaPairs("train", nb_positives=nb_positives_train)
    dataset_val = AvaPairs("val", nb_positives=nb_positives_val)

    # Dataloaders
    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset=dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # # Initialize summary file
    # open(summary_file, "w").close()

    train_losses = []
    train_accs = []

    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))

        # Train epoch
        train_loss, train_acc, train_distances = train_epoch(dataloader_train, model, epoch, loss_fn, optimizer, accuracy_fn)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Test epoch
        val_loss, val_acc, val_distances = test_epoch(dataloader_val, model, epoch, loss_fn, optimizer, accuracy_fn)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Write losses and accuracies to Tensorboard
        writer.add_scalar("training_loss", train_loss, global_step=epoch)
        writer.add_scalar("training_accuracy", train_acc, global_step=epoch)
        writer.add_scalar("validation_loss", val_loss, global_step=epoch)
        writer.add_scalar("validation_accuracy", val_acc, global_step=epoch)

        # # Write info to summary file
        # with open(summary_file, "a") as f:
        #     f.write(",".join(map(str, [epoch, train_loss, train_acc, val_loss, val_acc])) + "\n")

        # # Save weights
        # weights_file = "synci3d_weights/weights_lr{}_marg{}_epoch{}.pt".format(lr, margin, epoch)
        # torch.save(model.state_dict(), weights_file)

        # # Plot and save distance histogram
        # train_figure_file = "distance_histograms/hist_train_lr{}_marg{}_epoch{}.png".format(lr, margin, epoch)
        # plot_distance_distribution(train_distances, train_figure_file)
        # val_figure_file = "distance_histograms/hist_val_lr{}_marg{}_epoch{}.png".format(lr, margin, epoch)
        # plot_distance_distribution(val_distances, val_figure_file)


def train(epochs, batch_size, lr=0.01, margin=1.5, threshold=0.5, record=True):

    # writer = SummaryWriter()

    # Model
    model = SyncI3d(num_in_frames=16)
    model.cuda()

    # Loss function, optimizer
    loss_fn = ContrastiveLoss(margin=margin)
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=0.000001
    )

    # Accuracy function
    accuracy_fn = Accuracy(threshold)

    # Dataset
    dataset = AvaPairs("train", nb_positives=100)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # Summary file to print out loss and accuracy by epoch
    if record:
        summary_file = "summaries/summary_lr{}.csv".format(lr)
        open(summary_file, "w").close()

    for epoch in range(epochs):
        nb_batches = 0
        total_loss = 0
        total_acc = []
        distances = []
        for batch_id, (segment1, segment2, target) in enumerate(tqdm.tqdm(train_loader)):
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
            total_loss += loss.item()

            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute accuracy and predictions
            value, preds = accuracy_fn(dist, target)
            total_acc.append(value)

            # Compute distances
            distances += dist.tolist()

            nb_batches += 1

        total_loss /= nb_batches
        acc = sum(total_acc)/len(total_acc)
        mean_dist = np.mean(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        print("Epoch {} \t Loss \t {} \t Acc \t {} \t Dist min {} mean {} max {}".format(
            epoch, total_loss, acc, min_dist, mean_dist, max_dist))

        if record:
            # Write info to summary file
            with open(summary_file, "a") as f:
                f.write(",".join(map(str, [epoch, total_loss, acc])) + "\n")

            # Save weights
            weights_file = "synci3d_weights/weights_lr{}_epoch{}.pt".format(lr, epoch)
            torch.save(model.state_dict(), weights_file)

            # Plot and save distance histogram
            figure_file = "distance_histograms/hist_lr{}_epoch{}.png".format(lr, epoch)
            plot_distance_distribution(distances, figure_file)


if __name__ == "__main__":
    epochs = int(sys.argv[1])
    lr = float(sys.argv[2])
    print("Nb epochs : {}".format(epochs))
    print("Learning rate : {}".format(lr))
    train_model(epochs=epochs, train_data_size=400, batch_size=16, lr=lr)
