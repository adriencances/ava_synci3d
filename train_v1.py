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

from dataset import AvaPairs
from synci3d_v1 import SyncI3d_v1
from accuracy import multi_class_accuracy


def train_epoch(dataloader_train, model, epoch, loss_fn, optimizer, accuracy_fn):
    nb_batches = 0
    train_loss = 0
    train_acc = 0

    model.train()
    for batch_id, (segment1, segment2, target) in enumerate(tqdm.tqdm(dataloader_train)):
        # Pass inputs to GPU
        segment1 = segment1.cuda()
        segment2 = segment2.cuda()
        target = target.cuda()

        # Pass inputs to the model
        out = model(segment1, segment2) # shape : bx2

        # Compute loss
        loss = loss_fn(out, target)
        train_loss += loss.item()

        # Update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute probabilities and accuracy
        probs = F.softmax(out, dim=1)
        value, preds = accuracy_fn(probs, target)
        train_acc += value

        nb_batches += 1

    train_loss /= nb_batches
    train_acc /= nb_batches

    return train_loss, train_acc


def test_epoch(dataloader_val, model, epoch, loss_fn, optimizer, accuracy_fn):
    nb_batches = 0
    val_loss = 0
    val_acc = 0

    with torch.no_grad():
        model.eval()
        for batch_id, (segment1, segment2, target) in enumerate(tqdm.tqdm(dataloader_val)):
            # Pass inputs to GPU
            segment1 = segment1.cuda()
            segment2 = segment2.cuda()
            target = target.cuda()

            # Pass inputs to the model
            out = model(segment1, segment2) # shape : bx2

            # Compute loss
            loss = loss_fn(out, target)
            val_loss += loss.item()

            # Compute probabilities and accuracy
            probs = F.softmax(out, dim=1)
            value, preds = accuracy_fn(probs, target)
            val_acc += value

            nb_batches += 1

        val_loss /= nb_batches
        val_acc /= nb_batches

    return val_loss, val_acc


def load_checkpoint(model, optimizer, checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


def train_model(epochs, train_data_size, batch_size, lr=0.01, record=True, chkpt_delay=10, chkpt_file=None):
    if record:
        # Tensorboard writer
        writer = SummaryWriter("runs_v1/run_size{}_1".format(train_data_size))

    # Model
    model = SyncI3d_v1(num_in_frames=16)
    model.cuda()

    # Loss function, optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=0.000001
    )

    # Load checkpoint if checkpoint file given
    if chkpt_file is not None:
        model, optimizer, last_epoch = load_checkpoint(model, optimizer, chkpt_file)
        start_epoch = last_epoch + 1
    else:
        start_epoch = 0

    # Accuracy function
    accuracy_fn = multi_class_accuracy

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

    train_losses = []
    train_accs = []

    val_losses = []
    val_accs = []

    for epoch in range(start_epoch, start_epoch + epochs):
        print("Epoch {}/{}".format(epoch + 1 - start_epoch, epochs))

        # Train epoch
        train_loss, train_acc = train_epoch(dataloader_train, model, epoch, loss_fn, optimizer, accuracy_fn)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Test epoch
        val_loss, val_acc = test_epoch(dataloader_val, model, epoch, loss_fn, optimizer, accuracy_fn)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if record:
            # Write losses and accuracies to Tensorboard
            writer.add_scalar("training_loss", train_loss, global_step=epoch)
            writer.add_scalar("training_accuracy", train_acc, global_step=epoch)
            writer.add_scalar("validation_loss", val_loss, global_step=epoch)
            writer.add_scalar("validation_accuracy", val_acc, global_step=epoch)

            # Save checkpoint
            if epoch%chkpt_delay == chkpt_delay - 1:
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                chkpt_file = "checkpoints_v1/checkpoint_size{}_lr{}_epoch{}.pt".format(train_data_size, lr, epoch)
                torch.save(state, chkpt_file)


if __name__ == "__main__":
    epochs = int(sys.argv[1])
    lr = float(sys.argv[2])
    train_data_size = int(sys.argv[3])
    chkpt_file = None if len(sys.argv) < 5 else sys.argv[4]

    batch_size = 8
    record = True
    chkpt_delay = 5

    print("Nb epochs : {}".format(epochs))
    print("Learning rate : {}".format(lr))
    print("Train data size : {}".format(train_data_size))
    if chkpt_file is not None:
        print("Checkpoint: {}".format(chkpt_file.split("/")[-1]))
    train_model(epochs=epochs, train_data_size=train_data_size, batch_size=batch_size, lr=lr, record=record,
        chkpt_delay=chkpt_delay, chkpt_file=chkpt_file)
