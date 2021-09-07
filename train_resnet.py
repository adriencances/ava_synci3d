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

import argparse
import json

# from dataset import AvaPairs
from synci3d import SyncI3dResNet
from accuracy import two_class_simple_accuracy

sys.path.insert(0, "/home/adrien/Code/Friends")
from dataset_resnet import FriendsPairs


def train_epoch(dataloader_train, model, epoch, loss_fn, optimizer, accuracy_fn):
    nb_batches = 0
    train_loss = 0
    train_acc = 0

    nb_classes = 2
    nb_true_predictions_by_class = [0 for i in range(nb_classes)]
    total_by_class = [0 for i in range(nb_classes)]

    model.train()
    for batch_id, (segment1, segment2, frame, target) in enumerate(tqdm.tqdm(dataloader_train)):
        # Pass inputs to GPU
        segment1 = segment1.cuda()
        segment2 = segment2.cuda()
        frame = frame.cuda()
        target = target.cuda()

        # Pass inputs to the model
        out = model(segment1, segment2, frame) # shape : bx2

        # Compute loss
        loss = loss_fn(out.squeeze(), target.float())
        train_loss += loss.item()

        # Update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute probabilities and accuracy
        probs = F.softmax(out, dim=1)
        value, preds = accuracy_fn(out, target)
        train_acc += value

        # For accuracy by class
        for i in range(len(target)):
            k = target[i]
            total_by_class[k] += 1
            if preds[i] == target[i]:
                nb_true_predictions_by_class[k] += 1

        nb_batches += 1
    
    # Accuracy by class
    accuracies_by_class = [nb_true_predictions_by_class[k] / total_by_class[k] \
        for k in range(nb_classes)]
    train_mean_acc = sum(accuracies_by_class) / nb_classes

    train_loss /= nb_batches
    train_acc /= nb_batches

    return train_loss, train_acc, train_mean_acc, accuracies_by_class


def test_epoch(dataloader_val, model, epoch, loss_fn, optimizer, accuracy_fn):
    nb_batches = 0
    val_loss = 0
    val_acc = 0

    nb_classes = 2
    nb_true_predictions_by_class = [0 for i in range(nb_classes)]
    total_by_class = [0 for i in range(nb_classes)]

    with torch.no_grad():
        model.eval()
        for batch_id, (segment1, segment2, frame, target) in enumerate(tqdm.tqdm(dataloader_val)):
            # Pass inputs to GPU
            segment1 = segment1.cuda()
            segment2 = segment2.cuda()
            frame = frame.cuda()
            target = target.cuda()

            # Pass inputs to the model
            out = model(segment1, segment2, frame) # shape : bx2

            # Compute loss
            loss = loss_fn(out.squeeze(), target.float())
            val_loss += loss.item()

            # Compute probabilities and accuracy
            probs = F.softmax(out, dim=1)
            value, preds = accuracy_fn(out, target)
            val_acc += value

            # For accuracy by class
            for i in range(len(target)):
                k = target[i]
                total_by_class[k] += 1
                if preds[i] == target[i]:
                    nb_true_predictions_by_class[k] += 1

            nb_batches += 1
        
        # Accuracy by class
        accuracies_by_class = [nb_true_predictions_by_class[k] / total_by_class[k] \
            for k in range(nb_classes)]
        val_mean_acc = sum(accuracies_by_class) / nb_classes

        val_loss /= nb_batches
        val_acc /= nb_batches

    return val_loss, val_acc, val_mean_acc, accuracies_by_class


def load_checkpoint(model, optimizer, checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


def train_model(args):
    if args.option == "2":
        args.nb_layers = 2
        args.do_chkpts = True
        args.chkpt_delay = 1
    if args.option == "3":
        args.nb_layers = 1
        args.dropout_prob = 0

    config_id = abs(hash(json.dumps(vars(args), sort_keys=True)))
    if args.record:
        # File with list of configurations
        config_file = "runs_friends/configs_resnet.txt"
        mode = "a" if os.path.isfile(config_file) else "w"
        with open(config_file, mode) as f:
            f.write(str(config_id) + "\t" + str(args) + "\n")
        # Tensorboard writer
        writer = SummaryWriter("runs_friends/run_config_resnet_{}".format(config_id))

    # Model
    model = SyncI3dResNet(num_in_frames=16, nb_layers=args.nb_layers, dropout_prob=args.dropout_prob)
    model.cuda()

    # Loss function, optimizer
    if args.option in ["2", "3"]:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.000001
    )

    # Load checkpoint if checkpoint file given
    if args.chkpt_file is not None:
        model, optimizer, last_epoch = load_checkpoint(model, optimizer, args.chkpt_file)
        start_epoch = last_epoch + 1
    else:
        start_epoch = 0

    # Accuracy function
    accuracy_fn = two_class_simple_accuracy

    # Datasets
    if args.train_data_size is not None:
        train_data_size = args.train_data_size
        val_data_size = train_data_size // 4
        test_data_size = val_data_size
        nb_positives_train = train_data_size // 4
        nb_positives_val = val_data_size // 4
        nb_positives_test = test_data_size // 4
    else:
        nb_positives_train = None
        nb_positives_val = None
        nb_positives_test = None
    dataset_train = FriendsPairs("train", nb_positives=nb_positives_train, augmented=args.augmented)
    dataset_val = FriendsPairs("val", nb_positives=nb_positives_val)
    dataset_test = FriendsPairs("test", nb_positives=nb_positives_test)

    # Dataloaders
    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset=dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    labels = [
        "negative",
        "positive"
    ]

    train_losses = []
    train_accs = []
    train_mean_accs = []
    all_train_accs_by_class = []

    val_losses = []
    val_accs = []
    val_mean_accs = []
    all_val_accs_by_class = []

    test_losses = []
    test_accs = []
    test_mean_accs = []
    all_test_accs_by_class = []

    for epoch in range(start_epoch, start_epoch + args.epochs):
        print("Epoch {}/{}".format(epoch + 1 - start_epoch, args.epochs))

        # Train epoch
        train_loss, train_acc, train_mean_acc, train_accs_by_class \
            = train_epoch(dataloader_train, model, epoch, loss_fn, optimizer, accuracy_fn)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_mean_accs.append(train_mean_acc)
        all_train_accs_by_class.append(train_accs_by_class)

        # Val epoch
        val_loss, val_acc, val_mean_acc, val_accs_by_class \
            = test_epoch(dataloader_val, model, epoch, loss_fn, optimizer, accuracy_fn)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_mean_accs.append(val_mean_acc)
        all_val_accs_by_class.append(val_accs_by_class)

        if epoch % args.test_delay == 0:
            # Test epoch
            test_loss, test_acc, test_mean_acc, test_accs_by_class \
                = test_epoch(dataloader_test, model, epoch, loss_fn, optimizer, accuracy_fn)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            test_mean_accs.append(test_mean_acc)
            all_test_accs_by_class.append(test_accs_by_class)

        # print("train")
        # print("loss: \t", train_loss)
        # print("mean acc: \t", train_mean_acc)
        # print("class accs: \t", "\t".join(map(str, train_accs_by_class)))
        # print("val")
        # print("loss: \t", val_loss)
        # print("mean acc: \t", val_mean_acc)
        # print("class accs: \t", "\t".join(map(str, val_accs_by_class)))

        if args.record:
            # Write losses and accuracies to Tensorboard
            writer.add_scalar("training_loss", train_loss, global_step=epoch)
            writer.add_scalar("training_accuracy", train_acc, global_step=epoch)
            writer.add_scalars("training_class_accuracies", \
                dict([(label, train_accs_by_class[i]) for i, label in enumerate(labels)]), global_step=epoch)

            writer.add_scalar("validation_loss", val_loss, global_step=epoch)
            writer.add_scalar("validation_accuracy", val_acc, global_step=epoch)
            writer.add_scalars("validation_class_accuracies", \
                dict([(label, val_accs_by_class[i]) for i, label in enumerate(labels)]), global_step=epoch)
            
            if epoch % args.test_delay == 0:
                writer.add_scalar("test_loss", test_loss, global_step=epoch)
                writer.add_scalar("test_accuracy", test_acc, global_step=epoch)
                writer.add_scalars("test_class_accuracies", \
                    dict([(label, test_accs_by_class[i]) for i, label in enumerate(labels)]), global_step=epoch)
    
        # Save checkpoint
        if args.do_chkpts and epoch % args.chkpt_delay == args.chkpt_delay - 1:
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            chkpt_file = "checkpoints_friends/checkpoint_config_resnet_{}_epoch{}.pt".format(config_id, epoch)
            torch.save(state, chkpt_file)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int,
                        default=100,
                        help='number of epochs (default: 100)')
    parser.add_argument('-sz', '--train_data_size', type=int,
                        default=None,
                        help='number of pairs for the dataset')
    parser.add_argument('-aug', '--augmented', type=str2bool,
                        default=True,
                        help='use augmented data or not (default: True)')
    parser.add_argument('-td', '--test_delay', type=int,
                        default=3,
                        help='test delay in epochs (default: 5)')
    parser.add_argument('-bs', '--batch_size', type=int,
                        default=16,
                        help='batch size (default: 16)')
    parser.add_argument('-l', '--nb_layers', type=int,
                        default=2,
                        help='number of layers in the MLP of SyncI3d (default: 3)')
    parser.add_argument('-do', '--dropout_prob', type=float,
                        default=0,
                        help='dropout probability (default: 0)')
    parser.add_argument('-lr', '--learning_rate', type=float, dest='lr',
                        default=0.1,
                        help='learning rate (default: 0.1)')
    parser.add_argument('-rec', '--record', type=str2bool,
                        default=True,
                        help='record in TensorBoard or not (default: True)')
    parser.add_argument('-chp', '--do_chkpts', type=str2bool,
                        default=True,
                        help='do checkpoints or not (default: False)')
    parser.add_argument('-chpd', '--chkpt_delay', type=int,
                        default=5,
                        help='checkpoint delay in epochs (default: 10)')
    parser.add_argument('-chpf', '--chkpt_file', type=str,
                        default=None,
                        help='checkpoint file (default: None)')
    parser.add_argument('-opt', '--option', type=str,
                        default=1,
                        help='option for the model architecture (see Vicky\'s message')
    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    print(args)
    print(abs(hash(json.dumps(vars(args), sort_keys=True))))

    train_model(args)
