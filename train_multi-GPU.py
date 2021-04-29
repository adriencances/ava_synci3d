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

import argparse



# Parser
parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--lr', default=0.01, help='')
parser.add_argument('--resume', default=None, help='')
parser.add_argument('--batch_size', type=int, default=8, help='')
parser.add_argument('--epochs', type=int, default=1, help='')
parser.add_argument('--num_workers', type=int, default=8, help='')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")

parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')
args = parser.parse_args()

gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices



class Accuracy:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, dist, target, device):
        dist = dist.cuda(device)
        target = target.cuda(device)

        assert target.ndim == 1 and target.size() == dist.size()
        preds = (dist < self.threshold)
        accuracy = (preds == target).sum().item() / target.size(0)
        return accuracy, preds


def main():
    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()

    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    ngpus_per_node = torch.cuda.device_count()    
    print("Use GPU: {} for training".format(args.gpu))

    args.rank = args.rank * ngpus_per_node + gpu    
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)


    # Parallelization
    torch.cuda.set_device(args.gpu)
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.num_workers = int(args.num_workers / ngpus_per_node)

    # Model
    model = SyncI3d(num_in_frames=16)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    # Parameters
    train_data_size = 1000
    lr = 0.01
    margin = 1.5
    threshold = 0.5

    # Datasets
    val_data_size = train_data_size // 4
    nb_positives_train = train_data_size // 4
    # nb_positives_val = val_data_size // 4

    dataset_train = AvaPairs("train", nb_positives=nb_positives_train)
    # dataset_val = AvaPairs("val", nb_positives=nb_positives_val)

    sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
    # sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=8,
        sampler=sampler_train
    )

    # dataloader_val = torch.utils.data.DataLoader(
    #     dataset_val,
    #     batch_size=args.batch_size,
    #     num_workers=8,
    #     sampler=sampler_val
    # )

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

    # # Making sure the dataset is the same for all processes
    # seg1, seg2, label = dataset_train[0]
    # print("GPU {} \t {} \t {} \t {} \t {} \t {}".format(args.gpu, seg1[0, 0, 112, 112], seg1[1, 0, 112, 112], seg2[0, 0, 112, 112], seg2[1, 0, 112, 112], label))
    # seg1, seg2, label = dataset_train[497]
    # print("GPU {} \t {} \t {} \t {} \t {} \t {}".format(args.gpu, seg1[0, 0, 112, 112], seg1[1, 0, 112, 112], seg2[0, 0, 112, 112], seg2[1, 0, 112, 112], label))

    # Loop through epochs
    for epoch in range(args.epochs):
        train_epoch(dataloader_train, model, epoch, loss_fn, optimizer, accuracy_fn, args.gpu)


def train_epoch(dataloader_train, model, epoch, loss_fn, optimizer, accuracy_fn, device):
    nb_batches = 0
    model.train()
    for batch_id, (segment1, segment2, target) in enumerate(tqdm.tqdm(dataloader_train)):
        # Pass inputs to GPU
        segment1 = segment1.cuda(device)
        segment2 = segment2.cuda(device)
        target = target.cuda(device)

        # Pass inputs to the model
        features1, features2 = model(segment1, segment2) # shape : bx1024

        # Normalize each feature vector (separately)
        features1_norm = F.normalize(features1, p=2, dim=1)
        features2_norm = F.normalize(features2, p=2, dim=1)

        # Compute loss and distances
        loss, dist = loss_fn(features1_norm, features2_norm, target)

        # Update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        nb_batches += 1



if __name__ == "__main__":
    main()



