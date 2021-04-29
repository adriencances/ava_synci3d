import os
import sys
import numpy as np
import cv2
import glob
import tqdm
import pickle

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
    model.train()
    for batch_id, (segment1, segment2, target) in enumerate(tqdm.tqdm(dataloader_train)):
        # Pass inputs to GPU
        segment1 = segment1.cuda()
        segment2 = segment2.cuda()
        target = target.cuda()

        # # Pass inputs to the model
        # features1, features2 = model(segment1, segment2) # shape : bx1024

        # # Normalize each feature vector (separately)
        # features1_norm = F.normalize(features1, p=2, dim=1)
        # features2_norm = F.normalize(features2, p=2, dim=1)

        # # Compute loss and distances
        # loss, dist = loss_fn(features1_norm, features2_norm, target)

        # # Update weights
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # # Compute accuracy and predictions
        # value, preds = accuracy_fn(dist, target)


class DummyDataset(data.Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.build_data()
    
    def build_data(self):
        seg1 = torch.rand(3, 16, 224, 224)
        seg2 = torch.rand(3, 16, 224, 224)
        lab = np.random.randint(2)
        self.pair = (seg1, seg2, lab)

    def __getitem__(self, index):
        "Generates one sample of data"
        return self.pair

    def __len__(self):
        """Denotes the total number of samples"""
        return self.size


class LoadSamePairDataset(data.Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.pickle_file = "pairs_tensors/pair16_0.pkl"

    def __getitem__(self, index):
        "Generates one sample of data"
        with open(self.pickle_file, "rb") as f:
            seg1, seg2, lab = pickle.load(f)
        return seg1, seg2, lab

    def __len__(self):
        """Denotes the total number of samples"""
        return self.size


class ComputeSamePairDataset(data.Dataset):
    def __init__(self, phase="train", size=1000):
        self.size = size

        self.w = 224
        self.h = 224
        self.alpha = 0.1

        self.phase = phase
        self.frames_dir = "/media/hdd/adrien/Ava_v2.2/correct_frames"
        self.shots_dir = "/home/acances/Data/Ava_v2.2/final_shots"
        self.tracks_dir = "/home/acances/Data/Ava_v2.2/tracks"
        self.pairs_dir = "/home/acances/Data/Ava_v2.2/pairs16"

        self.frame_processor = FrameProcessor(self.w, self.h, self.alpha, self.phase, self.frames_dir, self.shots_dir, self.tracks_dir)

        self.gather_single_pair()
        self.create_data()

    def gather_single_pair(self):
        print("Gathering a single pair")
        self.positive_pairs = []
        pairs_file = glob.glob("{}/{}/positive/*".format(self.pairs_dir, self.phase))[0]
        with open(pairs_file, "r") as f:
            pair = f.readline().strip().split(",")
            self.pair = pair + [1]
    
    def create_data(self):
        self.data = [self.pair for i in range(self.size)]

    def __getitem__(self, index):
        "Generates one sample of data"
        nb_pairs = len(self.data)
        assert index < nb_pairs
        pair = self.data[index]

        video_id1, shot_id1, i1, begin1, end1, video_id2, shot_id2, i2, begin2, end2, label = pair
        shot_id1, track_id1, begin1, end1 = list(map(int, [shot_id1, i1, begin1, end1]))
        shot_id2, track_id2, begin2, end2 = list(map(int, [shot_id2, i2, begin2, end2]))
        assert end1 - begin1 == end2 - begin2

        tensor1 = self.frame_processor.processed_frames(video_id1, shot_id1, track_id1, begin1, end1)
        tensor2 = self.frame_processor.processed_frames(video_id2, shot_id2, track_id2, begin2, end2)
        
        return tensor1, tensor2, label

    def __len__(self):
        """Denotes the total number of samples"""
        return self.size


def train_model(epochs, batch_size, lr=0.01, margin=1.5, threshold=0.5):
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
    dataset_train = AvaPairs(phase="train", nb_positives=250)     # 2min06 # 39s (to cuda)
    # dataset_train = DummyDataset(size=1000)                       # 2min06 # 9s (to cuda)
    # dataset_train = LoadSamePairDataset(size=1000)                # 2min06 # 11s (to cuda)
    # dataset_train = ComputeSamePairDataset(size=1000)             # 2min08 # 36s (to cuda)

    # Dataloaders
    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # Train epochs
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        train_epoch(dataloader_train, model, epoch, loss_fn, optimizer, accuracy_fn)


if __name__ == "__main__":
    epochs = int(sys.argv[1])
    print("Nb epochs : {}".format(epochs))
    train_model(epochs=epochs, batch_size=8)
