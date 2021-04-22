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

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset_aux import FrameProcessor
from dataset import AvaPairs
from synci3d import SyncI3d
from train import train
from contrastive_loss import ContrastiveLoss

import time


def main(epochs, batch_size, lr):
    train(epochs=epochs, batch_size=batch_size, lr=lr)

    # dataset = AvaPairs("train")
    # dataloader_train = data.DataLoader(dataset, batch_size= 3, shuffle=True, num_workers=8)

    # model = SyncI3d()
    # model.cuda()

    # loss_fn = ContrastiveLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.000001)
    # lr_sched = optim.lr_scheduler.StepLR(optimizer, 10)

    # cuda = torch.cuda.is_available()

    # train_epoch(dataloader_train, model, 0, loss_fn, optimizer, lr_sched, cuda, log_interval=0, metric=BinaryClassificationAccuracy)


if __name__ == "__main__":
    epochs = int(sys.argv[1])
    lr = float(sys.argv[2])
    print("Nb epochs : {}".format(epochs))
    print("Learning rate : {}".format(lr))
    main(epochs=epochs, batch_size=16, lr=lr)

    # dataset = AvaPairs("train")

    # train_loader = torch.utils.data.DataLoader(
    #     dataset=dataset,
    #     batch_size=4,
    #     shuffle=False,
    #     num_workers=0,
    #     pin_memory=True
    # )

    # print("Starting loop")
    # start = time.time()
    # for batch in tqdm.tqdm(train_loader):
    #     continue
    # end = time.time()

    # print(end - begin)
