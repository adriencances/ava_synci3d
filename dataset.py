import os
import sys
import numpy as np
import cv2
import glob
import tqdm
import pickle

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from dataset_aux import FrameProcessor


class AvaPairs(data.Dataset):
    def __init__(self, phase="train", nb_positives=None, seed=0):
        self.w = 224
        self.h = 224
        self.alpha = 0.1

        self.phase = phase
        self.frames_dir = "/media/hdd/adrien/Ava_v2.2/correct_frames"
        self.shots_dir = "/home/acances/Data/Ava_v2.2/final_shots"
        self.tracks_dir = "/home/acances/Data/Ava_v2.2/tracks"
        self.pairs_dir = "/home/acances/Data/Ava_v2.2/pairs16"

        self.frame_processor = FrameProcessor(self.w, self.h, self.alpha, self.phase, self.frames_dir, self.shots_dir, self.tracks_dir)

        random.seed(seed)
        self.nb_positives = nb_positives
        self.gather_positive_pairs()
        self.gather_negative_pairs()

    def gather_positive_pairs(self):
        print("Gathering positive pairs")
        self.positive_pairs = []
        pairs_files = glob.glob("{}/{}/positive/*".format(self.pairs_dir, self.phase))
        for file in tqdm.tqdm(pairs_files):
            with open(file, "r") as f:
                for line in f:
                    pair = line.strip().split(",")
                    self.positive_pairs.append(pair + [1])
        
        if self.nb_positives == None:
            self.nb_positives = len(self.positive_pairs)
        
        self.positive_pairs = random.sample(self.positive_pairs, self.nb_positives)
        random.shuffle(self.positive_pairs)
    
    def gather_negative_pairs(self):
        nb_hard_negatives = 2*self.nb_positives
        nb_medium_negatives = self.nb_positives // 2
        nb_easy_negatives = self.nb_positives // 2
        self.one_epoch_data_size = self.nb_positives \
            + nb_hard_negatives \
            + nb_medium_negatives \
            + nb_easy_negatives

        print("Gathering negative pairs")

        # Hard negatives
        self.hard_negative_pairs = []
        pairs_files = glob.glob("{}/{}/hard_negative/*".format(self.pairs_dir, self.phase))
        for file in tqdm.tqdm(pairs_files):
            with open(file, "r") as f:
                for line in f:
                    pair = line.strip().split(",")
                    self.hard_negative_pairs.append(pair + [0])

        # Medium negatives
        self.medium_negative_pairs = []
        pairs_files = glob.glob("{}/{}/medium_negative/*".format(self.pairs_dir, self.phase))
        for file in tqdm.tqdm(pairs_files):
            with open(file, "r") as f:
                for line in f:
                    pair = line.strip().split(",")
                    self.medium_negative_pairs.append(pair + [0])

        # Easy negatives
        self.easy_negative_pairs = []
        pairs_files = glob.glob("{}/{}/easy_negative/*".format(self.pairs_dir, self.phase))
        for file in tqdm.tqdm(pairs_files):
            with open(file, "r") as f:
                for line in f:
                    pair = line.strip().split(",")
                    self.easy_negative_pairs.append(pair + [0])

        # Make sure that the proportion of the negative pairs are correct
        number = min(
            len(self.hard_negative_pairs) // 2,
            len(self.medium_negative_pairs) * 2,
            len(self.easy_negative_pairs) * 2
        )

        self.negative_pairs = []
        self.negative_pairs += random.sample(self.hard_negative_pairs, number*2)
        self.negative_pairs += random.sample(self.medium_negative_pairs, number//2)
        self.negative_pairs += random.sample(self.easy_negative_pairs, number//2)

    def __getitem__(self, index):
        "Generates one sample of data"
        assert index < self.one_epoch_data_size

        # For positive pairs, choose among the selected positive pairs.
        if index < self.nb_positives:
            pair = self.positive_pairs[index]
        # For negative pairs, randomly sample among all the negative pairs.
        else:
            pair = random.choice(self.negative_pairs)

        video_id1, shot_id1, i1, begin1, end1, video_id2, shot_id2, i2, begin2, end2, label = pair
        shot_id1, track_id1, begin1, end1 = list(map(int, [shot_id1, i1, begin1, end1]))
        shot_id2, track_id2, begin2, end2 = list(map(int, [shot_id2, i2, begin2, end2]))
        assert end1 - begin1 == end2 - begin2

        tensor1 = self.frame_processor.processed_frames(video_id1, shot_id1, track_id1, begin1, end1)
        tensor2 = self.frame_processor.processed_frames(video_id2, shot_id2, track_id2, begin2, end2)

        return tensor1, tensor2, label

    def __len__(self):
        """Denotes the total number of samples"""
        return self.one_epoch_data_size


if __name__ == "__main__":
    dataset = AvaPairs("train")
    # print(len(dataset))
    # tensor1, tensor2, label = dataset[0]
    # print(tensor1.shape)
    # print(tensor2.shape)


