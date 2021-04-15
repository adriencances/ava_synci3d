import os
import sys
import numpy as np
import cv2
import glob
import tqdm

from random import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from dataset_aux import FrameProcessor


class AvaPairs(data.Dataset):
    def __init__(self, phase="train"):
        self.w = 224
        self.h = 224
        self.alpha = 0.1

        self.phase = "train"
        self.frames_dir = "/media/hdd/adrien/Ava_v2.2/correct_frames"
        self.shots_dir = "/home/acances/Data/Ava_v2.2/final_shots"
        self.tracks_dir = "/home/acances/Data/Ava_v2.2/tracks"
        self.pairs_dir = "/home/acances/Data/Ava_v2.2/pairs"

        self.frame_processor = FrameProcessor(self.w, self.h, self.alpha, self.phase, self.frames_dir, self.shots_dir, self.tracks_dir)

        self.gather_positive_pairs()
        self.gather_negative_pairs()
        self.create_data()

    def gather_positive_pairs(self):
        print("Gathering positive pairs")
        self.positive_pairs = []
        pairs_files = glob.glob("{}/{}/positive/*".format(self.pairs_dir, self.phase))
        for file in tqdm.tqdm(pairs_files):
            with open(file, "r") as f:
                for line in f:
                    pair = line.strip().split(",")
                    self.positive_pairs.append(pair)
    
    def gather_negative_pairs(self):
        nb_positives = len(self.positive_pairs)

        nb_hard_negatives = 2*nb_positives
        nb_medium_negatives = nb_positives // 2
        nb_easy_negatives = nb_positives // 2

        print("Gathering negative pairs")

        # Hard negatives
        self.hard_negative_pairs = []
        pairs_files = glob.glob("{}/{}/hard_negative/*".format(self.pairs_dir, self.phase))
        for file in tqdm.tqdm(pairs_files):
            with open(file, "r") as f:
                for line in f:
                    pair = line.strip().split(",")
                    self.hard_negative_pairs.append(pair)
        # Medium negatives
        self.medium_negative_pairs = []
        pairs_files = glob.glob("{}/{}/medium_negative/*".format(self.pairs_dir, self.phase))
        for file in tqdm.tqdm(pairs_files):
            with open(file, "r") as f:
                for line in f:
                    pair = line.strip().split(",")
                    self.medium_negative_pairs.append(pair)
        # Easy negatives
        self.easy_negative_pairs = []
        pairs_files = glob.glob("{}/{}/easy_negative/*".format(self.pairs_dir, self.phase))
        for file in tqdm.tqdm(pairs_files):
            with open(file, "r") as f:
                for line in f:
                    pair = line.strip().split(",")
                    self.easy_negative_pairs.append(pair)

        # Suffle and sample in the right proportion
        shuffle(self.hard_negative_pairs)
        shuffle(self.medium_negative_pairs)
        shuffle(self.easy_negative_pairs)

        self.negative_pairs = []
        self.negative_pairs += self.hard_negative_pairs[:nb_hard_negatives]
        self.negative_pairs += self.medium_negative_pairs[:nb_medium_negatives]
        self.negative_pairs += self.easy_negative_pairs[:nb_easy_negatives]
    
    def create_data(self):
        # Concatenate positive and negative pairs, and shuffle
        self.data = self.positive_pairs + self.negative_pairs
        shuffle(self.data)

    def __getitem__(self, index):
        "Generates one sample of data"
        nb_pairs = len(self.positive_pairs)
        assert index < nb_pairs
        pair = self.positive_pairs[index]
        video_id1, shot_id1, i1, begin1, end1, video_id2, shot_id2, i2, begin2, end2 = pair
        shot_id1, track_id1, begin1, end1 = list(map(int, [shot_id1, i1, begin1, end1]))
        shot_id2, track_id2, begin2, end2 = list(map(int, [shot_id2, i2, begin2, end2]))
        assert end1 - begin1 == end2 - begin2

        tensor1 = self.frame_processor.processed_frames(video_id1, shot_id1, track_id1, begin1, end1)
        tensor2 = self.frame_processor.processed_frames(video_id2, shot_id2, track_id2, begin2, end2)
        
        return tensor1, tensor2, 1

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)


if __name__ == "__main__":
    dataset = AvaPairs("train")
    # print(len(dataset))
    # tensor1, tensor2, label = dataset[0]
    # print(tensor1.shape)
    # print(tensor2.shape)


