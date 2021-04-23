import os
import sys
import numpy as np
import cv2
import glob
import tqdm
import pickle

from random import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from dataset_aux import FrameProcessor


class AvaPairs(data.Dataset):
    def __init__(self, phase="train", nb_positives=None):
        self.w = 224
        self.h = 224
        self.alpha = 0.1

        self.phase = phase
        self.frames_dir = "/media/hdd/adrien/Ava_v2.2/correct_frames"
        self.shots_dir = "/home/acances/Data/Ava_v2.2/final_shots"
        self.tracks_dir = "/home/acances/Data/Ava_v2.2/tracks"
        self.pairs_dir = "/home/acances/Data/Ava_v2.2/pairs16"

        self.frame_processor = FrameProcessor(self.w, self.h, self.alpha, self.phase, self.frames_dir, self.shots_dir, self.tracks_dir)

        self.nb_positives = nb_positives
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
                    self.positive_pairs.append(pair + [1])
        shuffle(self.positive_pairs)
        
        if self.nb_positives == None:
            self.nb_positives = len(self.positive_pairs)
        self.positive_pairs = self.positive_pairs[:self.nb_positives]
    
    def gather_negative_pairs(self):
        nb_hard_negatives = 2*self.nb_positives
        nb_medium_negatives = self.nb_positives // 2
        nb_easy_negatives = self.nb_positives // 2

        print("Gathering negative pairs")

        # For each category (hard, medium, easy), we gather all the corresponding pairs,
        # shuffle them, and keep only the number we need

        # Hard negatives
        self.hard_negative_pairs = []
        pairs_files = glob.glob("{}/{}/hard_negative/*".format(self.pairs_dir, self.phase))
        for file in tqdm.tqdm(pairs_files):
            with open(file, "r") as f:
                for line in f:
                    pair = line.strip().split(",")
                    self.hard_negative_pairs.append(pair + [0])
        shuffle(self.hard_negative_pairs)
        self.hard_negative_pairs = self.hard_negative_pairs[:nb_hard_negatives]
        # Medium negatives
        self.medium_negative_pairs = []
        pairs_files = glob.glob("{}/{}/medium_negative/*".format(self.pairs_dir, self.phase))
        for file in tqdm.tqdm(pairs_files):
            with open(file, "r") as f:
                for line in f:
                    pair = line.strip().split(",")
                    self.medium_negative_pairs.append(pair + [0])
        shuffle(self.medium_negative_pairs)
        self.medium_negative_pairs = self.medium_negative_pairs[:nb_medium_negatives]
        # Easy negatives
        self.easy_negative_pairs = []
        pairs_files = glob.glob("{}/{}/easy_negative/*".format(self.pairs_dir, self.phase))
        for file in tqdm.tqdm(pairs_files):
            with open(file, "r") as f:
                for line in f:
                    pair = line.strip().split(",")
                    self.easy_negative_pairs.append(pair + [0])
        shuffle(self.easy_negative_pairs)
        self.easy_negative_pairs = self.easy_negative_pairs[:nb_easy_negatives]

        self.negative_pairs = []
        self.negative_pairs += self.hard_negative_pairs
        self.negative_pairs += self.medium_negative_pairs
        self.negative_pairs += self.easy_negative_pairs
    
    def create_data(self):
        # Concatenate positive and negative pairs, and shuffle
        self.data = self.positive_pairs + self.negative_pairs
        shuffle(self.data)

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

        # with open("pairs_tensors/pair_0.pkl", "rb") as f:
        #     tensor1, tensor2, label = pickle.load(f)
        
        return tensor1, tensor2, label

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)


if __name__ == "__main__":
    dataset = AvaPairs("train")
    # print(len(dataset))
    # tensor1, tensor2, label = dataset[0]
    # print(tensor1.shape)
    # print(tensor2.shape)


