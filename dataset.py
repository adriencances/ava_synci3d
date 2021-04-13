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

from dataset_aux import FrameProcessor


class AvaPairs(data.Dataset):
    def __init__(self, phase="train"):
        self.w = 224
        self.h = 224
        self.alpha = 0.1

        self.phase = "train"
        self.frames_dir = "/media/hdd/adrien/Ava_v2.2/frames"
        self.shots_dir = "/home/acances/Data/Ava_v2.2/final_shots"
        self.tracks_dir = "/home/acances/Data/Ava_v2.2/tracks"
        self.pairs_dir = "/home/acances/Data/Ava_v2.2/pairs"

        self.frame_processor = FrameProcessor(self.w, self.h, self.alpha, self.phase, self.frames_dir, self.shots_dir, self.tracks_dir)

        self.gather_positive_pairs()



        # #videos
        # self.all_videos = tvhid_splits(phase=phase)
        # self.n_videos = len(self.all_videos)
        # self.create(10) # intialise positive and negative candidates with temporal shift and stride
        # self.key_2_idx_pos = {i:k for i,k in enumerate(self.positives.keys())}
        # self.key_2_idx_neg = {i:k for i,k in enumerate(self.negatives.keys())}
        # self.create_total()
        # self.key_2_idx = {i:k for i,k in enumerate(self.total.keys())} #to ease data retrieval
        # self.key_list = [k for k in self.total.keys()] #useful for label checking
    
    def gather_positive_pairs(self):
        print("Gathering positive pairs")
        self.positive_pairs = []
        positive_pairs_files = glob.glob("{}/{}/positive/*".format(self.pairs_dir, self.phase))
        for file in tqdm.tqdm(positive_pairs_files):
            with open(file, "r") as f:
                for line in f:
                    pair = line.strip().split(",")
                    self.positive_pairs.append(pair)

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
        
        return tensor1, tensor2

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.positive_pairs)


if __name__ == "__main__":
    data = AvaPairs("val")
    print(len(data))
    tensor1, tensor2 = data[0]
    print(tensor1.shape)
    print(tensor2.shape)


