import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import sys

from utils import get_video_frames, get_specific_video_frames, write_frames


def make_annotated_video(video_file, tracks_pkl_file):
    frames = get_specific_video_frames(video_file, 902*30, 910*30)
    write_frames(frames, "AVA_random_frames/")
    print(len(frames))


if __name__ == "__main__":
    video_file = sys.argv[1]
    make_annotated_video(video_file, None)

