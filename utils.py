import numpy as np
import cv2


def get_video_frames(video_file):
    video = cv2.VideoCapture(video_file)
    frames = []
    while video.isOpened():
        success, frame = video.read()
        if success:
            frames.append(frame)
        else:
            break
    frames = np.array(frames)
    return frames


def get_specific_video_frames(video_file, start, end):
    video = cv2.VideoCapture(video_file)
    frames = []
    cnt = 0
    while video.isOpened():
        success, frame = video.read()
        if success:
            if start <= cnt <= end:
                frames.append(frame)
            if cnt > end:
                break
        else:
            break
        cnt += 1
    frames = np.array(frames)
    return frames


def write_frames(frames, frame_file_prefix):
    for frame_id, frame in enumerate(frames):
        frame_file = "{}{:06d}.jpg".format(frame_file_prefix, frame_id + 1)
        cv2.imwrite(frame_file, frame)

