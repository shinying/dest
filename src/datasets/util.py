"""Borrow from https://github.com/m-bain/frozen-in-time/blob/main/base/base_dataset.py
"""

import glob
import os.path as op
import random

import numpy as np
import torch


def sample_frames(num_frames, video_len, sample='rand', fix_start=-1):
    if num_frames >= video_len:
        return range(video_len)

    intv = np.linspace(start=0, stop=video_len, num=num_frames+1).astype(int)
    if sample == 'rand':
        frame_ids = [random.randrange(intv[i], intv[i+1]) for i in range(len(intv)-1)]
    elif fix_start >= 0:
        fix_start = int(fix_start)
        frame_ids = [intv[i]+fix_start for i in range(len(intv)-1)]
    elif sample == 'uniform':
        frame_ids = [(intv[i]+intv[i+1]-1) // 2 for i in range(len(intv)-1)]
    else:
        raise NotImplementedError
    return frame_ids


def read_frames(video_path, num_frames=-1, sample='rand', trim=1., fix_start=-1):
    frames = glob.glob(op.join(video_path, '*.png'))
    if not len(frames):
        raise FileNotFoundError("No such videos:", video_path)
    frames.sort(key=lambda n: int(op.basename(n)[:-4]))

    if trim < 1.:
        remain = (1. - trim) / 2
        start, end = int(len(frames) * remain), int(len(frames) * (1 - remain))
        frames = frames[start:end]
    if num_frames > 0:
        while len(frames) < num_frames: # duplicate frames
            frames = [f for frame in frames for f in (frame, frame)]
        frame_ids = sample_frames(num_frames, len(frames), sample, fix_start)
        return [frames[i] for i in frame_ids]
    return frames

