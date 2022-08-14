import csv
import json
import os.path as op

from PIL import Image
import h5py
import torch

from .base_dataset import BaseDataset
from .util import read_frames, sample_frames


class AnetQA(BaseDataset):
    def __init__(self, *args, split, **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        super().__init__(*args, **kwargs)

        self.data = list(csv.DictReader(open(op.join(self.data_dir, split+'.csv'))))
        self.video_feat = h5py.File(op.join(self.data_dir, "anetqa.h5"))

        self.ans2id = json.load(open(op.join(self.data_dir, "vocab.json")))
        self.ans = sorted(self.ans2id.keys(), key=lambda k: self.ans2id[k])
        # self.ans = torch.load(op.join(self.data_dir, "ans.pt"))

        assert len(self.transforms) == 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        frames = self.get_frames(sample["video_id"])
        video = self.get_video(sample["video_id"])
        # video = video[torch.randperm(video.size(0))]

        ans = sample["answer"]
        label = self.ans2id.get(ans, -1)

        return {
            "qid": sample["qid"],
            "frames": frames,
            "video": video,
            "text": sample["question"],
            "label": label,
        }

    def get_frames(self, video_name):
        v = op.join(self.data_dir, "frames", f"v_{video_name}")
        frames = read_frames(v, self.nframe, self.sampling, self.trim)
        frames = torch.stack([self.transforms[0](Image.open(frame)) for frame in frames])
        return frames

    def get_video(self, video_name):
        feat = torch.tensor(self.video_feat[video_name][:])
        if feat.size(0) > self.max_video_len:
            fid = sample_frames(self.max_video_len, feat.size(0), self.sampling)
            feat = feat[fid]
        return feat

