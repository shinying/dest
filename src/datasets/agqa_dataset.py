import json
import os.path as op

from PIL import Image
import h5py
import torch

from .base_dataset import BaseDataset
from .util import read_frames, sample_frames


class AGQA(BaseDataset):
    def __init__(self, *args, split, **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        super().__init__(*args, **kwargs)

        data = json.load(open(op.join(self.data_dir, split+'.json')))
        self.data = list(data.items())
        self.video_feat = h5py.File(op.join(self.data_dir, "agqa.h5"))

        self.ans2id = json.load(open(op.join(self.data_dir, "vocab.json")))
        self.ans = sorted(self.ans2id.keys(), key=lambda k: self.ans2id[k])
        # self.ans2id = {v: i for i, v in enumerate(self.ans)}

        assert len(self.transforms) == 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        qid, sample = self.data[index]
        frames = self.get_frames(sample["video_id"])
        video = self.get_video(sample["video_id"])
        # video = video[torch.randperm(video.size(0))]

        ans = sample["answer"]
        label = self.ans2id.get(ans, -1)

        return {
            "qid": qid,
            "frames": frames,
            "video": video,
            "text": sample["question"],
            "label": label,
        }

    def get_frames(self, video_name):
        v = op.join(self.data_dir, "frames", video_name)
        frames = read_frames(v, self.nframe, self.sampling, self.trim)
        frames = torch.stack([self.transforms[0](Image.open(frame)) for frame in frames])
        return frames

    def get_video(self, video_name):
        feat = torch.tensor(self.video_feat['videos/'+video_name][:])
        if feat.size(0) > self.max_video_len:
            fid = sample_frames(self.max_video_len, feat.size(0), self.sampling)
            feat = feat[fid]
        return feat

