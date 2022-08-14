import os.path as op

from torch.nn.utils.rnn import pad_sequence
import h5py
import numpy as np
import torch

from .base_dataset import BaseDataset
from .quegen import QuestionGenerator
from .util import sample_frames


class TRM(BaseDataset):

    def __init__(self, *args, split, data_name, anno, feat, key_fn, caption_fn, **kwargs):
        if split == "test": split = "val" # No test split
        assert split in ["train", "val"]
        self.train = split == "train"
        self.split = split
        super().__init__(*args, **kwargs)

        sub_dir = op.join(self.data_dir, data_name)
        self.qg = QuestionGenerator(sub_dir, anno, key_fn, caption_fn, self.split, self.num_clips)
        self.video_feat = h5py.File(op.join(self.data_dir, feat))

    def __len__(self):
        return 10**5 if self.train else len(self.qg)

    def __getitem__(self, idx):
        videos, question, choices, label, captions, qtype = next(self.qg) if self.train else self.qg[idx]
        feat, lens = self.get_video(videos)

        return {
            "video": feat,
            "lens": lens,
            "text": question,
            "label": label,
            "choices": choices,
            "captions": captions,
            "qtype": qtype,
        }

    def get_video(self, videos):
        feat = [torch.tensor(self.video_feat[key][:]) for key in videos]
        lens = np.cumsum([f.size(0) for f in feat])
        feat = torch.cat(feat)

        if feat.size(0) > self.max_video_len:
            lens = lens / feat.size(0) * self.max_video_len
            fid = sample_frames(self.max_video_len, feat.size(0), self.sampling)
            feat = feat[fid]
        lens = [slice(0, int(lens[0]+0.5))] + \
               [slice(int(lens[i]), int(lens[i+1]+0.5)) for i in range(len(lens)-1)]

        return feat, lens

    def encode_text(self, text):
        encoding = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_text_len+5,
            return_tensors="pt",
        )
        return encoding

    def encode_choices(self, choices):
        encoding = self.ans_tokenizer(
            choices,
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        return encoding

    def collate(self, batch): #, mlm_collator):
        video = pad_sequence([data["video"] for data in batch], batch_first=True)
        video_mask = torch.zeros(video.size(0), video.size(1)+2, dtype=torch.long)
        for i, data in enumerate(batch):
            video_mask[i, :data["video"].size(0)+2] = 1
        lens = [data["lens"] for data in batch]

        questions = [data["text"] for data in batch]
        question_encoding = self.encode_text(questions)

        choices = [ans for data in batch for ans in data["choices"]]
        choices_encoding = self.encode_choices(choices)

        match_text = [cap for data in batch for cap in data["captions"]]
        match_encoding = self.encode_text(match_text)

        labels = torch.tensor([data["label"] for i, data in enumerate(batch)], dtype=torch.long)
        qtypes = [data["qtype"] for data in batch]

        return {"video": video, "video_mask": video_mask, "lens": lens,
                "questions": question_encoding, "choices": choices_encoding,
                "matching": match_encoding,
                "labels": labels, "nchoice": self.qg.num_clips, "qtypes": qtypes}
