import torch

from torch.nn.utils.rnn import pad_sequence
from ..transforms import keys_to_transforms


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self,
            data_dir: str,
            transform_keys: list,
            image_size: int,
            max_text_len=None,
            max_ans_len=None,
            nframe=16,
            max_video_len=100,
            trim=1,
            num_clips=8,
    ):
        assert len(transform_keys) >= 1
        super().__init__()

        self.data_dir = data_dir

        self.transforms = keys_to_transforms(transform_keys, size=image_size)
        self.clip_transform = False
        for transform_key in transform_keys:
            if 'clip' in transform_key:
                self.clip_transform = True
                break

        self.max_video_len = max_video_len
        self.max_text_len = max_text_len
        self.max_ans_len = max_ans_len
        self.nframe = nframe
        self.trim = trim
        self.num_clips = num_clips

        self.sampling = "rand" if self.split == "train" else "uniform"

    def encode_text(self, text):
        encoding = self.tokenizer(
                text,
                padding=True,
                return_tensors="pt",
        )
        return encoding

    def encode_choices(self):
        encoding = self.ans_tokenizer(
                self.ans,
                padding=True,
                return_tensors="pt"
        )
        self.ans = encoding

    def collate(self, batch):
        qid = [data["qid"] for data in batch]

        frames = torch.cat([data["frames"] for data in batch])
        video = pad_sequence([data["video"] for data in batch], batch_first=True)
        video_mask = torch.zeros(video.size(0), video.size(1)+2, dtype=torch.long)
        for i, data in enumerate(batch):
            video_mask[i, :data["video"].size(0)] = 1

        questions = [data["text"] for data in batch]
        question_encoding = self.encode_text(questions)

        labels = torch.tensor([data["label"] for data in batch], dtype=torch.long)

        return {"qid": qid, "frames": frames, "nframe": self.nframe,
                "video": video, "video_mask": video_mask,
                "questions": question_encoding, "ans": self.ans,
                "labels": labels}
