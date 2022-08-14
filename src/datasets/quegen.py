import json
import random
import os.path as op

from tqdm import trange


class QuestionGenerator:

    def __init__(self, data_dir, annotation, key_fn, caption_fn, split, num_clips, size=5000):

        assert split in ("train", "val")
        self.train = split == "train"
        self.anno = annotation
        self.key_fn = key_fn
        self.caption_fn = caption_fn

        self.size = size
        self.num_clips = num_clips
        self.questions = (
                self.happen,
                self.happen_begin,
                self.happen_end,
                self.happen_before,
                self.happen_after,
                # self.happen_between,
        )

        if not self.train:
            fn = op.join(data_dir, 'tem_val.jsonl')
            if not op.isfile(fn):
                print("Warning: tem_val.jsonl not found. Generating validation data")
                self.val_data = self.generate_question_offline(size)
                with open(fn, 'w') as f:
                    for l in self.val_data:
                        print(json.dumps(l), file=f)
            else:
                self.val_data = [json.loads(l) for l in open(fn).readlines()]

    def __getitem__(self, idx):
        if self.train:
            raise NotImplementedError("only supports validation stage")
        return self.val_data[idx]

    def __len__(self):
        if self.train:
            raise NotImplementedError("only supports validation stage")
        return len(self.val_data)

    def __next__(self):
        """
        Returns:
            video names, question, choices, label, captions, question type
        """
        return random.choice(self.questions)()

    def generate_question_offline(self, size):
        N = len(self.questions)
        return [self.questions[i%N]() for i in trange(size)]

    def uniform_sample(self, extra=False):
        num_clips = self.num_clips + bool(extra)
        annos = random.sample(self.anno, k=num_clips)
        videos = [self.key_fn(anno) for anno in annos]
        captions = [self.caption_fn(anno) for anno in annos]
        return videos, captions

    def happen(self):
        videos, captions = self.uniform_sample()
        label = random.randrange(len(captions))
        choices = captions
        return [videos[label]], "what happened", choices, label, [captions[label]], 0

    def happen_begin(self):
        videos, captions = self.uniform_sample()
        label = 0
        choices = captions
        return videos, "what happened at the beginning", choices, label, captions, 1

    def happen_end(self):
        videos, captions = self.uniform_sample()
        label = len(captions) - 1
        choices = captions
        return videos, "what happened at the end", choices, label, captions, 2

    def happen_after(self):
        videos, captions = self.uniform_sample(extra=True)
        ref = random.randrange(len(captions)-2)
        ref_clip = captions[ref]
        label = ref + 1
        choices = captions[:-1]
        choices[ref] = captions[-1]
        return videos[:-1], f"what happened after {ref_clip}", choices, label, captions[:-1], 3

    def happen_before(self):
        videos, captions = self.uniform_sample(extra=True)
        ref = random.randrange(1, len(captions)-1)
        ref_clip = captions[ref]
        label = ref - 1
        choices = captions[:-1]
        choices[ref] = captions[-1]
        return videos[:-1], f"what happened before {ref_clip}", choices, label, captions[:-1], 4

    def happen_between(self):
        raise NotImplementedError
        videos, choices = self.uniform_sample(extra=True)
        ref1 = random.randrange(len(choices)-2)
        ref2 = ref1 + 2
        label = ref1 + 1
        return videos, f"what happened between {choices[ref1]} and {choices[ref2]}", choices, label, 5

