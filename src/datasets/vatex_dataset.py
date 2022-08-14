import json
import os.path as op
import random

from .trm_dataset import TRM


class Vatex(TRM):

    def __init__(self, *args, split, **kwargs):
        key_fn = lambda a: a["videoID"]
        caption_fn = lambda a: random.choice(a["enCap"])

        data_name = "vatex"
        f = "vatex/train.json" if split == "train" else "vatex/val.json"
        anno = json.load(open(op.join(kwargs["data_dir"], f)))
        feat = "vatex/vatex.h5"

        super().__init__(*args,
                         split=split,
                         data_name=data_name,
                         anno=anno,
                         feat=feat,
                         key_fn=key_fn,
                         caption_fn=caption_fn,
                         **kwargs)
