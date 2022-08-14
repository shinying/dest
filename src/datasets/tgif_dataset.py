import json
import os.path as op

from .trm_dataset import TRM


class Tgif(TRM):

    def __init__(self, *args, split, **kwargs):
        key_fn = lambda a: a["vid"]
        caption_fn = lambda a: a["cap"]

        data_name = "tgif"
        f = "tgif/train.json" if split == "train" else "tgif/val.json"
        anno = json.load(open(op.join(kwargs["data_dir"], f)))
        feat = "tgif/tgif.h5"

        super().__init__(*args,
                         split=split,
                         data_name=data_name,
                         anno=anno,
                         feat=feat,
                         key_fn=key_fn,
                         caption_fn=caption_fn,
                         **kwargs)
