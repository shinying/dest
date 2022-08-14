from ..datasets import AnetQA
from .datamodule_base import BaseDataModule
from collections import defaultdict


class AnetQADataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return AnetQA

    @property
    def dataset_name(self):
        return "anetqa"

    def setup(self, stage):
        super().setup(stage)

        self.train_dataset.ans_tokenizer = self.ans_tokenizer
        self.val_dataset.ans_tokenizer = self.ans_tokenizer
        self.test_dataset.ans_tokenizer = self.ans_tokenizer

        self.train_dataset.encode_choices()
        self.val_dataset.encode_choices()
        self.test_dataset.encode_choices()

        self.id2ans = defaultdict(lambda: "UNK")
        for ans, idx in self.train_dataset.ans2id.items():
            self.id2ans[idx] = ans
