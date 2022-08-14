from transformers import DistilBertTokenizer

from .datamodule_base import BaseDataModule
from ..datasets import Vatex


class VatexDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return Vatex

    @property
    def dataset_name(self):
        return "vatex"

    def setup(self, stage):
        super().setup(stage)
        # if not self.setup_flag:
        #     self.set_train_dataset()
        #     self.set_val_dataset()

        #     self.train_dataset.tokenizer = self.tokenizer
        #     self.val_dataset.tokenizer = self.tokenizer
        # self.train_dataset.ans_tokenizer = self.ans_tokenizer
        # self.val_dataset.ans_tokenizer = self.ans_tokenizer
        # self.test_dataset.ans_tokenizer = self.ans_tokenizer

        #     self.setup_flag = True
