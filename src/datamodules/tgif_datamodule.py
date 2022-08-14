from transformers import DistilBertTokenizer

from .datamodule_base import BaseDataModule
from ..datasets import Tgif


class TgifDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return Tgif

    @property
    def dataset_name(self):
        return "tgif"

    def setup(self, stage):
        super().setup(stage)
        # if not self.setup_flag:
        #     self.set_train_dataset()
        #     self.set_val_dataset()

        #     self.train_dataset.tokenizer = self.tokenizer
        #     self.val_dataset.tokenizer = self.tokenizer
        # self.train_dataset.set_ans_tokenizer(self.ans_tokenizer)
        # self.val_dataset.set_ans_tokenizer(self.ans_tokenizer)
        # self.test_dataset.set_ans_tokenizer(self.ans_tokenizer)

        #     self.setup_flag = True
