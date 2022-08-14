from .vatex_datamodule import VatexDataModule
from .tgif_datamodule import TgifDataModule

from .anetqa_datamodule import AnetQADataModule
from .agqa_datamodule import AGQADataModule


_datamodules = {
    "vatex": VatexDataModule,
    "tgif": TgifDataModule,

    "anetqa": AnetQADataModule,
    "agqa": AGQADataModule,
}
