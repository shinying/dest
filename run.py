import os
import copy
import pytorch_lightning as pl
import os

from pytorch_lightning.strategies.ddp import DDPStrategy
from src.config import ex
from src.modules.dest import DeST
from src.datamodules.multitask_datamodule import MTDataModule

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    print('', _config, '', sep='\n')
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)
    model = DeST(_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            verbose=True,
            monitor="val/the_metric",
            filename="step={step}-score={val/the_metric:.2%}",
            mode="max",
            auto_insert_metric_name=False,
            save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1].split(".")[0]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = max(_config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    ), 1)

    partial_tasks = ("trm", "anetqa_mean", "agqa_mean")
    find_unused_parameters = any(_config["loss_names"][task] for task in partial_tasks)

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        strategy = DDPStrategy(find_unused_parameters=find_unused_parameters),
        accelerator="gpu",
        benchmark=True,
        max_epochs=_config["max_epoch"],
        max_steps=_config["max_steps"],
        callbacks=callbacks,
        logger=logger,
        # prepare_data_per_node=False,
        replace_sampler_ddp=True,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm, ckpt_path=_config["resume_from"])
    else:
        trainer.test(model, datamodule=dm)
