import torch
import random

from transformers.optimization import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from ..gadgets.my_metrics import Accuracy, Scalar


def parse_loading_msg(msg):
    miss = set(m.split('.')[0] for m in msg.missing_keys)
    unexp = set(m.split('.')[0] for m in msg.unexpected_keys)
    print("Missing:", miss if len(miss) else "None")
    print("Unexpected:", unexp if len(unexp) else "None")


def set_metrics(pl_module):
    for split in ["train", "val"]:
        for task, flag in pl_module.hparams.config["loss_names"].items():
            if flag:
                task = task.split('_')[0]
                setattr(pl_module, f"{split}_{task}_loss", Scalar())
                setattr(pl_module, f"{split}_{task}_accuracy", Accuracy())
                if task == "trm":
                    setattr(pl_module, f"{split}_align_loss", Scalar())
                    setattr(pl_module, f"{split}_align_accuracy", Accuracy())


def epoch_wrapup(pl_module):
    phase = "train" if pl_module.training else "val"
    the_metric = 0

    for task, flag in pl_module.hparams.config["loss_names"].items():
        if not flag:
            continue

        value = 0
        task = task.split('_')[0] # for anetqa_mean and agqa_mean

        m = getattr(pl_module, f"{phase}_{task}_loss")
        pl_module.log(f"{task}/{phase}/loss_epoch", m.compute(), rank_zero_only=True)
        m.reset()

        if task == "trm":
            m = getattr(pl_module, f"{phase}_align_loss")
            pl_module.log(f"{task}/{phase}/align_loss_epoch", m.compute(), rank_zero_only=True)
            m.reset()
            m = getattr(pl_module, f"{phase}_align_accuracy")
            pl_module.log(f"{task}/{phase}/align_accuracy_epoch", m.compute(), rank_zero_only=True)
            m.reset()

        m = getattr(pl_module, f"{phase}_{task}_accuracy")
        value = m.compute()
        pl_module.log(f"{task}/{phase}/accuracy_epoch", value, rank_zero_only=True)
        m.reset()

        the_metric += value

    pl_module.log(f"{phase}/the_metric", the_metric, rank_zero_only=True)


def set_schedule(pl_module):
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
        "ln.bias",
        "ln.weight",
    ]

    time_model_names = ["temporal", "video_encoder"]
    head_names = ["vision_proj", "text_proj", "tau",
                  "video_head", "frame_head", "pooler"]
    ans_model_names = ["ans_encoder"]

    lr_mult_head = pl_module.hparams.config["lr_mult_head"]
    lr_mult_ans = pl_module.hparams.config["lr_mult_ans"]
    lr_time = pl_module.hparams.config["lr_time"]

    end_lr = pl_module.hparams.config["end_lr"]
    decay_power = pl_module.hparams.config["decay_power"]
    optim_type = pl_module.hparams.config["optim_type"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and not any(ht in n for ht in ans_model_names)
                and not any(tm in n for tm in time_model_names)
                and p.requires_grad
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and not any(ht in n for ht in ans_model_names)
                and not any(tm in n for tm in time_model_names)
                and p.requires_grad
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
                and not any(ht in n for ht in ans_model_names)
                and not any(tm in n for tm in time_model_names)
                and p.requires_grad
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult_head,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
                and not any(ht in n for ht in ans_model_names)
                and not any(tm in n for tm in time_model_names)
                and p.requires_grad
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult_head,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and any(ht in n for ht in ans_model_names)
                and not any(tm in n for tm in time_model_names)
                and p.requires_grad
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult_ans,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and any(ht in n for ht in ans_model_names)
                and not any(tm in n for tm in time_model_names)
                and p.requires_grad
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult_ans,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and not any(ht in n for ht in ans_model_names)
                and any(tm in n for tm in time_model_names)
                and p.requires_grad
            ],
            "weight_decay": wd,
            "lr": lr * lr_time,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and not any(ht in n for ht in ans_model_names)
                and any(tm in n for tm in time_model_names)
                and p.requires_grad
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_time,
        },
    ]

    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    if pl_module.trainer.max_steps < 0:
        max_steps = (
            len(pl_module.trainer.datamodule.train_dataloader())
            * pl_module.trainer.max_epochs
            // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )
