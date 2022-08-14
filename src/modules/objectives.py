import os
import glob
import json
import tqdm
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn as dist_nn

from .dist_utils import all_gather, get_rank, get_world_size


def gather(tensor):
    world_size = get_world_size()
    device = tensor.device

    local_size = torch.tensor(tensor.size(0), device=device)
    sizes = [torch.empty_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(sizes, local_size)

    M = max(sizes)
    if local_size < M:
        tensor = torch.cat([tensor, torch.zeros(M-local_size, tensor.size(1), device=device)])
    tensors = dist_nn.all_gather(tensor)
    tensors = [tensor[:s] for tensor, s in zip(tensors, sizes)]
    return torch.cat(tensors)


def compute_align(pl_module, task, vid_feat, text_feat):
    # if clip_feat.size(0) != text_feat.size(0):
    #     lens = torch.tensor(batch["lens"], device=text_feat.device).view(-1)
    #     text_feat = text_feat[lens>0]
        # text_feat = torch.cat([f[:len(l)] for f, l in zip(text_feat, batch["lens"])])

    vid_feat = F.normalize(pl_module.vision_proj(vid_feat), dim=1)
    text_feat = F.normalize(pl_module.text_proj(text_feat), dim=1)

    vid_feat = gather(vid_feat)
    text_feat = gather(text_feat)

    sim_v2t = vid_feat @ text_feat.transpose(0, 1).clone().detach() / pl_module.tau
    sim_t2v = text_feat @ vid_feat.transpose(0, 1).clone().detach() / pl_module.tau

    labels = torch.arange(sim_v2t.size(0), device=sim_v2t.device)
    loss = (F.cross_entropy(sim_v2t, labels) + F.cross_entropy(sim_t2v, labels)) / 2

    phase = "train" if pl_module.training else "val"
    loss_value = getattr(pl_module, f"{phase}_align_loss")(loss)
    score = getattr(pl_module, f"{phase}_align_accuracy")(sim_v2t, labels)

    B = sim_v2t.size(0)
    pl_module.log(f"{task}/{phase}/align_loss", loss_value, batch_size=B, rank_zero_only=True)
    pl_module.log(f"{task}/{phase}/align_accuracy", score, batch_size=B, rank_zero_only=True)

    return loss, labels, sim_v2t, sim_t2v


def compute_trm(pl_module, batch):
    ques_hidden = pl_module.encode_question(batch)
    vid_hidden, vid_feat = pl_module.encode_video(batch, ques_hidden, batch["questions"].attention_mask, return_hidden=True)
    cls_feat = vid_feat
    ans_feat = pl_module.ans_encoder(batch["choices"])

    B = batch["labels"].size(0)
    ans_feat = ans_feat.view(B, batch["nchoice"], -1)
    logits = torch.bmm(ans_feat, cls_feat.unsqueeze(2)).squeeze(2)
    # logits = cls_feat @ ans_feat.t()
    # labels = torch.arange(cls_feat.size(0), device=cls_feat.device) * batch["nchoice"] + batch["labels"]
    loss = F.cross_entropy(logits, batch["labels"])


    # ========== video-text aligning ========== #

    clip_feat = pl_module.encode_clip(batch, vid_hidden)
    text_feat = pl_module.encode_text(batch)[:, 0]

    align_loss, align_labels, sim_v2t, sim_t2v = compute_align(pl_module, "trm", clip_feat, text_feat)

    ret = {
        "trm_loss": loss,
        "trm_logits": logits,
        "trm_labels": batch["labels"],
        "align_loss": align_loss,
        "align_labels": align_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_trm_loss")(loss)
    score = getattr(pl_module, f"{phase}_trm_accuracy")(logits, batch["labels"])

    pl_module.log(f"trm/{phase}/loss", loss, batch_size=B, rank_zero_only=True)
    pl_module.log(f"trm/{phase}/accuracy", score, batch_size=B, rank_zero_only=True)

    return ret


def compute_anetqa_mean(pl_module, batch):
    ques_hidden = pl_module.encode_question(batch)
    frame_feat = pl_module.encode_frames(batch, ques_hidden)
    ans_feat = pl_module.ans_encoder(batch["ans"])

    logits = frame_feat @ ans_feat.T
    loss = F.cross_entropy(logits, batch["labels"], ignore_index=-1)

    ret = {
        "anetqa_loss": loss,
        "anetqa_logits": logits,
        "anetqa_labels": batch["labels"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_anetqa_loss")(loss)
    score = getattr(pl_module, f"{phase}_anetqa_accuracy")(logits, batch["labels"])

    B = logits.size(0)
    pl_module.log(f"anetqa/{phase}/loss", loss, batch_size=B, rank_zero_only=True)
    pl_module.log(f"anetqa/{phase}/accuracy", score, batch_size=B, rank_zero_only=True)

    return ret


def compute_anetqa(pl_module, batch):
    ques_hidden = pl_module.encode_question(batch)
    frame_feat = pl_module.encode_frames(batch, ques_hidden)
    vid_feat = pl_module.encode_video(batch, ques_hidden, batch["questions"].attention_mask, return_hidden=False)

    cls_feat = frame_feat + vid_feat
    ans_feat = pl_module.ans_encoder(batch["ans"])

    logits = cls_feat @ ans_feat.T
    loss = F.cross_entropy(logits, batch["labels"], ignore_index=-1)

    ret = {
        "anetqa_loss": loss,
        "anetqa_logits": logits,
        "anetqa_labels": batch["labels"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_anetqa_loss")(loss)
    score = getattr(pl_module, f"{phase}_anetqa_accuracy")(logits, batch["labels"])

    B = logits.size(0)
    pl_module.log(f"anetqa/{phase}/loss", loss, batch_size=B, rank_zero_only=True)
    pl_module.log(f"anetqa/{phase}/accuracy", score, batch_size=B, rank_zero_only=True)

    return ret


def compute_agqa_mean(pl_module, batch):
    ques_hidden = pl_module.encode_question(batch)
    frame_feat = pl_module.encode_frames(batch, ques_hidden)
    ans_feat = pl_module.ans_encoder(batch["ans"])

    logits = frame_feat @ ans_feat.T
    loss = F.cross_entropy(logits, batch["labels"], ignore_index=-1)

    ret = {
        "agqa_loss": loss,
        "agqa_logits": logits,
        "agqa_labels": batch["labels"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_agqa_loss")(loss)
    score = getattr(pl_module, f"{phase}_agqa_accuracy")(logits, batch["labels"])

    B = logits.size(0)
    pl_module.log(f"agqa/{phase}/loss", loss, batch_size=B, rank_zero_only=True)
    pl_module.log(f"agqa/{phase}/accuracy", score, batch_size=B, rank_zero_only=True)

    return ret


def compute_agqa(pl_module, batch):
    ques_hidden = pl_module.encode_question(batch)
    frame_feat = pl_module.encode_frames(batch, ques_hidden)
    vid_feat = pl_module.encode_video(batch, ques_hidden, batch["questions"].attention_mask, return_hidden=False)

    cls_feat = frame_feat + vid_feat
    ans_feat = pl_module.ans_encoder(batch["ans"])

    logits = cls_feat @ ans_feat.T
    loss = F.cross_entropy(logits, batch["labels"], ignore_index=-1)

    ret = {
        "agqa_loss": loss,
        "agqa_logits": logits,
        "agqa_labels": batch["labels"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_agqa_loss")(loss)
    score = getattr(pl_module, f"{phase}_agqa_accuracy")(logits, batch["labels"])

    B = logits.size(0)
    pl_module.log(f"agqa/{phase}/loss", loss, batch_size=B, rank_zero_only=True)
    pl_module.log(f"agqa/{phase}/accuracy", score, batch_size=B, rank_zero_only=True)

    return ret



def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def trm_test_step(pl_module, batch, output):
    logits = output["trm_logits"]
    preds = logits.argmax(dim=1).cpu()
    return {"preds": preds, "labels": batch["labels"].cpu(), "qtypes": batch["qtypes"]}


def anetqa_test_step(pl_module, batch, output):
    id2ans = pl_module.trainer.datamodule.dm_dicts["anetqa"].id2ans

    logits = output["anetqa_logits"]
    preds = logits.argmax(dim=-1).cpu().tolist()
    preds = [id2ans[pred] for pred in preds]

    preds10 = logits.argsort(dim=-1, descending=True).cpu()[:, :10].tolist()
    preds10 = [[id2ans[p] for p in pred10] for pred10 in preds10]

    return {"qids": batch["qid"], "preds": preds, "preds10": preds10}


def agqa_test_step(pl_module, batch, output):
    id2ans = pl_module.trainer.datamodule.dm_dicts["agqa"].id2ans

    logits = output["agqa_logits"]
    preds = logits.argmax(dim=-1).cpu().tolist()
    preds = [id2ans[pred] for pred in preds]

    return {"qids": batch["qid"], "preds": preds}


def trm_test_wrapup(outs, model_name):
    preds, labels, qtypes = [], [], []
    for out in outs:
        preds.append(out["preds"])
        labels.append(out["labels"])
        qtypes += out["qtypes"]

    preds = torch.cat(preds)
    labels = torch.cat(labels)
    qtypes = torch.tensor(qtypes)
    q, cnt = torch.unique(qtypes, sorted=True, return_counts=True)

    cor = preds == labels
    acc = [cor[qtypes==i].sum().item() for i in q]
    cnt = cnt.tolist()

    print("Counts:", cnt)
    print("Accuracy:", [f"{a/c:.2%}" for a, c in zip(acc, cnt)])
    print("Overall:", f"{cor.float().mean().item():.2%}")


def anetqa_test_wrapup(outs, model_name):
    rank = torch.distributed.get_rank()
    qids, preds, preds10 = list(), list(), list()
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]
        preds10 += out["preds10"]

    rets = [{"qid": qid, "answer": pred, "answer10": pred10} \
            for qid, pred, pred10 in zip(qids, preds, preds10)]
    with open(f"anetqa_tmp_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("anetqa_tmp_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result", exist_ok=True)
        with open(f"result/anetqa_by_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"anetqa_tmp_{rank}.json")


def agqa_test_wrapup(outs, model_name):
    rank = torch.distributed.get_rank()
    qids, preds, preds10 = list(), list(), list()
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]

    rets = [{"qid": qid, "answer": pred} \
            for qid, pred in zip(qids, preds)]
    with open(f"agqa_tmp_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("agqa_tmp_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result", exist_ok=True)
        with open(f"result/agqa_by_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"agqa_tmp_{rank}.json")
