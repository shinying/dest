import copy
from functools import partial

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import DistilBertModel, T5Model
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .vit import VisionTransformer, interpolate_pos_embed
from .xbert import BertConfig, BertModel
from . import heads
from . import objectives
from . import utils


class ALBEF(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.tasks = config["loss_names"]
        assert len(config["load_path"]), "no checkpoints"

        self.visual_encoder = VisionTransformer(
            img_size=config["image_size"], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        bert_config = BertConfig(**config["bert_config"])
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased", config=bert_config,
                                                      add_pooling_layer=False)


        # ==================== Checkpoint for Pretrain =================== #

        if self.tasks["tem"] and not config["test_only"]:
            ckpt = torch.load(config["load_path"], map_location='cpu')
            state_dict = ckpt
            for key in list(state_dict.keys()):
                if 'bert' in key:
                    new_key = key.replace('bert.', '')
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

            msg = self.load_state_dict(state_dict, strict=False)
            print("Load checkpoint from", config["load_path"])
            print(msg.missing_keys)


        # ==================== Video Encoding =================== #

        D = self.text_encoder.config.hidden_size
        with torch.no_grad():
            pos_embed = self.text_encoder.embeddings.position_embeddings.weight.data[:config["max_video_len"]+2]
        self.temporal_embedding = TemporalEmbedding(config["input_video_embed_size"],
                D, pos_embed, config["max_video_len"], config["drop_rate"])

        t5 = T5Model.from_pretrained("google/t5-v1_1-base").encoder
        t5.block = t5.block[:self.text_encoder.config.fusion_layer]
        self.video_encoder = t5

        te = copy.deepcopy(self.text_encoder)
        te.encoder.layer = te.encoder.layer[self.text_encoder.config.fusion_layer:]
        te.config.num_hidden_layers = len(te.encoder.layer)
        del te.embeddings
        self.temporal_encoder = te


        # ==================== Checkpoint for Downstream =================== #

        if not self.tasks["tem"] and not config["test_only"]:
            ckpt = torch.load(config["load_path"], map_location='cpu')
            state_dict = ckpt['state_dict']
            self.load_state_dict(state_dict)
            print("Load checkpoint from", config["load_path"])


        # ==================== Task-specific Modules =================== #

        if self.tasks["tem"]:
            D = self.text_encoder.config.hidden_size
            if "haa" in config["datasets"]:
                self.tem_head = nn.Sequential(
                        nn.Linear(D, 500),
                        nn.ReLU(),
                        nn.Linear(D, 500))
                self.tem_head.apply(objectives.init_weights)

                self.action_head = nn.Sequential(
                        nn.Linear(D, D),
                        nn.ReLU(),
                        nn.Linear(D, 500))
                self.action_head.apply(objectives.init_weights)
            else:
                self.video_head = nn.Sequential(
                        nn.Linear(D, D),
                        nn.ReLU(),
                        nn.Linear(D, D//2))
                self.video_head.apply(objectives.init_weights)

                self.vision_proj = nn.Linear(D, D//3)
                self.vision_proj.apply(objectives.init_weights)

                self.text_proj = nn.Linear(D, D//3)
                self.text_proj.apply(objectives.init_weights)

                self.tau = nn.Parameter(torch.tensor(0.07))
                self.ans_encoder = AnsEncoder(config["hidden_size"]//2)

        else:
            self.frame_head = nn.Sequential(
                    nn.Linear(D, D),
                    nn.ReLU(),
                    nn.Linear(D, D))
            self.frame_head.apply(objectives.init_weights)

            self.video_head = nn.Sequential(
                    nn.Linear(D, D),
                    nn.ReLU(),
                    nn.Linear(D, D))
            self.video_head.apply(objectives.init_weights)

            self.ans_encoder = AnsEncoder(config["hidden_size"])

        # ==================== Checkpoint for Downstream Inference =================== #

        if config["test_only"]:
            ckpt = torch.load(config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict)
            print("Load checkpoint from", config["load_path"])

        self.freeze_weights()
        self.set_forward(config)
        utils.set_metrics(self)

    def freeze_weights(self):
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.embeddings.parameters():
            param.requires_grad = False

    def infer(self, batch):
        ret = {}
        text_output = self.text_encoder(batch["questions"].input_ids,
                                        attention_mask=batch["questions"].attention_mask,
                                        mode="text")
        text_hidden_states = text_output["last_hidden_state"]

        if not self.tasks["tem"]:
            frame_embeds = self.visual_encoder(batch["frames"])
            frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(frame_embeds.device)

            B = frame_embeds.size(0)
            _, S, C = text_hidden_states.size()
            # text_ids = batch["questions"]["input_ids"].unsqueeze(1).expand(-1, batch["nframe"], -1).reshape(B, -1)
            exp_text_hidden_states = text_hidden_states.unsqueeze(1).expand(-1, batch["nframe"], -1, -1).reshape(B, S, C)
            exp_text_mask = batch["questions"]["attention_mask"].unsqueeze(1).expand(-1, batch["nframe"], -1).reshape(B, S)

            frame_output = self.text_encoder(encoder_embeds=exp_text_hidden_states,
            # frame_output = self.text_encoder(text_ids,
                                             attention_mask=exp_text_mask,
                                             encoder_hidden_states=frame_embeds,
                                             encoder_attention_mask=frame_atts,
                                             # output_hidden_states=True,
                                             mode="fusion",
                                             return_dict=True)
            # frame_feat = self.frame_pooler(frame_output["last_hidden_state"])
            frame_feat = frame_output[0][:, 0]
            # ret["frame_feat"] = frame_feat.view(batch["labels"].size(0), batch["nframe"], -1).mean(dim=1)
            ret["frame_feat"] = frame_feat

        # if self.tasks["nextqa"]:
        #     text_hidden_states = frame_output["hidden_states"][self.text_encoder.config.fusion_layer]
        #     B, S = batch["questions"]["input_ids"].size()
        #     text_hidden_states = text_hidden_states.view(B, batch["nframe"], S, -1).mean(dim=1)

        if self.tasks["tem"] or self.tasks["nextqa"] or self.tasks["anetqa"]:
            video_embed = self.temporal_embedding(batch)

            video_embed = self.video_encoder(inputs_embeds=video_embed,
                                             attention_mask=batch["video_mask"])[0]

            video_output = self.temporal_encoder(encoder_embeds=text_hidden_states,
                                                 attention_mask=batch["questions"].attention_mask,
                                                 encoder_hidden_states=video_embed,
                                                 encoder_attention_mask=batch["video_mask"],
                                                 mode="multi_modal")
            vid_feat = video_output[0][:, 0]
            ret["vid_feat"] = vid_feat

            if self.tasks["tem"]: # video-text matching
                s = batch["video_mask"].sum(dim=1)
                embeds = []
                choice_id = []
                choice_mask = []
                for i, embed in enumerate(video_embed):
                    embed = embed[:s[i]].split(batch["lens"][i])[1:-1] # ignore bos and eos
                    embeds.extend([e.mean(dim=0) for e in embed])

                embeds = torch.stack(embeds)
                ret["clip_feat"] = embeds

                text_feat = self.text_encoder(batch["matching"].input_ids,
                                              attention_mask=batch["matching"].attention_mask,
                                              mode="text")[0][:, 0]
                ret["text_feat"] = text_feat

        return ret

    def set_forward(self, config):
        if self.tasks["tem"]:
            if "haa" in config["datasets"]:
                self.forward_task = objectives.compute_tem_500
            else:
                self.forward_task = objectives.compute_tem
        elif self.tasks["anetqa"]:
            self.forward_task = objectives.compute_anetqa
        elif self.tasks["anetqa_mean"]:
            self.forward_task = objectives.compute_anetqa_mean
        elif self.tasks["nextqa"]:
            self.forward_task = objectives.compute_nextqa
        elif self.tasks["nextqa_mean"]:
            self.forward_task = objectives.compute_nextqa_mean
        else:
            raise ValueError("loss is not set")

    def forward(self, batch):
        return self.forward_task(self, batch)

    def training_step(self, batch, batch_idx):
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        return total_loss

    def training_epoch_end(self, outs):
        utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        output = self(batch)

    def validation_epoch_end(self, outs):
        utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        output = self(batch)
        ret = dict()

        if self.tasks["tem"]:
            ret.update(objectives.tem_test_step(self, batch, output))
        elif self.tasks["anetqa"] or self.tasks["anetqa_mean"]:
            ret.update(objectives.anetqa_test_step(self, batch, output))
        elif self.tasks["nextqa"] or self.tasks["nextqa_mean"]:
            ret.update(objectives.nextqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.tasks["tem"]:
            objectives.tem_test_wrapup(outs, model_name)
        elif self.tasks["anetqa"] or self.tasks["anetqa_mean"]:
            objectives.anetqa_test_wrapup(outs, model_name)
        elif self.tasks["nextqa"] or self.tasks["nextqa_mean"]:
            objectives.nextqa_test_wrapup(outs, model_name)

        utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return utils.set_schedule(self)


class TemporalEmbedding(nn.Module):

    def __init__(self, input_size, embed_size, position_embedding_data, max_video_len=100, dropout=0):
        super().__init__()
        self.proj = nn.Linear(input_size, embed_size)
        self.bos = nn.Parameter(torch.empty(embed_size))
        self.eos = nn.Parameter(torch.empty(embed_size))

        # Frame positional embedding
        # self.register_buffer("position_ids", torch.arange(max_video_len+2).expand(1, -1))
        # self.frame_pos_embed = nn.Embedding(max_video_len+2, embed_size, _weight=position_embedding_data)
        self.ln = nn.LayerNorm(embed_size)
        # self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        self.proj.apply(objectives.init_weights)
        nn.init.trunc_normal_(self.bos, mean=0, std=0.02)
        nn.init.trunc_normal_(self.eos, mean=0, std=0.02)
        self.ln.apply(objectives.init_weights)

    def forward(self, batch):
        video_embed = self.proj(batch["video"])
        B, S, D = video_embed.size()
        video_embed = torch.cat([self.bos.expand(B, 1, -1), video_embed,
                                  torch.zeros(B, 1, D, device=video_embed.device)], dim=1)
        ends = batch["video_mask"].sum(dim=1) - 1
        for i in range(B):
            # end = sum(batch["lens"][i]) - 1
            video_embed[i, ends[i]] = self.eos

        # pos_ids = self.position_ids[:, :video_embed.size(1)]
        # video_embed += self.frame_pos_embed(pos_ids)
        video_embed = self.ln(video_embed)
        # video_embed = self.dropout(video_embed)

        return video_embed


class AnsEncoder(nn.Module):

    def __init__(self, output_size):
        super().__init__()
        self.lm = DistilBertModel.from_pretrained('distilbert-base-uncased')
        D = self.lm.config.dim
        self.pooler = nn.Sequential(
            heads.Pooler(D),
            nn.Linear(D, output_size),
        )
        self.pooler.apply(objectives.init_weights)

        for param in self.lm.embeddings.parameters():
            param.requires_grad = False

    def forward(self, encoding):
        feat = self.lm(**encoding)[0] # last hidden state
        return self.pooler(feat)

