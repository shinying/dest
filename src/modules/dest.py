import copy
from functools import partial

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import DistilBertModel
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .vit import VisionTransformer, interpolate_pos_embed
from .xbert import BertConfig, BertModel
from . import heads
from . import objectives
from . import utils


class DeST(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.tasks = config["loss_names"]
        assert config["load_path"], "no checkpoints"

        self.visual_encoder = VisionTransformer(
            img_size=config["image_size"], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        bert_config = BertConfig(**config["bert_config"])
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased", config=bert_config,
                                                      add_pooling_layer=False)


        # ==================== Checkpoint for Pretrain =================== #

        if self.tasks["trm"] and not config["test_only"]:
            ckpt = torch.load(config["load_path"], map_location='cpu')
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            for key in list(state_dict.keys()):
                if 'bert' in key:
                    new_key = key.replace('bert.', '')
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

            msg = self.load_state_dict(state_dict, strict=False)
            print(">>> Load checkpoint for pretrain from", config["load_path"])
            utils.parse_loading_msg(msg)


        # ==================== Video Encoding =================== #

        if self.tasks["trm"] or self.tasks["anetqa"] or self.tasks["agqa"]:
            D = self.text_encoder.config.hidden_size
            with torch.no_grad():
                pos_embed = self.text_encoder.embeddings.position_embeddings.weight.data[:config["max_pos_len"]+2]
            self.temporal_embedding = TemporalEmbedding(config["input_video_embed_size"],
                    D, pos_embed, config["max_pos_len"], config["drop_rate"])

            self.temporal_encoder = copy.deepcopy(self.text_encoder)
            del self.temporal_encoder.embeddings

        # ==================== Pretarin-specific Modules =================== #

        D = self.text_encoder.config.hidden_size

        if self.tasks["trm"]:
            self.vision_proj = nn.Linear(D, D//3)
            self.vision_proj.apply(objectives.init_weights)

            self.text_proj = nn.Linear(D, D//3)
            self.text_proj.apply(objectives.init_weights)

            self.tau = nn.Parameter(torch.tensor(0.07))

            self.ans_encoder = AnsEncoder(config["hidden_size"]//2)
            self.video_head = nn.Sequential(
                    nn.Linear(D, D),
                    nn.ReLU(),
                    nn.Linear(D, D//2))
            self.video_head.apply(objectives.init_weights)

        # ==================== Checkpoint for Downstream =================== #

        if not self.tasks["trm"] and not config["test_only"]:
            ckpt = torch.load(config["load_path"], map_location='cpu')
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            msg = self.load_state_dict(state_dict, strict=False)
            print(">>> Load checkpoint for downstream from", config["load_path"])
            utils.parse_loading_msg(msg)

        # ==================== Task-specific Modules =================== #

        if not self.tasks["trm"]:
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
            print(">>> Load checkpoint for inference from", config["load_path"])

        self.freeze_weights()
        self.set_forward(config)
        utils.set_metrics(self)

    def freeze_weights(self):
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.embeddings.parameters():
            param.requires_grad = False

    def encode_question(self, batch):
        text_output = self.text_encoder(batch["questions"].input_ids,
                                        attention_mask=batch["questions"].attention_mask,
                                        mode="text")
        return text_output["last_hidden_state"]

    def encode_text(self, batch):
        text_output = self.text_encoder(batch["matching"].input_ids,
                                        attention_mask=batch["matching"].attention_mask,
                                        mode="text")
        return text_output["last_hidden_state"]

    def encode_frames(self, batch, question_hidden_states):
        frame_embeds = self.visual_encoder(batch["frames"])
        frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(frame_embeds.device)

        B = frame_embeds.size(0)
        _, S, C = question_hidden_states.size()
        exp_text_hidden_states = question_hidden_states.unsqueeze(1).expand(-1, batch["nframe"], -1, -1).reshape(B, S, C)
        exp_text_mask = batch["questions"]["attention_mask"].unsqueeze(1).expand(-1, batch["nframe"], -1).reshape(B, S)

        frame_output = self.text_encoder(encoder_embeds=exp_text_hidden_states,
                                         attention_mask=exp_text_mask,
                                         encoder_hidden_states=frame_embeds,
                                         encoder_attention_mask=frame_atts,
                                         mode="fusion",
                                         return_dict=True)
        frame_feat = frame_output[0][:, 0]
        frame_feat = self.frame_head(frame_feat)
        frame_feat = frame_feat.view(batch["labels"].size(0), batch["nframe"], -1).mean(dim=1)
        return frame_feat

    def encode_video(self, batch, text_hidden_states=None, text_mask=None, return_hidden=True):
        video_embed = self.temporal_embedding(batch)
        video_embed = self.temporal_encoder(encoder_embeds=video_embed,
                                            attention_mask=batch["video_mask"],
                                            mode="text")[0]
        if text_hidden_states is None:
            return video_embed

        video_output = self.temporal_encoder(encoder_embeds=text_hidden_states,
                                             attention_mask=text_mask,
                                             encoder_hidden_states=video_embed,
                                             encoder_attention_mask=batch["video_mask"],
                                             mode="fusion")
        video_feat = video_output[0][:, 0]
        video_feat = self.video_head(video_feat)

        if return_hidden:
            return video_embed, video_feat
        return video_feat

    def encode_clip(self, batch, video_embed):
        seq = batch["video_mask"].sum(dim=1)
        embeds = []
        choice_id = []
        choice_mask = []
        for i, embed in enumerate(video_embed):
            embed = embed[:seq[i]][1:-1]
            means = [embed[sli].mean(dim=0) for sli in batch["lens"][i]]
            embeds += means

        clip_feat = torch.stack(embeds)
        return clip_feat


    def set_forward(self, config):
        if self.tasks["trm"]:
            self.forward_task = objectives.compute_trm
            self.test_forward = objectives.trm_test_step
            self.test_wrapup = objectives.trm_test_wrapup
        elif self.tasks["anetqa"]:
            self.forward_task = objectives.compute_anetqa
            self.test_forward = objectives.anetqa_test_step
            self.test_wrapup = objectives.anetqa_test_wrapup
        elif self.tasks["anetqa_mean"]:
            self.forward_task = objectives.compute_anetqa_mean
            self.test_forward = objectives.anetqa_test_step
            self.test_wrapup = objectives.anetqa_test_wrapup
        elif self.tasks["agqa"]:
            self.forward_task = objectives.compute_agqa
            self.test_forward = objectives.agqa_test_step
            self.test_wrapup = objectives.agqa_test_wrapup
        elif self.tasks["agqa_mean"]:
            self.forward_task = objectives.compute_agqa_mean
            self.test_forward = objectives.agqa_test_step
            self.test_wrapup = objectives.agqa_test_wrapup
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
        return self.test_forward(self, batch, output)

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]
        self.test_wrapup(outs, model_name)
        utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return utils.set_schedule(self)


class TemporalEmbedding(nn.Module):

    def __init__(self, input_size, embed_size, position_embedding_data, max_pos_len=100, dropout=0):
        super().__init__()
        self.proj = nn.Linear(input_size, embed_size)
        self.bos = nn.Parameter(torch.empty(embed_size))
        self.eos = nn.Parameter(torch.empty(embed_size))

        # Frame positional embedding
        self.register_buffer("position_ids", torch.arange(max_pos_len+2).expand(1, -1))
        self.frame_pos_embed = nn.Embedding(max_pos_len+2, embed_size, _weight=position_embedding_data)
        self.ln = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

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
        video_embed[torch.arange(B), ends] = self.eos

        pos_ids = self.position_ids[:, :video_embed.size(1)]
        video_embed += self.frame_pos_embed(pos_ids)
        video_embed = self.ln(video_embed)
        video_embed = self.dropout(video_embed)

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

