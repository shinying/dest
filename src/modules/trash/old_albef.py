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


class ALBEF(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.visual_encoder = VisionTransformer(
            img_size=config["image_size"], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        bert_config = BertConfig(**config["bert_config"])
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased", config=bert_config,
                                                      add_pooling_layer=False)

        D = self.text_encoder.config.hidden_size
        # self.frame_pooler = heads.Pooler(D)
        # self.frame_pooler.apply(objectives.init_weights)

        # ==================== Video Feature Encoding =================== #

        # self.temporal_transform = nn.Linear(config["input_video_embed_size"], D)
        # self.temporal_transform.apply(objectives.init_weights)
        self.temporal_transform = nn.GRU(input_size=config["input_video_embed_size"],
                hidden_size=D, num_layers=2, batch_first=True,
                dropout=config["drop_rate"], bidirectional=True)
        self.temporal_proj = nn.Linear(D*2, D)

        # Frame positional embedding
        # self.register_buffer("position_ids", torch.arange(config["max_video_len"]+1).expand(1, -1))
        # with torch.no_grad():
        #     e = self.text_encoder.embeddings.position_embeddings.weight.data[:config["max_video_len"]+1]
        # self.frame_pos_embed = nn.Embedding(config["max_video_len"]+1, D, _weight=e)
        self.ln = nn.LayerNorm(D)
        # self.dropout = nn.Dropout(config["drop_rate"])

        self.temporal_encoder = copy.deepcopy(self.text_encoder)
        del self.temporal_encoder.embeddings
        c = self.temporal_encoder.config
        self.temporal_encoder.encoder.layer = self.temporal_encoder.encoder.layer[c.fusion_layer:]
        c.num_hidden_layers = c.fusion_layer
        c.fusion_layer = 0

        # self.video_pooler = heads.Pooler(D)
        # self.video_pooler.apply(objectives.init_weights)

        self.tasks = config["loss_names"]

        # ==================== Downstreams =================== #

        if config["load_path"] != "" and not config["test_only"]:
            checkpoint = torch.load(config["load_path"], map_location='cpu')
            if 'model' in checkpoint: # ALBEF checkpoint
                state_dict = checkpoint['model']

                # reshape positional embedding to accomodate for image resolution change
                pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], self.visual_encoder)
                state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
            elif 'state_dict' in checkpoint: # METER checkpoint
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            if self.tasks["tem"]: # FIXME
                for key in list(state_dict.keys()):
                    if 'bert' in key:
                        new_key = key.replace('bert.', '')
                        state_dict[new_key] = state_dict[key]
                        del state_dict[key]

                c = self.text_encoder.config
                for key in list(state_dict.keys()):
                    if 'text_encoder.encoder.layer' in key:
                        i = int(key.split('.')[3])
                        if i >= c.fusion_layer:
                            new_key = key.replace(f'text_encoder.encoder.layer.{i}',
                                    f'temporal_encoder.encoder.layer.{i-c.fusion_layer}')
                        state_dict[new_key] = state_dict[key]
                    elif 'text_encoder' in key:
                        new_key = key.replace('text_encoder', 'temporal_encoder')
                        state_dict[new_key] = state_dict[key]

            msg = self.load_state_dict(state_dict, strict=False)
            print("load checkpoint from", config["load_path"])
            print(msg.missing_keys)


        if self.tasks["tem"]:
            D = self.text_encoder.config.hidden_size
            if "haa" in config["datasets"]:
                self.tem_head = nn.Sequential(
                        nn.Linear(D, 500),
                        nn.ReLU(),
                        nn.Linear(D, 500))
                self.tem_head.apply(objectives.init_weights)

                D = self.temporal_encoder.config.hidden_size
                self.action_head = nn.Sequential(
                        nn.Linear(D, D),
                        nn.ReLU(),
                        nn.Linear(D, 500))
                self.action_head.apply(objectives.init_weights)
            else:
                self.video_head = nn.Sequential(
                        nn.Linear(D, D),
                        nn.ReLU(),
                        nn.Linear(D, D))
                self.video_head.apply(objectives.init_weights)
                self.ans_encoder = AnsEncoder(config["hidden_size"])

        else:
            self.frame_head = nn.Sequential(
                    nn.Linear(D, D),
                    nn.ReLU(),
                    nn.Linear(D, D))
            # self.frame_head = nn.Linear(D, D)
            self.frame_head.apply(objectives.init_weights)

            self.video_head = nn.Sequential(
                    nn.Linear(D, D),
                    nn.ReLU(),
                    nn.Linear(D, D))
            # self.video_head = nn.Linear(D, D)
            self.video_head.apply(objectives.init_weights)

            self.ans_encoder = AnsEncoder(config["hidden_size"])

        # ==================== load downstreams =================== #

        if config["load_path"] != "" and config["test_only"]:
            ckpt = torch.load(config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict)

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
        text_output = self.text_encoder(batch["questions"]["input_ids"],
                                        attention_mask=batch["questions"]["attention_mask"],
                                        mode='text')
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
            video = pack_padded_sequence(batch["video"], batch["video_mask"].sum(dim=1).cpu(),
                    batch_first=True, enforce_sorted=False)
            video, _ = self.temporal_transform(video)
            video, _ = pad_packed_sequence(video, batch_first=True, total_length=batch["video_mask"].size(1))
            video = self.temporal_proj(video)
            video = self.ln(video)

            # video_embeds = self.temporal_transform(batch["video"])
            # pos_ids = self.position_ids[:, :video_embeds.size(1)]
            # video_embeds += self.frame_pos_embed(pos_ids)
            # video_embeds = self.ln(video_embeds)
            # video_embeds = self.dropout(video_embeds)

            video_output = self.temporal_encoder(encoder_embeds=text_hidden_states,
                                                 attention_mask=batch["questions"]["attention_mask"],
                                                 encoder_hidden_states=video,
                                                 encoder_attention_mask=batch["video_mask"])
            vid_feat = video_output[0][:, 0]
            # vid_feat = self.video_pooler(video_output["last_hidden_state"])
            ret["vid_feat"] = vid_feat

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

        if self.tasks["vqa"]:
            ret.update(objectives.vqa_test_step(self, batch, output))
        elif self.tasks["anetqa"] or self.tasks["anetqa_mean"]:
            ret.update(objectives.anetqa_test_step(self, batch, output))
        elif self.tasks["nextqa"] or self.tasks["nextqa_mean"]:
            ret.update(objectives.nextqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.tasks["vqa"]:
            objectives.vqa_test_wrapup(outs, model_name)
        elif self.tasks["anetqa"] or self.tasks["anetqa_mean"]:
            objectives.anetqa_test_wrapup(outs, model_name)
        elif self.tasks["nextqa"] or self.tasks["nextqa_mean"]:
            objectives.nextqa_test_wrapup(outs, model_name)

        utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return utils.set_schedule(self)


class AnsEncoder(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        self.lm = DistilBertModel.from_pretrained('distilbert-base-uncased')
        D = self.lm.config.dim
        self.pooler = nn.Sequential(
            heads.Pooler(D),
            nn.Linear(D, output_dim),
        )
        self.pooler.apply(objectives.init_weights)

        for param in self.lm.embeddings.parameters():
            param.requires_grad = False

    def forward(self, encoding):
        feat = self.lm(**encoding)[0] # last hidden state
        return self.pooler(feat)

