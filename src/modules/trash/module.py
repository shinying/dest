import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

from transformers import DistilBertModel
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings, BertModel, BertEncoder, BertLayer
from .bert_model import BertCrossLayer, BertAttention
from . import swin_transformer as swin
from . import heads, objectives, utils
from .clip_model import build_model, adapt_position_encoding
from .swin_helpers import swin_adapt_position_encoding
from transformers import RobertaConfig, RobertaModel


class METERTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        # ==================== Image Encoder ==================== #

        self.is_clip= (not 'swin' in config['vit'])
        resolution_after=config['image_size']

        if self.is_clip:
            self.vit_model = build_model(config['vit'], resolution_after=resolution_after)
        else:
            self.vit_model = getattr(swin, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config,
            )
            self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.cross_modal_image_transform = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.cross_modal_image_transform.apply(objectives.init_weights)

        # self.temporal_image_transform = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        # self.temporal_image_transform.weight.data = self.cross_modal_image_transform.weight.data
        # self.temporal_image_transform.bias.data = self.cross_modal_image_transform.bias.data


        # if torch.distributed.is_initialized():
        #     if torch.distributed.get_rank() == 0:
        #         if self.is_clip:
        #             build_model(config['vit'], resolution_after=resolution_after)
        #         else:
        #             getattr(swin, self.hparams.config["vit"])(
        #                 pretrained=True, config=self.hparams.config,
        #             )

        #         RobertaModel.from_pretrained(config['tokenizer'])

        #     torch.distributed.barrier()

        self.tasks = config["loss_names"]

        # ==================== Text Encoder ==================== #

        self.text_transformer = RobertaModel.from_pretrained(config['tokenizer'])

        bert_config = RobertaConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.cross_modal_text_transform = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.cross_modal_text_transform.apply(objectives.init_weights)

        # ==================== Cross Modality ==================== #

        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.cross_modal_image_layers.apply(objectives.init_weights)
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.cross_modal_text_layers.apply(objectives.init_weights)

        self.cross_modal_image_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_image_pooler.apply(objectives.init_weights)
        self.cross_modal_text_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_text_pooler.apply(objectives.init_weights)

        # for name, param in self.named_parameters():
        #     if "cross_modal" in name:
        #         param.requires_grad = False

        # if self.hparams.config["loss_names"]["vqa"]:
        #     vs = self.hparams.config["vqav2_label_size"]
        #     self.vqa_classifier = nn.Sequential(
        #         nn.Linear(hs * 2, hs * 2),
        #         nn.LayerNorm(hs * 2),
        #         nn.GELU(),
        #         nn.Linear(hs * 2, vs),
        #     )
        #     self.vqa_classifier.apply(objectives.init_weights)

        for param in self.parameters():
            param.requires_grad = False

        self.time_transformer = TemporalTransformer(config["hidden_size"]*2)

        if self.tasks["tem"]:
            finetune_modules = [
                    # self.temporal_image_transform,
                    # self.text_transformer.embeddings.position_embeddings,
                    # self.text_transformer.encoder,
                    self.cross_modal_text_layers,
                    self.cross_modal_image_layers,
                    self.cross_modal_text_pooler,
                    self.cross_modal_image_pooler,
            ]
        else:
            finetune_modules = (
                    self.temporal_image_transform,
                    self.text_transformer.embeddings.position_embeddings,
                    self.text_transformer.encoder,

                    # self.cross_modal_text_transform,
                    # self.cross_modal_image_transform,
                    # self.cross_modal_text_layers,
                    # self.cross_modal.image_layers,
                    # self.cross_modal_text_pooler,
                    # self.cross_modal_image_pooler,
            )

        for module in finetune_modules:
            for param in module.parameters():
                param.requires_grad = True

        hs = config["hidden_size"]
        if self.tasks["tem"]:
            self.tem_head = nn.Sequential(
                nn.Linear(hs, hs),
                nn.LayerNorm(hs),
                nn.GELU(),
                nn.Linear(hs, 510),
            )
            self.tem_head.apply(objectives.init_weights)
            self.action_head = nn.Sequential(
                    nn.LayerNorm(hs),
                    nn.Dropout(0.5),
                    nn.GELU(),
                    nn.Linear(hs, 500))
            self.action_head.apply(objectives.init_weights)

        # ===================== Downstream ===================== #

        if config["load_path"] != "" and not config["test_only"]:
            ckpt = torch.load(config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if self.is_clip:
                state_dict = adapt_position_encoding(state_dict, after=resolution_after, patch_size=config['patch_size'])
            else:
                state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after, before=config['resolution_before'])

            self.load_state_dict(state_dict, strict=False)


        if self.tasks["anetqa"] or self.tasks["anetqa_mean"] or self.tasks["nextqa"] or self.tasks["nextqa_mean"]:
            self.qa_head = nn.Sequential(
                    nn.Linear(hs*2, hs*2),
                    nn.LayerNorm(hs),
                    nn.GELU(),
                    nn.Linear(hs*2, hs))
            self.qa_head.apply(objectives.init_weights)

            self.tem_head = nn.Sequential(
                    nn.Linear(hs, hs),
                    nn.LayerNorm(hs),
                    nn.GELU(),
                    nn.Linear(hs, hs))
            self.tem_head.apply(objectives.init_weights)

            self.ans_model = AnsModel(hs)

        utils.set_metrics(self)
        # self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if config["load_path"] != "" and config["test_only"]:
            ckpt = torch.load(config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if self.is_clip:
                state_dict = adapt_position_encoding(state_dict, after=resolution_after, patch_size=config['patch_size'])
            else:
                state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after, before=config['resolution_before'])
            self.load_state_dict(state_dict, strict=False)

    def infer(self, batch, image_token_type_idx=1):

        # ================== Vision ==================== #

        image_embeds = self.vit_model(batch["image"])
        image_cls_raw = image_embeds[:, 0]
        image_embeds = self.cross_modal_image_transform(image_embeds)

        # image_cls_raw = self.temporal_image_transform(image_cls_raw)
        # image_cls = image_cls_raw.view(batch["labels"].size(0), batch["nframe"], -1)
        # image_cls_masks = torch.ones((image_cls.size(0), image_cls.size(1)), dtype=torch.long, device=image_cls.device)

        # ================== Text ==================== #

        text_ids = batch["text_ids"]
        text_masks = batch[f"text_masks"]

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        device = text_embeds.device
        input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, device)

        # text_embeds = self.text_transformer.embeddings.word_embeddings(text_ids)

        # ================== Temporal Model ==================== #

        # cross_feats = torch.cat([text_embeds, image_cls], dim=1)
        # cross_feats = self.text_transformer.embeddings(inputs_embeds=cross_feats)

        # cross_masks = torch.cat([text_masks, image_cls_masks], dim=1)
        # cross_extend_masks = self.text_transformer.get_extended_attention_mask(cross_masks, cross_feats.shape, cross_feats.device)

        for layer in self.text_transformer.encoder.layer:
            # cross_feats = layer(cross_feats, cross_extend_masks)[0]
            text_embeds = layer(text_embeds, extend_text_masks)[0]
        # cls_tem = cross_feats[:, 0]

        # ================== Descriptive Model ==================== #

        # text_embeds = self.cross_modal_text_transform(cross_feats[:, :text_ids.size(1)])
        text_embeds = self.cross_modal_text_transform(text_embeds)
        text_embeds += self.token_type_embeddings(torch.zeros_like(text_masks))

        # image_embeds = image_embeds[::2]
        image_masks = torch.ones((image_embeds.size(0), image_embeds.size(1)), dtype=torch.long, device=image_embeds.device)
        extend_image_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.size(), image_embeds.device)
        image_embeds += self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx))

        # batch["nframe"] = batch["nframe"] // 2
        text_embeds = text_embeds.unsqueeze(1).expand(-1, batch["nframe"], -1, -1).reshape(image_embeds.size(0), text_ids.size(1), -1)
        text_masks = text_masks.unsqueeze(1).expand(-1, batch["nframe"], -1).reshape(image_embeds.size(0), text_masks.size(1))
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, text_masks.size(), text_masks.device)

        x, y = text_embeds, image_embeds
        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
            y1 = image_layer(y, x, extend_image_masks, extend_text_masks)
            x, y = x1[0], y1[0]

        text_feats, image_feats = x, y
        cls_feats_text = self.cross_modal_text_pooler(x)
        if self.is_clip:
            cls_feats_image = self.cross_modal_image_pooler(y)
        else:
            avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
            cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)

        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)

        cls_feats = cls_feats.view(batch["labels"].size(0), batch["nframe"], -1)
        cls_tem = self.time_transformer(cls_feats)

        ret = {
            # "text_feats": text_feats,
            # "image_feats": image_feats,
            "image_cls": image_cls_raw,
            # "cls_feats": cls_feats,
            "cls_tem": cls_tem,
            # "text_labels": text_labels,
            # "text_ids": text_ids,
            # "text_masks": text_masks,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        # if len(self.current_tasks) == 0:
        #     ret.update(self.infer(batch))
        #     return ret

        # Visual Question Answering
        # if "vqa" in self.current_tasks:
        # if self.tasks["vqa"]:
        #     ret.update(objectives.compute_vqa(self, batch))
        if self.tasks["tem"]:
            ret.update(objectives.compute_tem_500(self, batch))

        # Video Question Answering
        # elif "anetqa_mean" in self.current_tasks:
        elif self.tasks["anetqa"]:
            ret.update(objectives.compute_anetqa(self, batch))
        elif self.tasks["anetqa_mean"]:
            ret.update(objectives.compute_anetqa_mean(self, batch))
        elif self.tasks["nextqa"]:
            ret.update(objectives.compute_nextqa(self, batch))
        elif self.tasks["nextqa_mean"]:
            ret.update(objectives.compute_nextqa_mean(self, batch))
        else:
            ret.update(self.infer(batch))

        return ret

    def training_step(self, batch, batch_idx):
        # utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        # utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        # utils.set_task(self)
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


class TemporalTransformer(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.model = RobertaModel.from_pretrained('roberta-base')
        self.model.encoder.layer = nn.ModuleList([self.model.encoder.layer[i] for i in range(6, 12)])
        for module in (self.model.embeddings.word_embeddings, self.model.embeddings.token_type_embeddings):
            for param in module.parameters():
                param.requires_grad = False

        self.temporal_proj = nn.Linear(hidden_size, self.model.config.hidden_size)

        with torch.no_grad():
            ts = self.model.embeddings.word_embeddings(torch.tensor([[self.model.config.bos_token_id]]))
        self.cls_token = nn.Parameter(ts)

    def forward(self, x):
        x = self.temporal_proj(x)
        resid = x.mean(dim=1)
        x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim=1)
        return resid + self.model(inputs_embeds=x)[1] # pooler output

    def forward_grid(self, x, bs, nframe):
        resid = x

        # B, S, C --> bs, nframe, S, C --> bs, S, nframe, C --> bs*S, nframe, C
        B, S, C = x.size()
        x = x.view(bs, nframe, S, C).permute(0, 2, 1, 3).reshape(bs*S, nframe, C)
        x = self._forward_impl(x)
        x = x.view(bs, S, nframe, C).permute(0, 2, 1, 3) + resid.view(bs, nframe, S, C)
        return self.ln(x.mean(dim=1))

    def forward_cls(self, vid, vid_mask, text, text_mask, bs, nframe):
        B, S, C = vid.size()
        vid = vid.view(bs, nframe, S, C)[:, :, 0] # bs, nframe, C
        x = torch.cat([self.cls_token.expand(bs, -1, -1), x], dim=1) # bs, nframe+1, C

        vid = self._forward_impl(vid)
        return vid


class AnsModel(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        self.lm = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pooler = nn.Sequential(
            heads.Pooler(self.lm.config.dim),
            nn.Linear(self.lm.config.dim, output_dim),
        )
        self.pooler.apply(objectives.init_weights)

        for param in self.lm.embeddings.parameters():
            param.requires_grad = False

    def forward(self, encoding):
        feat = self.lm(**encoding)[0] # last hidden state
        return self.pooler(feat)
