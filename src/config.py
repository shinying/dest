from sacred import Experiment

ex = Experiment("DeST")


def _loss_names(d):
    ret = {
        "trm": 0,
        "anetqa": 0,
        "anetqa_mean": 0,
        "agqa": 0,
        "agqa_mean": 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "trm"
    seed = 0
    datasets = []
    loss_names = _loss_names({})
    batch_size = 64

    # Image setting
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_size = 384
    patch_size = 16
    # draw_false_image = 0
    # image_only = False
    # resolution_before = 224
    nframe = 1
    trim = 1.
    max_video_len = 100
    max_pos_len = 100
    num_clips = 8

    # Text Setting
    aqa_label_size = 1654
    max_text_len = None
    max_ans_len = None
    tokenizer = "bert-base-uncased"
    vocab_size = 30522

    bert_config = {
      "architectures": ["BertForMaskedLM"],
      "attention_probs_dropout_prob": 0.1,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 768,
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "layer_norm_eps": 1e-12,
      "max_position_embeddings": 512,
      "model_type": "bert",
      "num_attention_heads": 12,
      "num_hidden_layers": 12,
      "pad_token_id": 0,
      "type_vocab_size": 2,
      "vocab_size": 30522,
      "fusion_layer": 6,
      "encoder_width": 768
    }

    # Transformer Setting
    # num_top_layer = 6
    # input_image_embed_size = 768
    # input_text_embed_size = 768
    input_video_embed_size = 1024
    # vit = 'ViT-B/16'
    hidden_size = 768
    # num_heads = 12
    # num_layers = 6
    # mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-5
    weight_decay = 0.01
    decay_power = 1
    max_epoch = None
    max_steps = -1
    warmup_steps = 0
    end_lr = 0
    lr_mult_head = 5  # multiply lr for downstream heads
    lr_mult_ans = 5  # multiply lr for the cross-modal module
    lr_time = 1

    # Downstream Setting
    # get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 8
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16


@ex.named_config
def task_pretrain_trm():
    exp_name = "pretrain_trm"
    datasets = ["vatex", "tgif"]
    loss_names = _loss_names({"trm": 1})
    batch_size = 128
    max_epoch = None
    max_steps = 60000
    warmup_steps = 1000
    val_check_interval = 400
    learning_rate = 1e-5
    lr_mult_head = 25
    lr_mult_ans = 2
    lr_time = 5
    max_video_len = 100
    max_text_len = 40
    num_clips = 8


@ex.named_config
def task_finetune_anetqa_mean():
    exp_name = "finetune_anetqa_mean"
    datasets = ["anetqa"]
    loss_names = _loss_names({"anetqa_mean": 1})
    batch_size = 64
    max_epoch = 5
    val_check_interval = 1.
    warmup_steps = 0.1
    learning_rate = 1e-5
    lr_mult_head = 50
    lr_mult_ans = 1
    nframe = 16
    trim = 0.8
    max_video_len = 100
    max_text_len = 25


@ex.named_config
def task_finetune_anetqa():
    exp_name = "finetune_anetqa"
    datasets = ["anetqa"]
    loss_names = _loss_names({"anetqa": 1})
    batch_size = 64
    max_epoch = 5
    val_check_interval = 1.
    warmup_steps = 0.1
    learning_rate = 2e-5
    lr_mult_head = 50
    lr_mult_ans = 1
    lr_time = 10
    nframe = 16
    trim = 0.8
    max_video_len = 100
    max_text_len = 25


@ex.named_config
def task_finetune_agqa_mean():
    exp_name = "finetune_agqa_mean"
    datasets = ["agqa"]
    loss_names = _loss_names({"agqa_mean": 1})
    batch_size = 64
    max_epoch = 6
    val_check_interval = 1.
    warmup_steps = 0.1
    learning_rate = 1e-5
    lr_mult_head = 50
    lr_mult_ans = 2
    nframe = 8


@ex.named_config
def task_finetune_agqa():
    exp_name = "finetune_agqa"
    datasets = ["agqa"]
    loss_names = _loss_names({"agqa": 1})
    batch_size = 64
    max_epoch = 4
    val_check_interval = 1.
    warmup_steps = 0.1
    learning_rate = 2e-5
    lr_mult_head = 10
    lr_mult_ans = 1
    lr_time = 2.5
    nframe = 8
    max_video_len = 40


# vision encoder

@ex.named_config
def clip16():
    vit = 'ViT-B/16'
    image_size = 384 # 224
    patch_size = 16
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768


# text encoder
@ex.named_config
def text_bert():
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    input_text_embed_size = 768
