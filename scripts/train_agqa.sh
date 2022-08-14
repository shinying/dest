CKPT=ckpt/trm.ckpt

python run.py with \
    data_root=data/agqa \
    num_gpus=2 \
    num_nodes=1 \
    task_finetune_agqa \
    batch_size=64 \
    per_gpu_batchsize=8 \
    max_epoch=6 \
    load_path=$CKPT \
    # clip16 text_bert
