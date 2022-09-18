CKPT=ckpt/trm.ckpt

python run.py with \
    data_root=data/agqa \
    num_gpus=2 \
    num_nodes=1 \
    load_path=$CKPT \
    task_finetune_agqa \
    per_gpu_batchsize=8 \
    nframe=16
    # clip16 text_bert
