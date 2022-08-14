python run.py with \
    data_root=data \
    num_gpus=1 \
    num_nodes=1 \
    task_pretrain_trm \
    max_steps=60000 \
    batch_size=128 \
    per_gpu_batchsize=32 \
    load_path=ckpt/vqa.pth \
    # clip16 text_bert
