CKPT=ckpt/vqa.pth

python run.py with \
    data_root=data/anetqa \
    num_gpus=2 \
    num_nodes=1 \
    task_finetune_anetqa_mean \
    batch_size=64  \
    per_gpu_batchsize=8 \
    load_path=$CKPT
    # clip16 text_bert
