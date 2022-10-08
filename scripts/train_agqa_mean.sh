CKPT=ckpt/vqa.pth

python run.py with \
    data_root=data/agqa \
    num_gpus=2 \
    num_nodes=1 \
    load_path=$CKPT
    task_finetune_agqa_mean \
    per_gpu_batchsize=32 \
