CKPT=ckpt/vqa.pth

python run.py with \
    data_root=data/anetqa \
    num_gpus=2 \
    num_nodes=1 \
    load_path=$CKPT
    task_finetune_anetqa_mean \
    per_gpu_batchsize=8 \
