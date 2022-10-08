if [ -z $CKPT ]; then
    echo "Error: set checkpoint path to CKPT"
    exit 1
fi

python run.py with \
    data_root=data/anetqa \
    num_gpus=1 \
    num_nodes=1 \
    load_path=$CKPT \
    task_finetune_anetqa_mean \
    per_gpu_batchsize=8 \
    test_only=True
