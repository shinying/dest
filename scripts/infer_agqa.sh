if [ -z $CKPT ]; then
    echo "Error: set checkpoint path to CKPT"
    exit 1
fi

python run.py with \
    data_root=data/agqa \
    num_gpus=2 \
    num_nodes=1 \
    load_path=$CKPT \
    task_finetune_agqa \
    per_gpu_batchsize=32 \
    test_only=True
