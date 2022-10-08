if [ -z $CKPT ]; then
    echo "Error: set checkpoint path to CKPT"
    exit 1
fi

python run.py with \
    data_root=data/anetqa \
    num_gpus=2 \
    num_nodes=1 \
    load_path=$CKPT \
    task_finetune_anetqa \
    per_gpu_batchsize=16 \
    test_only=True
