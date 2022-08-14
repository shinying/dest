if [ -z $CKPT ]; then
    echo "Error: set checkpoint path to CKPT"
    exit 1
fi

python run.py with \
    data_root=data/agqa \
    num_gpus=1 \
    num_nodes=1 \
    task_finetune_agqa_mean \
    per_gpu_batchsize=32 \
    load_path=$CKPT \
    test_only=True
    # clip16 text_bert \
