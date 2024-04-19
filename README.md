# Learning Fine-Grained Visual Understanding <br> for Video Question Answering <br> via Decoupling Spatial-Temporal Modeling

**BMVC 2022**

[Project Page](https://shinying.github.io/dest) | [arXiv](https://arxiv.org/abs/2311.18832)

## Installation

Python 3.7 and CUDA 11.1 

```
pip install -r requirements.txt
```

## Checkpoints

Checkpoints of pre-training (trm), ActivityNet-QA (anetqa), and AGQA (agqa) can be downloaded [here](https://drive.google.com/drive/folders/1NpJyCZf-5kVIeB6yTheNtNsHLbw4U5yp?usp=sharing). \
Image-language pre-training weights are from [ALBEF](https://github.com/salesforce/ALBEF).

## Preprocess

Input data is organized as follows:

```
data/
├── vatex/
│   ├── train.json
│   ├── val.json
│   └── vatex.h5
├── tgif/
│   ├── train.json
│   ├── val.json
│   └── tgif.h5
├── anetqa/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── vocab.json
│   ├── frames/
│   └── anetqa.h5
└── agqa/
    ├── train.json
    ├── val.json
    ├── test.json
    ├── vocab.json
    ├── frames/
    └── agqa.h5
```
For AGQA, `train_balanced.txt` and `test_balanced.txt` are renamed as `train.json` and `test.json`. \
We randomly sample 10% data from `train.json` for validation in `val.json`. 

### Annotations

Following [Just-Ask](https://github.com/antoyang/just-ask), we remove rare answers of ActivityNet and AGQA. 

```sh
python preproc/preproc_anetqa.py -i INPUT_DIR [-o OUTPUT_DIR]
python preproc/preproc_agqa.py -i INPUT_DIR [-o OUTPUT_DIR]
```

`INPUT_DIR` contains annotation files. `OUTPUT_DIR` is the same as `INPUT_DIR` if not specified. \
For example,

```
python preproc/preproc_anetqa.py -i activitynet-qa/dataset data/anetqa
```

### Frames

We extract frames at 3 FPS with [FFmpeg](https://ffmpeg.org/),

```sh
ffmpeg -i VIDEO_PATH -vf fps=3 VIDEO_ID/%04d.png
```

The frames of a video are collected in a directory. \
For example, the content of `data/anetqa/frames/` is

```
data/anetqa/frames
├── v_PLqTX6ij52U/
│   ├── 0001.png
│   ├── 0002.png
│   ├── ...
├── v_d_A-ylxNbFU/
│   ├── 0001.png
│   ├── ...
├── ...

```

To process all videos in parallel,

```sh
find VIDEO_DIR -type f | parallel -j8 "mkdir frames/{/.} && ffmpeg -i {} -vf fps=3 frames/{/.}/%04d.png"
```

### Video Features

We extract video features with [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer) (Swin-B on Kinetics-600). \
Please see the detailed instruction [here](https://github.com/shinying/Video-Swin-Transformer). \
The features are gathered in an hdf5 file. Please rename and put it under the dataset directory.


## Pre-training with Temporal Referring Modeling

```sh
# Training
python run.py with \
    data_root=data \
    num_gpus=1 \
    num_nodes=1 \
    task_pretrain_trm \
    per_gpu_batchsize=32 \
    load_path=/path/to/vqa.pth

# Inference
python run.py with \
    data_root=data \
    num_gpus=1 \
    num_nodes=1 \
    task_pretrain_trm \
    per_gpu_batchsize=32 \
    load_path=/path/to/trm.ckpt \
    test_only=True
```

## Downstream

### ActivityNet-QA

```sh
# Training
python run.py with \
    data_root=data/anetqa \
    num_gpus=1 \
    num_nodes=1 \
    load_path=/path/to/trm.ckpt \
    task_finetune_anetqa \
    per_gpu_batchsize=8

# Inference
python run.py with \
    data_root=data/anetqa \
    num_gpus=1 \
    num_nodes=1 \
    load_path=/path/to/anetqa.ckpt \
    task_finetune_anetqa \
    per_gpu_batchsize=16 \
    test_only=True
```

### AGQA

```sh
# Training
python run.py with \
    data_root=data/agqa \
    num_gpus=1 \
    num_nodes=1 \
    load_path=/path/to/trm.ckpt \
    task_finetune_agqa \
    per_gpu_batchsize=16

# Inference
python run.py with \
    data_root=data/agqa \
    num_gpus=1 \
    num_nodes=1 \
    load_path=/path/to/agqa.ckpt \
    task_finetune_agqa \
    per_gpu_batchsize=32 \
    test_only=True
```

### Evaluation

```sh
python eval.py {anet, agqa} PREDICTION GROUNDTRUTH
```

For example,

```sh
python eval.py anet result/anetqa_by_anetqa.json data/anetqa/test.csv
python eval.py agqa result/agqa_by_agqa.json data/agqa/test.json
```

## Preliminary Analysis with ALBEF

```sh
bash scripts/train_anetqa_mean.sh
bash scripts/train_agqa_mean.sh
```

## Citation

```BibTeX
@InProceedings{lee2022learning,
  author = {Lee, Hsin-Ying and Su, Hung-Ting and Tsai, Bing-Chen and 
            Wu, Tsung-Han and Yeh, Jia-Fong and Hsu, Winston H.},
  title = {{Learning Fine-Grained Visual Understanding for Video Question Answering 
            via Decoupling Spatial-Temporal Modeling}},
  booktitle = {British Machine Vision Conference},
  year = {2022}
}
```

## Acknowledgements

The code is based on [METER](https://github.com/zdou0830/METER) and [ALBEF](https://github.com/salesforce/ALBEF).

