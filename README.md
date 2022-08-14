# Learning Fine-Grained Visual Understanding <br> for Video Question Answering <br> via Decoupling Spatial-Temporal Modeling


## Installation

Python 3.7 and CUDA 11.1 

```
pip install -r requirements.txt
```

## Checkpoints

Checkpoints of pre-training (trm), ActivityNet-QA (anetqa), and AGQA (agqa) can be downloaded from [Drive](https://drive.google.com/drive/folders/1NpJyCZf-5kVIeB6yTheNtNsHLbw4U5yp?usp=sharing). \
Image-language pre-training weights are from [ALBEF](https://github.com/salesforce/ALBEF).

## Preprocess

Input data is organized as follows:

```
data/
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
The features are gathered in an h5py file. Please rename and put it under the dataset directory.


## Pre-training with Temporal Referring Modeling

```sh
bash scripts/train_trm.sh # training
CKPT=PATH/TO/trm.ckpt bash scripts/infer_trm.sh # inference
```

## Downstream

### ActivityNet-QA

```sh
bash scripts/train_anetqa.sh # training
CKPT=PATH/TO/anetqa.ckpt bash scripts/infer_anetqa.sh # inference
```

### AGQA

```sh
bash scripts/train_agqa.sh # training
CKPT=PATH/TO/agqa.ckpt bash scripts/infer_agqa.sh # inference
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


## Acknowledgements

The code is based on [METER](https://github.com/zdou0830/METER) and [ALBEF](https://github.com/salesforce/ALBEF).

