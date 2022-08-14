import argparse
import csv
import json
import os
import os.path as op
import collections
import random


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='input data directory')
parser.add_argument('-o', '--output', help='output data directory; default is the input directory')
args = parser.parse_args()
if args.output is None:
    args.ouptut = args.input
os.makedirs(args.output, exist_ok=True)


def get_vocabulary(train, save=False):
    ans = [x["answer"] for x in train.values()]
    train_counter = collections.Counter(ans)
    most_common = train_counter.most_common()
    vocab = {x[0]: i for i, x in enumerate(most_common)}
    print('Vocab size:', len(vocab)) # 171
    if save:
        with open(op.join(args.output, "vocab.json"), "w") as outfile:
            json.dump(vocab, outfile)
    return vocab


def train_val_split(train, val_ratio=0.1):
    vids = set(d['video_id'] for d in train.values())
    valvids = set(random.sample(list(vids), k=int(len(vids)*val_ratio)))

    trainsp = {}
    valsp = {}
    for k, v in train.items():
        if v['video_id'] in valvids:
            valsp[k] = v
        else:
            trainsp[k] = v

    return trainsp, valsp


train = json.load(open(op.join(args.input, 'train_balanced.txt')))
test = json.load(open(op.join(args.input, 'test_balanced.txt')))

vocab = get_vocabulary(train, True)
train, val = train_val_split(train)
train = {k: v for k, v in train.items() if v['answer'] in vocab}
print('Size:', len(train), len(val))

json.dump(train, open(op.join(args.output, 'train.json'), 'w'))
json.dump(val, open(op.join(args.output, 'val.json'), 'w'))
os.system(f'cp {op.join(args.input, "test_balanced.txt")} {op.join(args.output, "test.json")}')
