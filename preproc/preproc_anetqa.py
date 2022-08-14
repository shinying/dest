import argparse
import collections
import csv
import json
import os
import os.path as op


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='input data directory')
parser.add_argument('-o', '--output', help='output data directory; default is the input directory')
args = parser.parse_args()
if args.output is None:
    args.output = args.input
os.makedirs(args.output, exist_ok=True)


def get_vocabulary(train_a, save=False):
    ans = [x["answer"] for x in train_a]
    train_counter = collections.Counter(ans)
    most_common = train_counter.most_common()
    vocab = {}
    for i, x in enumerate(most_common):  # 1654 answers present twice
        if x[1] >= 2:
            vocab[x[0]] = i
        else:
            break
    print('Vocab size:', len(vocab))
    if save:
        with open(op.join(args.output, "vocab.json"), "w") as outfile:
            json.dump(vocab, outfile)
    return vocab


def json_to_csv(vocab, train_q, train_a, val_q, val_a, test_q, test_a, save=False):
    # Verify alignment of files
    for q, a in zip(train_q, train_a):
        assert q["question_id"] == a["question_id"]
    for q, a in zip(val_q, val_a):
        assert q["question_id"] == a["question_id"]
    for q, a in zip(test_q, test_a):
        assert q["question_id"] == a["question_id"]

    train_df = [{
            "question": q["question"],
            "answer": a["answer"],
            "video_id": q["video_name"],
            "type": a["type"],
    } for q, a in zip(train_q, train_a)]

    print("Total train size:", len(train_df))
    train_df = [d for d in train_df if d["answer"] in vocab]
    add_qid(train_df)

    val_df = [{
            "question": q["question"],
            "answer": a["answer"],
            "video_id": q["video_name"],
            "type": a["type"],
    } for q, a in zip(val_q, val_a)]
    add_qid(val_df)

    test_df = [{
            "question": q["question"],
            "answer": a["answer"],
            "video_id": q["video_name"],
            "type": a["type"],
    } for q, a in zip(test_q, test_a)]
    add_qid(test_df)

    print("Size:", len(train_df), len(val_df), len(test_df))

    if save:
        write_csv(train_df, 'train.csv')
        write_csv(val_df, 'val.csv')
        write_csv(test_df, 'test.csv')


def add_qid(df):
    vid = set(d['video_id'] for d in df)
    vidmap = {v: 1 for v in vid}
    for d in df:
        d['qid'] = d['video_id']+'_'+str(vidmap[d['video_id']])
        vidmap[d['video_id']] += 1


def write_csv(df, name):
    with open(op.join(args.output, name), 'w', newline='') as f:
        col = ['qid', 'question', 'answer', 'video_id', 'type']
        writer = csv.DictWriter(f, fieldnames=col)
        writer.writeheader()
        writer.writerows(df)


train_q = json.load(open(op.join(args.input, "train_q.json"), "r"))
val_q = json.load(open(op.join(args.input, "val_q.json"), "r"))
test_q = json.load(open(op.join(args.input, "test_q.json"), "r"))

train_a = json.load(open(op.join(args.input, "train_a.json"), "r"))
val_a = json.load(open(op.join(args.input, "val_a.json"), "r"))
test_a = json.load(open(op.join(args.input, "test_a.json"), "r"))


vocab = get_vocabulary(train_a, True)
json_to_csv(vocab, train_q, train_a, val_q, val_a, test_q, test_a, True)

