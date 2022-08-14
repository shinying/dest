from argparse import ArgumentParser
import csv
import json


def double_line(L):
    print('='*L)


def single_line(L):
    print('-'*L)


def anet(pred, truth):
    id2gt, id2type = {}, {}
    for d in truth:
        id2gt[d['qid']] = d['answer']
        id2type[d['qid']] = ord(d['type']) - ord('0')

    types = ['Motion', 'Spatial Relation', 'Temporal Relation', 'Yes/No',
             'Color', 'Object', 'Location', 'Number', 'Other']
    N = [0] * len(types)
    acc = [0] * len(types)
    acc10 = [0] * len(types)

    for d in pred:
        t = id2type[d['qid']]
        N[t] += 1
        acc[t] += d['answer'] == id2gt[d['qid']]
        if 'answer10' in d:
            acc10[t] += id2gt[d['qid']] in d['answer10']


    L = 36
    double_line(L)
    for i in range(len(types)):
        print(f' {types[i]:<20}{acc[i]/N[i]:6.2%}  {acc10[i]/N[i]:6.2%}')

    single_line(L)
    print(f' {"Overall":<20}{sum(acc)/sum(N):6.2%}  {sum(acc10)/sum(N):6.2%}')
    double_line(L)


def next(pred, truth):
    types = ['CW', 'CH', 'TP', 'TN', 'TC', 'DC', 'DL', 'DO']
    type2id = {t: i for i, t in enumerate(types)}
    N = [0] * len(types)
    acc = [0] * len(types)

    for i in range(len(truth)):
        tid = type2id[truth[i]['type']]
        N[tid] += 1
        acc[tid] += int(truth[i]['answer']) == pred[i]['answer']

    L = 18
    double_line(L)

    for i in range(2): # Causal: Why, How
        print(f' {types[i]:<10}{acc[i]/N[i]:6.2%}')
    single_line(L)

    # Causal
    print(f' {"Causal":<10}{sum(acc[:2])/sum(N[:2]):6.2%}')
    double_line(L)

    for i in range(2, 5): # Temporal: Previous, Next, Concurrent
        print(f' {types[i]:<10}{acc[i]/N[i]:6.2%}')
    single_line(L)

    print(f' {"P+N":<10}{sum(acc[2:4])/sum(N[2:4]):6.2%}')
    single_line(L)

    # Temporal
    print(f' {"Temporal":<10}{sum(acc[2:5])/sum(N[2:5]):6.2%}')
    double_line(L)

    for i in range(5, 8): # Descriptive: Count, Location, Other
        print(f' {types[i]:<10}{acc[i]/N[i]:6.2%}')
    single_line(L)

    # Descriptive
    print(f' {"Descrip":<10}{sum(acc[5:])/sum(N[5:]):6.2%}')
    double_line(L)

    # Overall
    print(f' {"Overall":<10}{sum(acc)/sum(N):6.2%}')
    double_line(L)


def agqa(pred, truth):
    # Reasoning
    types1 = ['obj-rel', 'rel-act', 'obj-act', 'superlative',
              'sequencing', 'exists', 'duration-comparison', 'action-recognition']
    cnt = [0] * len(types1)
    acc = [0] * len(types1)
    for pr in pred:
        anno = truth[pr['qid']]
        for i, tp in enumerate(types1):
            if tp in anno['global']:
                acc[i] += anno['answer'] == pr['answer']
                cnt[i] += 1
    assert sum(cnt) == sum(len(anno['global']) for anno in truth.values())
    score1 = [a/c for a, c in zip(acc, cnt)]

    # Semantic
    types2 = ['object', 'relation', 'action']
    cnt = [0] * len(types2)
    acc = [0] * len(types2)
    for pr in pred:
        anno = truth[pr['qid']]
        for i, tp in enumerate(types2):
            if anno['semantic'] == tp:
                acc[i] += anno['answer'] == pr['answer']
                cnt[i] += 1
    assert sum(cnt) == len(truth), f"{sum(cnt)} {len(truth)}"
    score2 = [a/c for a, c in zip(acc, cnt)]

    # Structure
    types3 = ['query', 'compare', 'choose', 'logic', 'verify']
    cnt = [0] * len(types3)
    acc = [0] * len(types3)
    for pr in pred:
        anno = truth[pr['qid']]
        for i, tp in enumerate(types3):
            if anno['structural'] == tp:
                acc[i] += anno['answer'] == pr['answer']
                cnt[i] += 1
    assert sum(cnt) == len(truth)
    score3 = [a/c for a, c in zip(acc, cnt)]

    # Overall
    types4 = ['binary', 'open']
    cnt = [0] * len(types4)
    acc = [0] * len(types4)
    for pr in pred:
        anno = truth[pr['qid']]
        for i, tp in enumerate(types4):
            if anno['ans_type'] == tp:
                acc[i] += anno['answer'] == pr['answer']
                cnt[i] += 1
    assert sum(cnt) == len(truth)
    score4 = [a/c for a, c in zip(acc, cnt)]

    score5 = sum(pr['answer']==truth[pr['qid']]['answer'] for pr in pred) / len(truth)

    L = 30
    double_line(L)
    for tp, score in zip(types1, score1):
        print(f' {tp:<22}{score:.2%}')
    single_line(L)
    for tp, score in zip(types2, score2):
        print(f' {tp:<22}{score:.2%}')
    single_line(L)
    for tp, score in zip(types3, score3):
        print(f' {tp:<22}{score:.2%}')
    single_line(L)
    for tp, score in zip(types4, score4):
        print(f' {tp:<22}{score:.2%}')
    print(f' {"all":<22}{score5:.2%}')

    double_line(L)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset', choices=['anet', 'next', 'agqa'])
    parser.add_argument('pred')
    parser.add_argument('truth')
    args = parser.parse_args()

    pred = json.load(open(args.pred))
    if args.dataset == 'agqa':
        truth = json.load(open(args.truth))
        qids = set()
        preds = []
        for p in pred:
            if p['qid'] not in qids:
                preds.append(p)
                qids.add(p['qid'])
        pred = preds
    else:
        truth = list(csv.DictReader(open(args.truth)))

    assert len(pred) == len(truth), f'Pred: {len(pred)}, Truth: {len(truth)}'

    eval(args.dataset)(pred, truth)
