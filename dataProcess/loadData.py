import json
import random

import numpy as np

import constant


def load(path, level=-1, lower=True):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for row in f:
            ins = json.loads(row)
            labels = ins['mention']['labels']
            baselevel = len(labels)
            baselabel = labels[baselevel - 1]
            if baselevel < level:
                padding = '/padding'
                if not lower:
                    padding = padding.upper()
                for i in range(level - baselevel):
                    padlabel = baselabel + padding * (i + 1)
                    labels.append(padlabel)
            ins['mention']['labels'] = labels
            data.append(ins)
    return data


def getTestValSplit(data, seed, val_ratio=0.2):
    random.seed(seed)
    typeset = set()
    type_idxes = dict()
    for i, ins in enumerate(data):
        labels = ins['mention']['labels']
        level = len(labels)
        _type = labels[level - 1]
        if _type not in typeset:
            typeset.add(_type)
            type_idxes[_type] = [i]
        else:
            type_idxes[_type].append(i)
    data = np.asarray(data)
    val_idxes = []
    for k, v in type_idxes.items():
        # print(f'{k} : {len(v) * val_ratio}')
        if len(v) * val_ratio < 1:
            continue
        size = round(len(v) * val_ratio)
        # print(f'size: {size}')
        val_idxes.extend(random.sample(v, size))

    val_mask = np.zeros(len(data), dtype=bool)
    val_mask[val_idxes] = True
    val_data = data[val_mask]
    test_data = data[~val_mask]
    return list(test_data), list(val_data)


def loadDataset(dataset, seed, split_val=0, val_ratio=0.2):
    paths = constant.split_dataPath[dataset]
    if dataset == 'BBN':
        level = 2
        lower = False
    elif dataset == 'OntoNotes':
        level = 3
        lower = True
    elif dataset == 'FewNerd':
        level = 2
        lower = True
    else:
        raise Exception(f'not support dataset {dataset}')
    labeled_data = load(paths[0], level, lower)
    unlabeled_data = load(paths[1], level, lower)
    unknown_data = load(paths[2], level, lower)
    print('unknown_data')
    print(len(unknown_data))
    print('unlabeled_data')
    print(len(unlabeled_data))
    train_data = unknown_data + unlabeled_data
    labeled_known, val_data = getTestValSplit(labeled_data, seed, val_ratio=val_ratio)

    if split_val == 1:
        return train_data, labeled_known, val_data
    else:
        return train_data, labeled_data, None
