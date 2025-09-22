import csv
import json
import os
import random

import tqdm

import constant
from utils import utils
from utils.utils import getDataInfo

"""
将数据集转化为统一格式
"""


def BBNProcess(dataPath, outPath=None, infoPath=None, reData=False):
    types = dict()
    typesSet = set()

    data = []
    _id = 0
    print("getting data...")
    with open(dataPath, mode='r', encoding="utf-8") as f:
        for row in f:
            ins = json.loads(row)
            sentence = ins['tokens']
            mentionList = ins['mentions']
            mentionList_ = []
            for mention in mentionList:
                start, end, labels = mention['start'], mention['end'], mention['labels']
                if start == -1 and end == -1:
                    continue
                _mention = sentence[start:end]

                if labels[0].startswith('/FAC'):
                    for i, t in enumerate(labels):
                        labels[i] = t.replace('/FAC', '/FACILITIES')
                if len(labels) == 1:
                    label = labels[0]
                else:
                    if labels[0].startswith(labels[1]):
                        label = labels[0]
                        if labels[1] not in typesSet:
                            typesSet.add(labels[1])
                            types[labels[1]] = []
                        labels = [labels[1], labels[0]]
                    else:
                        label = labels[1]
                        if labels[0] not in typesSet:
                            typesSet.add(labels[0])
                            types[labels[0]] = []
                if label not in typesSet:
                    typesSet.add(label)
                    types[label] = [_id]
                else:
                    types[label].append(_id)
                mentionList_.append({'start': start, 'end': end, 'labels': labels, 'dtype': 'known_unlabeled'})

            if len(mentionList_) > 0:
                data.append({'sentence': sentence, 'mentions': mentionList_})
                _id += 1
    types, typesSet, data = utils.merge(types, data, 2)
    data = utils.downLevel(('/CONTACT_INFO/PHONE', types['/CONTACT_INFO/PHONE']), data, 2)
    typesSet.remove('/CONTACT_INFO/PHONE')
    types['/CONTACT_INFO'].extend(types['/CONTACT_INFO/PHONE'])
    del types['/CONTACT_INFO/PHONE']

    typeOnly, typeAdd = utils.getTypeInfo(types, mode='list')

    if outPath is not None:
        if infoPath is None:
            dirPath = os.path.dirname(outPath)
            infoPath = dirPath + '/' + 'info.csv'
        print("writing dataset info...")
        level = len(typeOnly)
        info = getDataInfo(typeOnly, typeAdd, level)
        with open(infoPath, mode='w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            for row in info:
                w.writerow(row)

        print("writing data...")
        with open(outPath, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
    print("done!")
    if reData:
        return data
    else:
        return None


def OntoProcess(dataPath, outPath=None, infoPath=None, reData=False):
    types = dict()
    typesSet = set()

    data = []
    _id = 0
    print("getting data...")
    _data = []
    for p in dataPath:
        with open(p, mode='r', encoding="utf-8") as f:
            for row in f:
                ins = json.loads(row)
                mention = ins['mention_span']
                _mention = mention.split(' ')
                leftContext = ins['left_context_token']
                rightContext = ins['right_context_token']
                labels = ins['y_str']

                maxLevel = 0
                tmp = {1: [], 2: [], 3: []}
                for label in labels:
                    _level = label.count('/')
                    tmp[_level].append(label)
                    maxLevel = max(maxLevel, _level)

                flag = False
                for i in range(1, maxLevel + 1):
                    if len(tmp[i]) >= 2:
                        flag = True
                        break
                if flag:
                    continue

                sentence = leftContext + _mention + rightContext
                start = len(leftContext)
                end = start + len(_mention)
                labels = [tmp[i][0] for i in range(1, maxLevel + 1)]
                mentionList_ = [{'start': start, 'end': end, 'labels': labels, 'dtype': 'known_unlabeled'}]
                _data.append({'sentence': sentence, 'mentions': mentionList_})

    mask = [False for _ in range(len(_data))]
    for idx, item in enumerate(_data):
        if mask[idx]:
            continue
        mask[idx] = True
        ins = item
        for i in range(idx + 1, len(_data)):
            if mask[i]:
                continue
            if ins['sentence'] == _data[i]['sentence']:
                ins['mentions'].extend(_data[i]['mentions'])
                mask[i] = True
            else:
                continue
        data.append(ins)

    for ins in data:
        for mention in ins['mentions']:
            level = len(mention['labels'])
            label = mention['labels'][level - 1]
            labels = mention['labels']
            for i in range(level - 1):
                if labels[i] not in typesSet:
                    typesSet.add(labels[i])
                    types[labels[i]] = []
            if label not in typesSet:
                typesSet.add(label)
                types[label] = [_id]
            else:
                types[label].append(_id)
        _id += 1

    types, typesSet, data = utils.merge(types, data, 3)
    typeOnly, typeAdd = utils.getTypeInfo(types, mode='list')

    if outPath is not None:
        if infoPath is None:
            dirPath = os.path.dirname(outPath)
            infoPath = dirPath + '/' + 'info.csv'
        print("writing dataset info...")
        level = len(typeOnly)
        info = getDataInfo(typeOnly, typeAdd, level)
        with open(infoPath, mode='w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            for row in info:
                w.writerow(row)

        print("writing data...")
        with open(outPath, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
    print("done!")
    if reData:
        return data
    else:
        return None


def getFewNerdLabel(label):
    labels = label.split('-')
    tmp = labels[0]
    labels_list = [tmp]
    for i in range(1, len(labels)):
        tmp += f'-{labels[i]}'
        labels_list.append(tmp)
    return labels_list


def FewNerdProcess(dataPath, outPath=None, infoPath=None, reData=False):
    types = dict()
    typesSet = set()

    data = []
    _id = 0
    print("getting data...")
    for p in dataPath:
        with open(p, mode='r', encoding="utf-8") as f:
            ins = {'sentence': [], "mentions": []}
            mention = None
            idx = 0
            label = None
            rows = f.readlines()
            for row in tqdm.tqdm(rows, total=len(rows)):
                row = row.strip()
                if row == '':
                    if mention is not None:
                        mention['end'] = idx
                        ins['mentions'].append(mention)
                        mention = None
                    if ins['mentions']:
                        data.append(ins)
                    ins = {'sentence': [], "mentions": []}
                    idx = 0
                    continue
                row = row.split('\t')
                ins['sentence'].append(row[0])
                if row[1] != 'O':
                    if mention is None:
                        mention = {'start': idx, 'end': 0, 'labels': [], 'dtype': 'known_unlabeled'}
                        label = row[1]
                        mention['labels'] = getFewNerdLabel(label)
                    else:
                        if label != row[1]:
                            mention['end'] = idx
                            ins['mentions'].append(mention)
                            mention = {'start': idx, 'end': 0, 'labels': [], 'dtype': 'known_unlabeled'}
                            label = row[1]
                            mention['labels'] = getFewNerdLabel(label)
                else:
                    if mention is not None:
                        mention['end'] = idx
                        ins['mentions'].append(mention)
                        mention = None
                idx += 1

    for ins in data:
        for mention in ins['mentions']:
            level = len(mention['labels'])
            label = mention['labels'][level - 1]
            labels = mention['labels']
            for i in range(level - 1):
                if labels[i] not in typesSet:
                    typesSet.add(labels[i])
                    types[labels[i]] = []
            if label not in typesSet:
                typesSet.add(label)
                types[label] = [_id]
            else:
                types[label].append(_id)
        _id += 1

    # types, typesSet, data = utils.merge(types, data, 3)
    typeOnly, typeAdd = utils.getTypeInfo_FewNerd(types, mode='list')

    if outPath is not None:
        if infoPath is None:
            dirPath = os.path.dirname(outPath)
            infoPath = dirPath + '/' + 'info.csv'
        print("writing dataset info...")
        level = len(typeOnly)
        info = getDataInfo(typeOnly, typeAdd, level)
        with open(infoPath, mode='w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            for row in info:
                w.writerow(row)

        print("writing data...")
        with open(outPath, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
    print("done!")
    if reData:
        return data
    else:
        return None


def FewNerdSample(dataPath, outPath=None, infoPath=None, reData=False, seed=11, size=150):
    random.seed(seed)
    types = dict()
    typesSet = set()

    data = []
    sample_data = []
    type_indexs = {}
    with open(dataPath, mode='r', encoding="utf-8") as f:
        idx = 0
        for row in f:
            ins = json.loads(row)
            mentions = ins['mentions']
            sentence = ins['sentence']
            for mention in mentions:
                labels = mention['labels']
                label = labels[-1]
                data.append({'sentence': sentence, 'mentions': [mention]})
                type_indexs.setdefault(label, [])
                type_indexs[label].append(idx)
                idx += 1

    for k, v in type_indexs.items():
        indexes = random.sample(v, size)
        for i in indexes:
            sample_data.append(data[i])

    together_data = []
    mask = [False for _ in range(len(sample_data))]
    for idx, item in enumerate(sample_data):
        if mask[idx]:
            continue
        mask[idx] = True
        ins = item
        for i in range(idx + 1, len(sample_data)):
            if mask[i]:
                continue
            if ins['sentence'] == sample_data[i]['sentence']:
                ins['mentions'].extend(sample_data[i]['mentions'])
                mask[i] = True
            else:
                continue
        together_data.append(ins)

    _id = 0
    for ins in together_data:
        for mention in ins['mentions']:
            level = len(mention['labels'])
            label = mention['labels'][level - 1]
            labels = mention['labels']
            for i in range(level - 1):
                if labels[i] not in typesSet:
                    typesSet.add(labels[i])
                    types[labels[i]] = []
            if label not in typesSet:
                typesSet.add(label)
                types[label] = [_id]
            else:
                types[label].append(_id)
            _id += 1

    # types, typesSet, data = utils.merge(types, data, 3)
    typeOnly, typeAdd = utils.getTypeInfo_FewNerd(types, mode='list')

    if outPath is not None:
        if infoPath is None:
            dirPath = os.path.dirname(outPath)
            infoPath = dirPath + '/' + 'info.csv'
        print("writing dataset info...")
        level = len(typeOnly)
        info = getDataInfo(typeOnly, typeAdd, level)
        with open(infoPath, mode='w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            for row in info:
                w.writerow(row)

        print("writing data...")
        with open(outPath, 'w', encoding='utf-8') as f:
            for item in together_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
    print("done!")
    if reData:
        return together_data
    else:
        return None


def FewNerdMapping(dataPath, mapPath, outPath=None, infoPath=None, reData=False):
    type_map = {}
    with open(mapPath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            print(line)
            type_map[line[0]] = line[1]

    types = dict()
    typesSet = set()

    data = []
    with open(dataPath, mode='r', encoding="utf-8") as f:
        for row in f:
            ins = json.loads(row)
            for i, mention in enumerate(ins['mentions']):
                labels = mention['labels']
                tmp = []
                for label in labels:
                    tmp.append(type_map[label])
                ins['mentions'][i]['labels'] = tmp
            data.append(ins)

    _id = 0
    for ins in data:
        for mention in ins['mentions']:
            level = len(mention['labels'])
            label = mention['labels'][level - 1]
            labels = mention['labels']
            for i in range(level - 1):
                if labels[i] not in typesSet:
                    typesSet.add(labels[i])
                    types[labels[i]] = []
            if label not in typesSet:
                typesSet.add(label)
                types[label] = [_id]
            else:
                types[label].append(_id)
            _id += 1

    # types, typesSet, data = utils.merge(types, data, 3)
    typeOnly, typeAdd = utils.getTypeInfo(types, mode='list')

    if outPath is not None:
        if infoPath is None:
            dirPath = os.path.dirname(outPath)
            infoPath = dirPath + '/' + 'info.csv'
        print("writing dataset info...")
        level = len(typeOnly)
        info = getDataInfo(typeOnly, typeAdd, level)
        with open(infoPath, mode='w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            for row in info:
                w.writerow(row)

        print("writing data...")
        with open(outPath, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
    print("done!")
    if reData:
        return data
    else:
        return None


def getStdDataset(dataset, reData=False):
    if dataset == 'BBN':
        data = BBNProcess(constant.raw_dataPath[dataset], constant.std_dataPath[dataset], reData=reData)
    elif dataset == 'OntoNotes':
        data = OntoProcess(constant.raw_dataPath[dataset], constant.std_dataPath[dataset], reData=reData)
    else:
        raise Exception(f'can\'t support dataset: {dataset}')
    return data
