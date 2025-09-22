import csv
import heapq
import json
from math import ceil
from typing import Optional, Tuple, Any

from utils.utils import getDataInfo, getTypeInfo

"""
将统一格式数据集进行划分
"""


def dataSplit(unknownSet: set, dataPath, labeled_ratio=0.5, outPtah=None, reData=False):
    data = []
    types_num = dict()
    typesSet = set()
    _id = 0
    with open(dataPath, mode='r', encoding="utf-8") as f:
        for row in f:
            ins = json.loads(row)
            mentions = ins['mentions']
            _mentions = []
            labeled = 0
            unlabeled = 0
            unknown = 0
            types = set()
            for mention in mentions:
                level = len(mention['labels'])
                label = mention['labels'][level - 1]
                if label in unknownSet:
                    mention['dtype'] = 'unknown'
                    unknown += 1
                else:
                    unlabeled += 1
                    types.add(label)
                    if label not in typesSet:
                        typesSet.add(label)
                        types_num[label] = 1
                    else:
                        types_num[label] += 1
                _mentions.append(mention)
            data.append({'id': _id, 'sentence': ins['sentence'], 'mentions': _mentions,
                         'labelnum': [labeled, unlabeled, unknown], 'types': types})

    class ComparableDict:
        def __init__(self, data_dict):
            self.data = data_dict

        def __lt__(self, other):
            if self.data['labelnum'][2] > other.data['labelnum'][2]:
                return True
            elif self.data['labelnum'][2] < other.data['labelnum'][2]:
                return False
            else:
                if self.data['labelnum'][1] > other.data['labelnum'][1]:
                    return True
                elif self.data['labelnum'][1] < other.data['labelnum'][1]:
                    return False
                else:
                    if len(self.data['types']) > len(other.data['types']):
                        return True
                    elif len(self.data['types']) < len(other.data['types']):
                        return False
                    else:
                        return len(self.data['mentions']) > len(other.data['mentions'])

    comparable_data = [ComparableDict(d) for d in data]
    heapq.heapify(comparable_data)

    temp_data = []
    for _type, num in types_num.items():
        toLabel = ceil(num * labeled_ratio)
        temp = []
        while toLabel > 0:
            item = heapq.heappop(comparable_data)
            if _type not in item.data['types']:
                temp.append(item)
                continue
            if item.data['labelnum'][0] + item.data['labelnum'][2] == len(item.data['mentions']):
                temp_data.append(item.data)
                continue
            flag = False
            for idx, mention in enumerate(item.data['mentions']):
                if _type != mention['labels'][len(mention['labels']) - 1]:
                    continue
                if mention['dtype'] == 'known_labeled':
                    continue
                item.data['mentions'][idx]['dtype'] = 'known_labeled'
                item.data['labelnum'][0] += 1
                item.data['labelnum'][1] -= 1
                flag = True
                break
            if not flag:
                temp.append(item)
                continue
            toLabel -= 1
            heapq.heappush(comparable_data, item)
        for ins in temp:
            heapq.heappush(comparable_data, ins)

    while comparable_data:
        temp_data.append(heapq.heappop(comparable_data).data)
    if outPtah is not None:
        with open(outPtah, 'w', encoding='utf-8') as f:
            for ins in temp_data:
                item = {'sentence': ins['sentence'], 'mentions': ins['mentions']}
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
    if reData:
        return temp_data
    else:
        return None


def getSplitDataAndInfo(dataPath=None, outPath: Optional[Tuple[Any, Any, Any]] = None,
                        infoPath: Optional[str] = None,
                        data=None,
                        reData=False):
    splitData = {1: [], 2: [], 3: []}  # labeled,unlabeled,unknown
    types = {1: dict(), 2: dict(), 3: dict()}
    typesSet = {1: set(), 2: set(), 3: set()}
    if data is None:
        data = []
        with open(dataPath, mode='r', encoding="utf-8") as f:
            for row in f:
                ins = json.loads(row)
                data.append(ins)
    for ins in data:
        sentence = ins['sentence']
        mentions = ins['mentions']
        for mention in mentions:
            level = len(mention['labels'])
            label = mention['labels'][level - 1]
            if mention['dtype'] == 'known_labeled':
                flag = 1
            elif mention['dtype'] == 'known_unlabeled':
                flag = 2
            else:
                flag = 3
            if label not in typesSet[flag]:
                typesSet[flag].add(label)
                types[flag][label] = 1
            else:
                types[flag][label] += 1
            for i in range(level - 1):
                _label = mention['labels'][i]
                if _label not in typesSet[flag]:
                    typesSet[flag].add(_label)
                    types[flag][_label] = 0
            splitData[flag].append({'sentence': sentence, 'mention': mention})
    for i in range(1, 3 + 1):
        if outPath is not None:
            with open(outPath[i - 1], 'w', encoding='utf-8') as f:
                for item in splitData[i]:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
        if infoPath is not None:
            title = ['labeled', 'unlabeled', 'unknown']
            Types = types[i]
            typesOnly, typesAdd = getTypeInfo(Types, mode='num')
            level = len(typesOnly)
            info = getDataInfo(typesOnly, typesAdd, level)
            with open(infoPath, mode='a', encoding='utf-8', newline='') as f:
                w = csv.writer(f)
                w.writerow([title[i - 1]])
                for row in info:
                    w.writerow(row)
                w.writerow([])
    print('done!')
    if reData:
        return splitData
    else:
        return None
