import csv
import json
import collections
import os


def BBNProcess(dirPath, dataPath, outDir=None, outPath=None):
    Path = dirPath + '/' + dataPath
    # 所有提及
    mentions = []
    # 各级类型
    types = {"level1": [], "level2": []}
    typesOnly = {"level1": [], "level2": []}
    # 各级类型实例数量
    level = {1: 0, 2: 0}
    levelOnly = {1: 0, 2: 0}

    data = []
    _id = 0
    print("getting data...")
    with open(Path, mode='r', encoding="utf-8") as f:
        for row in f:
            ins = json.loads(row)
            sentence = ins['tokens']
            mentionList = ins['mentions']
            for mention in mentionList:
                start, end, labels = mention['start'], mention['end'], mention['labels']
                leftContext, rightContext = sentence[:start], sentence[end:]
                _mention = sentence[start:end]

                mentions.append(' '.join(_mention))
                level[1] += 1
                if len(labels) == 1:
                    levelOnly[1] += 1
                    maxLevel = 1
                    types["level1"].append(labels[0])
                    typesOnly["level1"].append(labels[0])
                else:
                    if len(labels) > 2:
                        print('have label > 2!!!')
                    level[2] += 1
                    maxLevel = 2
                    if labels[0].startswith(labels[1]):
                        types["level1"].append(labels[1])
                        types["level2"].append(labels[0])
                    else:
                        types["level1"].append(labels[0])
                        types["level2"].append(labels[1])
                data.append(
                    {'id': _id, 'mention': _mention, 'max_level': maxLevel, 'labels': labels, 'leftC': leftContext,
                     'rightC': rightContext})
                _id += 1
    mention_counter = collections.Counter(mentions)
    type1_counter = collections.Counter(types["level1"])
    type1only_counter = collections.Counter(typesOnly["level1"])
    type2_counter = collections.Counter(types["level2"])

    print("writing dataset info...")
    with open('datasetInfo.csv', mode='a', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['BBN'])
        w.writerow(['total instance', 'unique mention', 'aver mention count'])
        w.writerow([len(data), len(mention_counter), len(data) / len(mention_counter)])
        w.writerow(['total types', '1-level types', '1-level instance', 'only 1-level instance', '2-level types',
                    '2-level instance'])
        w.writerow(
            [len(type1_counter) + len(type2_counter), len(type1_counter), level[1], levelOnly[1], len(type2_counter),
             level[2]])
        w.writerow(['type instance count'])
        type2 = type2_counter.items()
        for t in type1_counter.items():
            head = []
            num = []
            head.append(t[0] + ' total')
            num.append(t[1])
            head.append(t[0])
            num.append(type1only_counter[t[0]])
            for t2 in type2:
                if t2[0].startswith(t[0]):
                    head.append(t2[0])
                    num.append(t2[1])
            w.writerow(head)
            w.writerow(num)

    if not os.path.exists(outDir):
        os.makedirs(outDir)
    _outPath = outDir + '/' + outPath
    print("writing data...")
    with open(_outPath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    _outPath = outDir + '/' + 'BBNMention.txt'
    with open(_outPath, 'w', encoding='utf-8') as f:
        for i, m in enumerate(mention_counter.items()):
            f.write(f"{m[0]} {i}\n")
    print("done!")


def OntoProcess(dirPath, dataPath, outDir=None, outPath=None):
    # 所有提及
    mentions = []
    # 各级类型在实例中出现的次数
    types = {1: [], 2: [], 3: []}
    # 只以当前层级在实例出现的次数，不是父类
    typesOnly = {1: [], 2: [], 3: []}
    # 包含当前层级的实例数量
    level = {1: 0, 2: 0, 3: 0}
    # 当前层级为最高层级的实例数量
    levelOnly = {1: 0, 2: 0, 3: 0}

    data = []
    _id = 0
    print("getting data...")
    for p in dataPath:
        Path = dirPath + '/' + p
        with open(Path, mode='r', encoding="utf-8") as f:
            for row in f:
                ins = json.loads(row)
                mention = ins['mention_span']
                _mention = mention.split(' ')
                leftContext = ins['left_context_token']
                rightContext = ins['right_context_token']
                labels = ins['y_str']

                mentions.append(mention)
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

                for i in range(1, maxLevel + 1):
                    types[i].extend(tmp[i])
                    level[i] += 1
                levelOnly[maxLevel] += 1
                typesOnly[maxLevel].extend(tmp[maxLevel])
                data.append(
                    {'id': _id, 'mention': _mention, 'max_level': maxLevel, 'labels': labels, 'leftC': leftContext,
                     'rightC': rightContext})
                _id += 1
    mention_counter = collections.Counter(mentions)
    type1_counter = collections.Counter(types[1])
    type1only_counter = collections.Counter(typesOnly[1])
    type2_counter = collections.Counter(types[2])
    type2only_counter = collections.Counter(typesOnly[2])
    type3_counter = collections.Counter(types[3])

    print("writing dataset info...")
    with open('datasetInfo.csv', mode='a', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow([])
        w.writerow(['OntoNotes'])
        w.writerow(['total instance', 'unique mention', 'aver mention count'])
        w.writerow([len(data), len(mention_counter), len(data) / len(mention_counter)])
        w.writerow(['total types', '1-level types', '1-level instance', 'only 1-level instance', '2-level types',
                    '2-level instance', 'only 2-level instance', '3-level types', '3-level instance'])
        w.writerow(
            [len(type1_counter) + len(type2_counter) + len(type3_counter), len(type1_counter), level[1], levelOnly[1],
             len(type2_counter),
             level[2], levelOnly[2], len(type3_counter), level[3]])
        w.writerow(['type instance count'])
        type2 = type2_counter.items()
        type3 = type3_counter.items()
        for t in type1_counter.items():
            head = []
            num = []
            head.append(t[0] + ' total')
            num.append(t[1])
            head.append(t[0])
            num.append(type1only_counter[t[0]])
            for t2 in type2:
                if t2[0].startswith(t[0]):
                    head.append(t2[0] + ' total')
                    num.append(t2[1])
                    head.append(t2[0])
                    num.append(type2only_counter[t2[0]])
                    for t3 in type3:
                        if t3[0].startswith(t2[0]):
                            head.append(t3[0])
                            num.append(t3[1])
            w.writerow(head)
            w.writerow(num)

    if not os.path.exists(outDir):
        os.makedirs(outDir)
    _outPath = outDir + '/' + outPath
    print("writing data...")
    with open(_outPath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    _outPath = outDir + '/' + 'OntoNotesMention.txt'
    with open(_outPath, 'w', encoding='utf-8') as f:
        for i, m in enumerate(mention_counter.items()):
            f.write(f"{m[0]} {i}\n")
    print("done!")


def FewNerdProcess(dirPath, dataPath, outDir=None, outPath=None):
    # 所有提及
    mentions = []
    # 各级类型
    types = {1: [], 2: []}
    typesOnly = {1: [], 2: []}
    # 各级类型实例数量
    level = {1: 0, 2: 0}
    levelOnly = {1: 0, 2: 0}

    data = []
    _id = 0

    def getInstance(sentence, index_):
        nonlocal _id
        instances_ = []
        for m_ in index_:
            _mention = sentence[m_[0]: m_[1]]
            leftContext = sentence[:m_[0]]
            rightContext = sentence[m_[1]:]
            mentions.append(' '.join(_mention))
            level[1] += 1
            if m_[2].find('-') != -1:
                maxLevel = 2
                level[2] += 1
                tmp = m_[2].split('-')
                labels = [tmp[0], m_[2]]
                types[1].append(tmp[0])
                types[2].append(m_[2])
            else:
                maxLevel = 1
                levelOnly[1] += 1
                labels = [m_[2]]
                types[1].append(m_[2])
                typesOnly[1].append(m_[2])
            instances_.append(
                {'id': _id, 'mention': _mention, 'max_level': maxLevel, 'labels': labels, 'leftC': leftContext,
                 'rightC': rightContext})
            _id += 1
        return instances_

    print("getting data...")
    for p in dataPath:
        Path = dirPath + '/' + p
        ins = []
        index = []
        start = -1
        end = -1
        idx = 0
        label = ''
        with open(Path, mode='r', encoding="utf-8") as f:
            for row in f:
                if row != '\n':
                    row = row.strip().split('\t')
                    ins.append(row[0])
                    if row[1] == 'O':
                        if start != -1:
                            end = idx
                            index.append((start, end, label))
                            start, end = -1, -1
                            label = ''
                    else:
                        if start == -1:
                            start = idx
                            label = row[1]
                        else:
                            if row[1] != label:
                                end = idx
                                index.append((start, end, label))
                                start, end = idx, -1
                                label = row[1]
                    idx += 1
                else:
                    if start != -1:
                        index.append((start, idx, label))
                    instances = getInstance(ins, index)
                    data.extend(instances)
                    ins = []
                    index = []
                    start = -1
                    end = -1
                    idx = 0
                    label = ''
            if start != -1:
                index.append((start, idx, label))
            instances = getInstance(ins, index)
            data.extend(instances)

    mention_counter = collections.Counter(mentions)
    type1_counter = collections.Counter(types[1])
    type1only_counter = collections.Counter(typesOnly[1])
    type2_counter = collections.Counter(types[2])

    print("writing dataset info...")
    with open('datasetInfo.csv', mode='a', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow([])
        w.writerow(['Few-Nerd'])
        w.writerow(['total instance', 'unique mention', 'aver mention count'])
        w.writerow([len(data), len(mention_counter), len(data) / len(mention_counter)])
        w.writerow(['total types', '1-level types', '1-level instance', 'only 1-level instance', '2-level types',
                    '2-level instance'])
        w.writerow(
            [len(type1_counter) + len(type2_counter), len(type1_counter), level[1], levelOnly[1], len(type2_counter),
             level[2]])
        w.writerow(['type instance count'])
        type2 = type2_counter.items()
        for t in type1_counter.items():
            head = []
            num = []
            head.append(t[0] + ' total')
            num.append(t[1])
            head.append(t[0])
            num.append(type1only_counter[t[0]])
            for t2 in type2:
                if t2[0].startswith(t[0]):
                    head.append(t2[0])
                    num.append(t2[1])
            w.writerow(head)
            w.writerow(num)

    if not os.path.exists(outDir):
        os.makedirs(outDir)
    _outPath = outDir + '/' + outPath
    print("writing data...")
    with open(_outPath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    _outPath = outDir + '/' + 'FewNerdMention.txt'
    with open(_outPath, 'w', encoding='utf-8') as f:
        for i, m in enumerate(mention_counter.items()):
            f.write(f"{m[0]} {i}\n")
    print("done!")


if __name__ == "__main__":
    FewNerdProcess('dataset/FewNerd', ['supervised/dev.txt', 'supervised/test.txt', 'supervised/train.txt'], 'output',
                   'FewNerd.json')
