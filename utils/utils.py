import copy
import csv
import os
import pickle
import random
import re
from pathlib import Path

import numpy as np
import pulp
import torch


def downLevel(_type, data, cur_level):
    """在实例中删除当前类型"""
    name = _type[0]
    idx = _type[1]
    for i in idx:
        mentions = data[i]['mentions']
        for j, mention in enumerate(mentions):
            if len(mention['labels']) != cur_level:
                continue
            else:
                if name in mention['labels']:
                    mention['labels'].remove(name)
                    data[i]['mentions'][j] = mention
    return data


def haveNext(Type, types, cur_level, maxLevel):
    """
    判断当前类型是否有子类型
    """
    if cur_level == maxLevel:
        return False
    for t in types.items():
        if t[0].count('/') > cur_level and t[0].startswith(Type):
            return True
    return False


def merge(types: dict, data: list, maxLevel):
    """
    将只有一个实例的type进行合并
    """
    newTypes = dict()
    newTypesSet = set()
    newData = data
    cur_level = maxLevel
    while cur_level > 0:
        for typeItem in types.items():
            if typeItem[0].count('/') != cur_level:
                continue
            if len(typeItem[1]) == 1 and cur_level >= 2:
                st = typeItem[0].strip('/').split('/')
                parent = '/' + '/'.join(st[:cur_level - 1])
                if parent not in newTypesSet:
                    newTypesSet.add(parent)
                    newTypes[parent] = typeItem[1]
                else:
                    newTypes[parent].extend(typeItem[1])
                if haveNext(typeItem[0], newTypes, cur_level, maxLevel):
                    newTypesSet.add(typeItem[0])
                    newTypes[typeItem[0]] = []
                newData = downLevel(typeItem, newData, cur_level)
            else:
                if typeItem[0] in newTypesSet:
                    newTypes[typeItem[0]].extend(typeItem[1])
                    if len(newTypes[typeItem[0]]) == 1 and cur_level >= 2:
                        st = typeItem[0].strip('/').split('/')
                        parent = '/' + '/'.join(st[:cur_level - 1])
                        newTypesSet.add(parent)
                        newTypes[parent] = newTypes[typeItem[0]]
                        tmp = [typeItem[0], newTypes[typeItem[0]]]
                        if haveNext(typeItem[0], newTypes, cur_level, maxLevel):
                            newTypes[typeItem[0]] = []
                        else:
                            newTypesSet.remove(typeItem[0])
                            del newTypes[typeItem[0]]
                        newData = downLevel(tmp, newData, cur_level)
                else:
                    newTypesSet.add(typeItem[0])
                    newTypes[typeItem[0]] = typeItem[1]
        cur_level -= 1
    return newTypes, newTypesSet, newData


def getTypeAdd(types):
    layer = len(types)
    types_add = copy.deepcopy(types)
    while layer > 1:
        for _type in types_add[layer].items():
            st = _type[0].strip('/').split('/')
            parent = '/' + '/'.join(st[:layer - 1])
            types_add[layer - 1][parent] += _type[1]
        layer -= 1
    return types_add


def getTypeAdd_FewNerd(types):
    layer = len(types)
    types_add = copy.deepcopy(types)
    while layer > 1:
        for _type in types_add[layer].items():
            st = _type[0].strip('-').split('-')
            parent = '-'.join(st[:layer - 1])
            types_add[layer - 1][parent] += _type[1]
        layer -= 1
    return types_add


def getTypeInfo(types: dict, mode='num'):
    """获得类型统计信息"""
    _types = dict()
    for _type in types.items():
        level = _type[0].count('/')
        if level not in _types:
            _types[level] = dict()
        if mode == 'num':
            _types[level][_type[0]] = _type[1]
        elif mode == 'list':
            _types[level][_type[0]] = len(_type[1])
        else:
            raise Exception(f'getTypeInfo mode error: {mode}')
    _types_add = getTypeAdd(_types)
    return _types, _types_add


def getTypeInfo_FewNerd(types: dict, mode='num'):
    """获得类型统计信息"""
    _types = dict()
    for _type in types.items():
        level = _type[0].count('-') + 1
        if level not in _types:
            _types[level] = dict()
        if mode == 'num':
            _types[level][_type[0]] = _type[1]
        elif mode == 'list':
            _types[level][_type[0]] = len(_type[1])
        else:
            raise Exception(f'getTypeInfo mode error: {mode}')
    _types_add = getTypeAdd_FewNerd(_types)
    return _types, _types_add


def getDataInfo(typesOnly: dict, typesAdd: dict, level):
    """获得数据统计信息"""
    info = []
    info.append(['total instance'])
    total_ins = 0
    for t in typesAdd[1].items():
        total_ins += t[1]
    info.append([total_ins])

    tl = ['total types']
    for i in range(1, level + 1):
        tl.append(f'{i}-level types')
        tl.append(f'{i}-level instance')
        if i < level:
            tl.append(f'only {i}-level instance')
    info.append(tl)
    tl = []
    tot_types = 0
    for i in range(1, level + 1):
        tot_types += len(typesOnly[i])
    tl.append(tot_types)
    for i in range(1, level + 1):
        tl.append(len(typesOnly[i]))
        num = 0
        for t in typesAdd[i].items():
            num += t[1]
        tl.append(num)
        if i < level:
            num = 0
            for t in typesOnly[i].items():
                num += t[1]
            tl.append(num)
    info.append(tl)

    info.append(['type instance count'])

    def getHead_Num(Only, Add, parent, curLevel):
        nonlocal level
        _head = []
        _num = []
        for _t in Add[curLevel].items():
            if _t[0].startswith(parent):
                if curLevel < level:
                    _head.append(_t[0] + ' total')
                    _num.append(_t[1])
                    _head.append(_t[0])
                    _num.append(Only[curLevel][_t[0]])
                    _h, _n = getHead_Num(Only, Add, _t[0], curLevel + 1)
                    _head.extend(_h)
                    _num.extend(_n)
                else:
                    _head.append(_t[0])
                    _num.append(_t[1])
        return _head, _num

    for t in typesAdd[1].items():
        head = []
        num = []
        head.append(t[0] + ' total')
        num.append(t[1])
        head.append(t[0])
        num.append(typesOnly[1][t[0]])
        h, n = getHead_Num(typesOnly, typesAdd, t[0], 2)
        head.extend(h)
        num.extend(n)
        info.append(head)
        info.append(num)
    return info


def getType2Id(path):
    type2id = dict()
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split()
            type2id[line[0]] = int(line[1])
    return type2id


def genType2Id(data, path, type2id=None, start=0):
    typeset = set()
    type2child = dict()
    type1level = []
    for ins in data:
        labels = ins['mention']['labels']
        for i, label in enumerate(labels):
            if i == 0 and label not in type1level:
                type1level.append(label)
            if label not in typeset:
                typeset.add(label)
                type2child[label] = []
            if i > 0:
                if label not in type2child[labels[i - 1]]:
                    type2child[labels[i - 1]].append(label)

    def f(parent):
        nonlocal type2child
        info_ = []
        for t in type2child[parent]:
            info_.append(t)
            info_.extend(f(t))
        return info_

    info = []
    for label in type1level:
        info.append(label)
        info.extend(f(label))

    with open(path, 'a', encoding='utf-8') as f:
        idx = 0
        for t in info:
            if type2id is None or t not in type2id:
                f.write(t + ' ' + str(start + idx) + '\n')
                idx += 1


def setSeed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def getLevelTarget(target, level, id2type):
    # print(target.shape)
    # print(target)
    levelTarget = np.zeros((level, target.shape[0]), dtype=int)
    indexes = [np.where(row != 0)[0] for row in target]
    # print(indexes)
    for i, t in enumerate(indexes):
        types = [id2type[idx] for idx in t]
        # print(types)
        for pos, type_ in enumerate(types):
            lv = type_.count('/')
            # print(str(lv) + ' ' + str(i) + ' ' + str(pos))
            levelTarget[lv - 1, i] = t[pos]
    return levelTarget


def loadCheckConfig(checkpointConfigPath):
    with open(checkpointConfigPath, 'rb') as f:
        args = pickle.load(f)
    return args


def saveCheckConfig(args, checkpointConfigPath):
    with open(checkpointConfigPath, 'wb') as f:
        pickle.dump(args, f)


def loadLossHistory(path):
    with open(path, 'rb') as f:
        lossHistory = pickle.load(f)
    return lossHistory


def saveLossHistory(history, path):
    with open(path, 'wb') as f:
        pickle.dump(history, f)


def printConfig(path):
    with open(path, 'rb') as f:
        args = pickle.load(f)
    for key, value in vars(args).items():
        print(f"{key}--{value}")


def getTypeNum(data):
    typeset = set()
    for ins in data:
        labels = ins['mention']['labels']
        for label in labels:
            if label not in typeset:
                typeset.add(label)
    return len(typeset)


def getValBestFromLog(path, outpath):
    """outpath是csv格式"""
    best = []
    best_epoch = []
    best_info = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            matches = re.findall(r'\d\.\d*', line)
            tmp = [float(t) for t in matches]
            if idx == 0:
                best = tmp
                best_epoch = [0 for _ in tmp]
                best_info = [[] for _ in tmp]
            else:
                for i, v in enumerate(tmp):
                    if v > best[i]:
                        best[i] = v
                        best_epoch[i] = idx
                        best_info[i] = tmp

    with open(outpath, 'w', encoding='utf-8') as f:
        w = csv.writer(f)
        base2print = ['lab acc', 'b3 prec', 'b3 recall', 'b3 f1', 'v_measure', 'homo', 'complete',
                      'ari', 'nmi']
        t = ['', 'best value', 'best epoch', 'lab acc', 'b3 prec', 'b3 recall', 'b3 f1', 'v_measure', 'homo',
             'complete',
             'ari', 'nmi']
        w.writerow(t)
        for i in range(len(best)):
            t = [base2print[i], best[i], best_epoch[i], best_info[i][0], best_info[i][1], best_info[i][2],
                 best_info[i][3],
                 best_info[i][4], best_info[i][5], best_info[i][6], best_info[i][7], best_info[i][8]]
            w.writerow(t)


def getEpochInfo(dirpath, epoch):
    # epoch = 190
    # folder_path = Path("aaa")
    # file_names = [f.name for f in folder_path.iterdir() if f.is_file()]
    # for filepath in file_names:
    #     with open(f'aaa/{filepath}', 'r', encoding='utf-8') as f:
    #         lines = f.readlines()
    #         matches = re.findall(r'\d\.\d+', lines[epoch+1])
    #         if filepath.startswith('log_acc'):
    #             tt = re.findall(r'(?<= )\d\d', lines[epoch+1])
    #         tmp = [float(t) for t in matches]
    #         if filepath.startswith('log_acc'):
    #             tmp.append(int(tt[0]))
    #         data = tmp
    #     with open(f'tmp.csv', 'a', encoding='utf-8') as f:
    #         w = csv.writer(f)
    #         w.writerow([filepath])
    #         w.writerow(data)
    folder_path = Path(dirpath)
    file_names = [f.name for f in folder_path.iterdir() if f.is_file()]
    for filepath in file_names:
        if filepath.endswith('lab.txt'):
            with open(f'{dirpath}/{filepath}', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                matches = re.findall(r'\d\.\d+', lines[epoch + 1])
                if filepath.startswith('log_acc'):
                    tt = re.findall(r'(?<= )\d\d', lines[epoch + 1])
                tmp = [float(t) for t in matches]
                if filepath.startswith('log_acc'):
                    tmp.append(int(tt[0]))
                data = tmp
            with open(f'tmp.csv', 'a', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow([filepath])
                w.writerow(data)
    for filepath in file_names:
        if filepath.endswith('tot.txt'):
            with open(f'{dirpath}/{filepath}', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                matches = re.findall(r'\d\.\d+', lines[epoch + 1])
                if filepath.startswith('log_acc'):
                    tt = re.findall(r'(?<= )\d\d', lines[epoch + 1])
                tmp = [float(t) for t in matches]
                if filepath.startswith('log_acc'):
                    tmp.append(int(tt[0]))
                data = tmp
            with open(f'tmp.csv', 'a', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow([filepath])
                w.writerow(data)
    for filepath in file_names:
        if filepath.endswith('unlab_known.txt'):
            with open(f'{dirpath}/{filepath}', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                matches = re.findall(r'\d\.\d+', lines[epoch + 1])
                if filepath.startswith('log_acc'):
                    tt = re.findall(r'(?<= )\d\d', lines[epoch + 1])
                tmp = [float(t) for t in matches]
                if filepath.startswith('log_acc'):
                    tmp.append(int(tt[0]))
                data = tmp
            with open(f'tmp.csv', 'a', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow([filepath])
                w.writerow(data)
    for filepath in file_names:
        if filepath.endswith('unknown.txt'):
            with open(f'{dirpath}/{filepath}', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                matches = re.findall(r'\d\.\d+', lines[epoch + 1])
                if filepath.startswith('log_acc'):
                    tt = re.findall(r'(?<= )\d\d', lines[epoch + 1])
                tmp = [float(t) for t in matches]
                if filepath.startswith('log_acc'):
                    tmp.append(int(tt[0]))
                data = tmp
            with open(f'tmp.csv', 'a', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow([filepath])
                w.writerow(data)


def getTypesDesc(path):
    type_desc = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if line == '\n':
                idx += 1
                continue
            if line.startswith('/'):
                type = line.strip()
                desc = lines[idx + 1].strip()
                type_desc[type] = desc
                idx += 2
    return type_desc


def getTypesInput(path, num, tokenizer, type_desc=None):
    types = []
    types_input = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        need = lines[:num]
        for line in need:
            line = line.strip()
            line = line.split()
            line = line[0]
            types.append(line)
            line = line.replace('/PADDING', '')
            line = line.replace('/padding', '')
            if type_desc is not None:
                desc = type_desc[line]
            name = line.strip('/')
            name = name.split('/')
            name = ' '.join(name)
            name = name.replace('_', ' ')
            if type_desc is not None:
                input_ = tokenizer.encode_plus(
                    name, desc,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
            else:
                input_ = tokenizer.encode_plus(
                    name,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
            types_input.append(input_)
    return types, types_input


def getPC(types, type2id):
    child = {}
    parent = {}

    for t in types:
        level = t.count('/')
        id_ = type2id[t]
        child.setdefault(id_, [])
        parent.setdefault(id_, -1)
        for t2 in types:
            level2 = t2.count('/')
            id2_ = type2id[t2]
            child.setdefault(id2_, [])
            parent.setdefault(id2_, -1)
            if t2.startswith(t) and level2 == level + 1:
                parent[id2_] = id_
                if id2_ not in child[id_]:
                    child[id_].append(id2_)
            if t.startswith(t2) and level2 == level - 1:
                parent[id_] = id2_
                if id_ not in child[id2_]:
                    child[id2_].append(id_)
    return child, parent


def getNegSampleList(types, type2id):
    child, parent = getPC(types, type2id)

    np_child = {}
    for k, v in child.items():
        np_child[k] = np.asarray(v, dtype=np.int32)

    neg_list = {}
    for t in types:
        level = t.count('/')
        if level == 1:
            continue
        id_ = type2id[t]
        neg_list.setdefault(id_, {'level': 0, 'parent': -1, 'bro': [], 'uncle': [], 'cousin': [], 'rest': []})
        neg_list[id_]['level'] = level
        neg_list[id_]['parent'] = parent[id_]
        bro = np_child[parent[id_]]
        if len(bro) > 1:
            idxes = np.where(bro != id_)[0]
            neg_list[id_]['bro'].extend(bro[idxes])
        if parent[parent[id_]] != -1:
            uncle = np_child[parent[parent[id_]]]
            if len(uncle) > 1:
                idxes = np.where(uncle != parent[id_])[0]
                neg_list[id_]['uncle'].extend(uncle[idxes])
                tmp = []
                for i in uncle[idxes]:
                    tmp.extend(child[i])
                neg_list[id_]['cousin'].extend(tmp)

        neg_list[id_]['bro'] = np.asarray(neg_list[id_]['bro'])
        neg_list[id_]['uncle'] = np.asarray(neg_list[id_]['uncle'])
        neg_list[id_]['cousin'] = np.asarray(neg_list[id_]['cousin'])

        p = parent[id_]
        while parent[p] != -1:
            p = parent[p]

        tmp = []
        for k, v in parent.items():
            if k != p and v == -1:
                tmp.append(k)
        while len(tmp) > 0:
            tmp = np.asarray(tmp)
            neg_list[id_]['rest'].append(tmp)
            tmp2 = []
            for i in tmp:
                tmp2.extend(np_child[i])
            tmp = tmp2

    return neg_list


def getNegSampleList_Cross(types, type2id):
    child, parent = getPC(types, type2id)

    np_child = {}
    for k, v in child.items():
        np_child[k] = np.asarray(v, dtype=np.int32)

    neg_list = {}
    for t in types:
        level = t.count('/')
        id_ = type2id[t]
        neg_list.setdefault(id_, {'level': 0, 'bro': [], 'uncle': [], 'cousin': [], 'rest': []})
        neg_list[id_]['level'] = level
        if level != 1:
            bro = np_child[parent[id_]]
            if len(bro) > 1:
                idxes = np.where(bro != id_)[0]
                neg_list[id_]['bro'].extend(bro[idxes])
            if parent[parent[id_]] != -1:
                uncle = np_child[parent[parent[id_]]]
                if len(uncle) > 1:
                    idxes = np.where(uncle != parent[id_])[0]
                    neg_list[id_]['uncle'].extend(uncle[idxes])
                    tmp = []
                    for i in uncle[idxes]:
                        tmp.extend(child[i])
                    neg_list[id_]['cousin'].extend(tmp)

        neg_list[id_]['bro'] = np.asarray(neg_list[id_]['bro'])
        neg_list[id_]['uncle'] = np.asarray(neg_list[id_]['uncle'])
        neg_list[id_]['cousin'] = np.asarray(neg_list[id_]['cousin'])

        if level != 1:
            p = parent[id_]
            while parent[p] != -1:
                p = parent[p]
        else:
            p = id_

        tmp = []
        for k, v in parent.items():
            if k != p and v == -1:
                tmp.append(k)
        while len(tmp) > 0:
            tmp = np.asarray(tmp)
            neg_list[id_]['rest'].append(tmp)
            tmp2 = []
            for i in tmp:
                tmp2.extend(np_child[i])
            tmp = tmp2

    return neg_list


def getConceptCounts(types, type2id, targets, level):
    child, parent = getPC(types, type2id)
    type_count = {type2id[t]: 0 for t in types}
    target_num = targets.sum(dim=0)
    index = np.where(target_num > 0)[0]
    for i in index:
        type_count[i] = target_num[i].item()

    type_path = []
    for t in types:
        path = []
        if t.count('/') == level:
            id_ = type2id[t]
            path.append(id_)
            while parent[id_] != -1:
                path.append(parent[id_])
                id_ = parent[id_]
            type_path.append(path)

    return {'type_path': type_path, 'type_count': type_count}


def getInstanceCounts(target, all_idxes, level_pred, mask, types, type2id, id2type):
    all_idxes = np.array(all_idxes)
    target_all = target
    target = target[:, :len(types)]
    type_count = {}
    type_count_all = {}
    parent = {}
    for i, preds in enumerate(level_pred):
        parent[i] = {}
        unique_values = np.unique(preds)
        unique_values = unique_values - 1
        type_count[i] = {}
        type_count_all[i] = {}
        for j in unique_values:
            if i == 0:
                parent[i][j] = -1
            else:
                index = np.where(preds == j + 1)[0]
                parent_pred = level_pred[i - 1][index]
                v = parent_pred[0]
                if np.sum(parent_pred - v) != 0:
                    raise Exception('level_pred not only one parent')
                parent[i][j] = v - 1
            type_count[i].setdefault(j, {type2id[t]: 0 for t in types})
            type_count_all[i].setdefault(j, {type2id[t]: 0 for t in types})
            indexes = np.where(preds == j + 1)[0]
            idxes = all_idxes[indexes]
            tmp = target_all[idxes]
            tmp_num = tmp.sum(dim=0)
            index = np.where(tmp_num > 0)[0]
            for i2 in index:
                t_type = id2type[i2]
                if t_type.count('/') == i + 1:
                    type_count_all[i][j][i2] = tmp_num[i2].item()

            tmp_mask = mask[idxes]
            tmp = target[idxes]
            tmp = tmp[tmp_mask]
            tmp_num = tmp.sum(dim=0)
            index = np.where(tmp_num > 0)[0]
            for i2 in index:
                t_type = id2type[i2]
                if t_type.count('/') == i + 1:
                    type_count[i][j][i2] = tmp_num[i2].item()

    type_path = []
    level = len(level_pred)
    for k, v in parent[level - 1].items():
        tmp = [k, v]
        tmp_level = level
        tmp_level -= 2
        while parent[tmp_level][v] != -1:
            v = parent[tmp_level][v]
            tmp.append(v)
            tmp_level -= 1
        type_path.append(tmp)

    return {'type_path': type_path, 'type_count': type_count}, type_count_all


def getInstanceCounts_unknown(target, all_idxes, level_pred, mask, types, type2id, id2type):
    all_idxes = np.array(all_idxes)
    type_count = {}
    parent = {}
    for i, preds in enumerate(level_pred):
        parent[i] = {}
        unique_values = np.unique(preds)
        unique_values = unique_values - 1
        type_count[i] = {}
        for j in unique_values:
            if i == 0:
                parent[i][j] = -1
            else:
                index = np.where(preds == j + 1)[0]
                parent_pred = level_pred[i - 1][index]
                v = parent_pred[0]
                if np.sum(parent_pred - v) != 0:
                    raise Exception('level_pred not only one parent')
                parent[i][j] = v - 1
            type_count[i].setdefault(j, {type2id[t]: 0 for t in types})
            indexes = np.where(preds == j + 1)[0]
            idxes = all_idxes[indexes]

            tmp_mask = mask[idxes]
            tmp = target[idxes]
            tmp = tmp[tmp_mask]
            tmp_num = tmp.sum(dim=0)
            index = np.where(tmp_num > 0)[0]
            for i2 in index:
                t_type = id2type[i2]
                if t_type.count('/') == i + 1:
                    type_count[i][j][i2] = tmp_num[i2].item()

    type_path = []
    level = len(level_pred)
    for k, v in parent[level - 1].items():
        tmp = [k, v]
        tmp_level = level
        tmp_level -= 2
        while parent[tmp_level][v] != -1:
            v = parent[tmp_level][v]
            tmp.append(v)
            tmp_level -= 1
        type_path.append(tmp)

    return {'type_path': type_path, 'type_count': type_count}


def getPathMatch(concept, instance):
    concept_type_path = concept['type_path']
    concept_type_count = concept['type_count']
    instance_type_path = instance['type_path']
    instance_type_count = instance['type_count']

    matrix = np.zeros((len(concept_type_path), len(instance_type_path)))
    level_matrix = np.zeros((len(concept_type_path), len(instance_type_path), len(concept_type_path[0])))

    for i, tp in enumerate(concept_type_path):
        for j, ip in enumerate(instance_type_path):
            level = len(tp)
            tot_score = 0
            tot_num = 0
            # for k in range(level):
            #     n_type = concept_type_count[tp[k]]
            #     ins_num = instance_type_count[level - k - 1][ip[k]]
            #     n_same = ins_num[tp[k]]
            #     n_ins = 0
            #     for kk, vv in ins_num.items():
            #         n_ins += vv
            #     score = n_same / (n_type + n_ins - n_same) if (n_type + n_ins - n_same) != 0 else 0
            #
            #     level_matrix[i][j][k] = score
            #     tot_score += score
            # aver_score = tot_score / level

            k = 0
            n_type = concept_type_count[tp[k]]
            ins_num = instance_type_count[level - k - 1][ip[k]]
            n_same = ins_num[tp[k]]
            n_ins = 0
            for kk, vv in ins_num.items():
                n_ins += vv
            score = n_same / (n_type + n_ins - n_same) if (n_type + n_ins - n_same) != 0 else 0
            # score = n_same / (n_ins) if (n_ins) != 0 else 0
            level_matrix[i][j][k] = score
            aver_score = score

            #     score = score * (n_type+n_ins)
            #     tot_num += (n_type+n_ins)
            #     level_matrix[i][j][k] = score
            #     tot_score += score
            # aver_score = tot_score / tot_num
            # for k in range(level):
            #     level_matrix[i][j][k] /= tot_num
            matrix[i][j] = aver_score
    return matrix, level_matrix


def maxMatchScore(matrix):
    problem = pulp.LpProblem('matchProblem', pulp.LpMaximize)

    I, J = matrix.shape
    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)
         for i in range(I) for j in range(J)}

    # 目标函数
    problem += pulp.lpSum(matrix[i][j] * x[i, j] for i in range(I) for j in range(J))

    # 行约束：每行选一列
    for i in range(I):
        problem += pulp.lpSum(x[i, j] for j in range(J)) == 1, f"row_{i}_constraint"

    # 列约束：每列最多被选一次
    for j in range(J):
        problem += pulp.lpSum(x[i, j] for i in range(I)) <= 1, f"col_{j}_constraint"

    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    # 提取结果
    match_result = []
    match_score = []
    for i in range(I):
        for j in range(J):
            if pulp.value(x[i, j]) == 1:
                match_result.append((i, j))
                match_score.append(matrix[i, j])

    return match_result, match_score


def logMatch(level_matrix, match_result, concept_type_path, ins_type_path, ins_type_count, id2type, path, epoch):
    match_dict = dict()
    data = []
    num = {i: 0 for i in range(len(concept_type_path[0]))}
    level_num = {i: 0 for i in range(len(concept_type_path[0]))}
    D = dict()
    for pair in match_result:
        c_idx = pair[0]
        i_idx = pair[1]
        info = ''
        c_p = concept_type_path[c_idx]
        i_p = ins_type_path[i_idx]
        tot_score = 0
        flag = False
        for i, idx in enumerate(c_p):
            match_dict.setdefault(idx, -1)
            type_ = id2type[idx]
            D.setdefault(idx, 0)
            if match_dict[idx] != -1 and i_p[i] != match_dict[idx]:
                info += f'{type_} conflict '
                if D[idx] == 0:
                    num[i] += 1
                    D[idx] = 1
            else:
                if match_dict[idx] == -1:
                    lv = type_.count('/')
                    level_num[lv - 1] += 1
                match_dict[idx] = i_p[i]

            score = level_matrix[c_idx][i_idx][i]
            tot_score += score
            tmp = ''
            i_numDict = ins_type_count[len(c_p) - i - 1][i_p[i]]
            for k, v in i_numDict.items():
                if v > 0:
                    tmp += f'{id2type[k]}:{v},'
            info = f'{type_}:{score:.6f}({tmp}),\t' + info
        info = f'match score:{tot_score / len(c_p)},\t' + info
        data.append(info)
    ttt = ''
    for k, v in num.items():
        ttt += f'level {len(concept_type_path[0]) - k - 1}:{v}/{level_num[len(concept_type_path[0]) - k - 1]},({v / level_num[len(concept_type_path[0]) - k - 1]:.6f})\t'
        # ttt += f'level {len(concept_type_path[0]) - k - 1}:{v}/{len(concept_type_path)},({v / len(concept_type_path):.6f})\t'
    print(ttt)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(f'epoch {epoch}\n')
        f.write(ttt)
        f.write('\n')
        for item in data:
            f.write(item)
            f.write('\n')
        f.write('\n')


def cross_loss_weight(epoch, max_epoch):
    return (epoch + 1) / max_epoch


def logClusterMatch(match_result, concept_type_path, ins_type_path, ins_type_count, id2type, path, epoch):
    info = {i: {} for i in range(len(concept_type_path[0]))}
    match_set = set()
    for pair in match_result:
        c_idx = pair[0]
        i_idx = pair[1]
        c_p = concept_type_path[c_idx]
        i_p = ins_type_path[i_idx]
        for i, idx in enumerate(c_p):
            type_ = id2type[idx]
            ins_cluster = i_p[i]
            if i == 0:
                match_set.add(ins_cluster)
            info[i].setdefault(ins_cluster, {'ins_num': {}, 'type_map': [], 'child': []})
            i_numDict = ins_type_count[len(c_p) - i - 1][ins_cluster]
            info[i][ins_cluster]['ins_num'] = i_numDict
            if type_ not in info[i][ins_cluster]['type_map']:
                info[i][ins_cluster]['type_map'].append(type_)
            if i > 0 and i_p[i - 1] not in info[i][ins_cluster]['child']:
                info[i][ins_cluster]['child'].append(i_p[i - 1])

    for i_p in ins_type_path:
        for i, ins_cluster in enumerate(i_p):
            if i == 0 and ins_cluster in match_set:
                break
            info[i].setdefault(ins_cluster, {'ins_num': {}, 'type_map': [], 'child': []})
            i_numDict = ins_type_count[len(i_p) - i - 1][ins_cluster]
            info[i][ins_cluster]['ins_num'] = i_numDict
            if not info[i][ins_cluster]['type_map']:
                info[i][ins_cluster]['type_map'].append('unknown')
            if i > 0 and i_p[i - 1] not in info[i][ins_cluster]['child']:
                info[i][ins_cluster]['child'].append(i_p[i - 1])

    info_sorted = {}
    for level, CM_info in info.items():
        tmp = []
        for cluster, C_info in CM_info.items():
            tmp.append((cluster, C_info))
        tmp = sorted(tmp, key=lambda x: x[0])
        info_sorted[level] = tmp

    with open(path, 'a', encoding='utf-8') as f:
        f.write(f'{epoch}\n')
        for level, CM_info in info_sorted.items():
            for cluster, C_info in CM_info:
                f.write(f'level {len(concept_type_path[0]) - level}: {cluster}\tchild: ')
                tmp = C_info['child']
                tmp = sorted(tmp)
                for idx in tmp:
                    f.write(f'{idx},')
                f.write('\tins_num==>')
                for idx, num in C_info['ins_num'].items():
                    if num > 0:
                        f.write(f'{id2type[idx]}:{num}, ')
                f.write('\ttype_match==>')
                for t in C_info['type_map']:
                    f.write(f'{t}, ')
                f.write('\n')
        f.write('\n')


def getBroCousinAndDepthWeight(mathc_result, concept_path, instance_path, neg_list):
    depth = len(concept_path[0])
    Bro_dict = {p[0]: [p[0]] for p in instance_path}
    Cousin_dict = {p[0]: [] for p in instance_path}
    base_score = 1
    depth_weight = {p[0]: [base_score for _ in instance_path] for p in instance_path}
    c_i_map = {}

    for k, v in mathc_result:
        c_p = concept_path[k]
        i_p = instance_path[v]
        c_i_map[c_p[0]] = i_p[0]
    for k, v in mathc_result:
        c_p = concept_path[k]
        i_p = instance_path[v]
        bro = neg_list[c_p[0]]['bro']
        bro_map = [c_i_map[c] for c in bro]
        bro_score = 1 - (depth - 1) / depth
        Bro_dict[i_p[0]].extend(bro_map)
        for idx in bro_map:
            depth_weight[i_p[0]][idx] = bro_score
        cousin = neg_list[c_p[0]]['cousin']
        cousin_map = [c_i_map[c] for c in cousin]
        cousin_score = 1 - (depth - 2) / depth
        Cousin_dict[i_p[0]].extend(cousin_map)
        for idx in cousin_map:
            depth_weight[i_p[0]][idx] = cousin_score

    tmp = []
    for k, v in depth_weight.items():
        tmp.append((k, torch.from_numpy(np.asarray(v, dtype=float))))
    tmp = sorted(tmp, key=lambda x: x[0])
    Depth_weight = [x[1] for x in tmp]
    Depth_weight = torch.stack(Depth_weight)
    return Bro_dict, Cousin_dict, Depth_weight
