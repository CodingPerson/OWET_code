import numpy as np
import tqdm

# 方式1
from scipy.spatial.distance import squareform, pdist
from sklearn.neighbors import NearestNeighbors, KDTree


def compute_mutual_reachability(X, min_samples, core_dist=None, metric='euclidean', method='kdtree'):
    # 计算核心距离
    if core_dist is None:
        if method == 'knn':
            nn = NearestNeighbors(n_neighbors=min_samples, metric=metric)
            nn.fit(X)
            distances, _ = nn.kneighbors(X)
            core_distances = distances[:, -1]
        elif method == 'kdtree':
            tree = KDTree(X, metric=metric, leaf_size=100)
            core_distances = tree.query(
                X, k=min_samples + 1, dualtree=True, breadth_first=True
            )[0][:, -1].copy(order="C")
        else:
            raise ValueError(f'{method} not support!')
    else:
        core_distances = core_dist

    # 计算原始距离矩阵
    distance_matrix = squareform(pdist(X, metric=metric))

    # 计算相互可达距离矩阵
    mutual_reachability = np.maximum(distance_matrix, np.maximum.outer(core_distances, core_distances))
    np.fill_diagonal(mutual_reachability, 0)
    mutual_reachability = squareform(mutual_reachability)
    return mutual_reachability


# 方式2
def findParent(parentList, idx):
    if parentList[idx] != idx:
        parentList[idx] = findParent(parentList, parentList[idx])
    return parentList[idx]


def MSTLinked(mst):
    n_samples = len(mst) + 1
    parentList = list(range(n_samples * 2))
    num = np.zeros(n_samples * 2)
    num[:n_samples] = 1
    From = []
    To = []
    Dist = []
    linked = []
    for f, t, d in zip(mst['from'], mst['to'], mst['distance']):
        From.append(f)
        To.append(t)
        Dist.append(d)
    for i in range(len(From)):
        p1 = findParent(parentList, From[i])
        p2 = findParent(parentList, To[i])

        new_parent = n_samples + i
        num_ = num[p1] + num[p2]
        num[new_parent] = num_
        linked.append((p1, p2, Dist[i], num_))

        parentList[p2] = p1
        parentList[p1] = new_parent
    return np.asarray(linked)


# 方式3
class hdbscanManager:
    def __init__(self, labels, condensed_tree, soft_probs):
        self.labels = labels
        self.condensed_tree = condensed_tree
        self.soft_probs = soft_probs
        self.treeDict = dict()
        self.sortList = []
        self.cluster_num = 0
        self.n_samples = len(self.soft_probs)

        self.__process__()

    def __process__(self):
        treeDict = dict()
        nodeSet = set()
        for parent, child, lambda_val, child_size in tqdm.tqdm(zip(self.condensed_tree['parent'],
                                                                   self.condensed_tree['child'],
                                                                   self.condensed_tree['lambda_val'],
                                                                   self.condensed_tree['child_size']),
                                                               total=len(self.condensed_tree['child'])):
            if parent not in nodeSet:
                newNode = Node()
                treeDict[parent] = {'self': newNode, 'parent': -1, 'child': []}
                nodeSet.add(parent)
            if child >= self.n_samples:
                label = -1
            else:
                label = self.labels[child]
            treeDict[parent]['self'].append(child, lambda_val, child_size, label)

            if child_size > 1:
                if child not in nodeSet:
                    newNode = Node()
                    treeDict[child] = {'self': newNode, 'parent': -1, 'child': []}
                    nodeSet.add(child)
                if child not in treeDict[parent]['child']:
                    treeDict[parent]['child'].append(child)
                if treeDict[child]['parent'] == -1:
                    treeDict[child]['parent'] = parent
        for i in nodeSet:
            if treeDict[i]['child']:
                birth_lambda = treeDict[i]['self'].birth_lambda
                self.sortList.append(birth_lambda)
        self.sortList = sorted(self.sortList, key=lambda x: -x)
        self.cluster_num = len(self.sortList) + 1

        noise_pos = np.where(self.labels == -1)[0]
        label_dict = dict()
        for i in noise_pos:
            label = np.argmax(self.soft_probs[i])
            label_dict.setdefault(label, []).append(i)
        for kk, vv in treeDict.items():
            for k, v in label_dict.items():
                if vv['self'].label == k:
                    vv['self'].idxes.extend(v)
            self.treeDict[kk] = vv

    def getClusterPred(self, num):
        pred = np.zeros(self.n_samples)
        if num == self.cluster_num:
            for k, v in self.treeDict.items():
                if v['child']:
                    continue
                label = v['self'].label
                pos = np.asarray(v['self'].idxes, dtype=int)
                pred[pos] = label
        else:
            threshold = self.sortList[-num]
            idxes_dict = dict()
            id_ = 0
            for k, v in self.treeDict.items():
                if not self.judge(v, threshold):
                    continue
                idxes = self.getIdxes(k)
                idxes = np.asarray(idxes, dtype=int)
                idxes_dict[id_] = idxes
                id_ += 1
            for k, v in idxes_dict.items():
                pred[v] = k
        return pred

    def judge(self, v, threshold):
        if v['self'].min_lambda > threshold:
            return False
        if v['self'].min_lambda <= threshold:
            if not v['child']:
                return True
            else:
                for child in v['child']:
                    if self.treeDict[child]['self'].min_lambda <= threshold:
                        return False
                return True

    def getIdxes(self, k):
        idxes = []
        if not self.treeDict[k]['child']:
            idxes.extend(self.treeDict[k]['self'].idxes)
            return idxes
        else:
            for child in self.treeDict[k]['child']:
                idxes.extend(self.getIdxes(child))
            return idxes


class Node:
    def __init__(self):
        # 出现时lambda值,即簇出现时的密度
        self.birth_lambda = -1
        # 簇中最小lambda值，即簇中最小密度
        self.min_lambda = -1
        self.size = 0
        self.idxes = []
        self.child_idx = []
        self.label = -1
        self.labelChange = True

    def append(self, idx, lambda_, size, label):
        if self.birth_lambda == -1 or self.min_lambda == -1:
            self.birth_lambda = lambda_
            self.min_lambda = lambda_
        if lambda_ > self.birth_lambda:
            self.birth_lambda = lambda_
        if lambda_ < self.min_lambda:
            self.min_lambda = lambda_

        # size > 1 说明是一个子簇
        if size > 1:
            self.child_idx.append(idx)
            self.size += size
            self.labelChange = False
            self.label = -1
            self.idxes = []
            return

        if self.labelChange:
            self.idxes.append(idx)
            self.size += 1
            if self.label == -1:
                self.label = label
            if self.label != label:
                raise Exception('Node append error!!! label not match!!!')
