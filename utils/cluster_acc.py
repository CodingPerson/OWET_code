from typing import List

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment as linear_assignment

from utils.finch import getFinchPred
from utils.hdbscanTools import hdbscanManager


def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def split_cluster_acc_v2(y_true, y_pred, mask):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """

    unlabeled_known = set(y_true[mask])
    unlabeled_unknown = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    unlabeled_total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    unlabeled_known_acc = 0
    total_unlabeled_known_instances = 0
    for i in unlabeled_known:
        unlabeled_known_acc += w[ind_map[i], i]
        total_unlabeled_known_instances += sum(w[:, i])
    unlabeled_known_acc /= total_unlabeled_known_instances

    unlabeled_unknown_acc = 0
    total_unlabeled_unknown_instances = 0
    for i in unlabeled_unknown:
        unlabeled_unknown_acc += w[ind_map[i], i]
        total_unlabeled_unknown_instances += sum(w[:, i])
    unlabeled_unknown_acc /= total_unlabeled_unknown_instances

    return unlabeled_total_acc, unlabeled_known_acc, unlabeled_unknown_acc


EVAL_FUNCS = {
    'v2': split_cluster_acc_v2,
}


def log_accs_from_preds(y_true, y_pred, mask, eval_funcs: List[str]):
    """
    :param y_true: GT labels
    :param y_pred: Predicted indices
    :param mask: Which instances belong to Old and New classes
    :return:
    """

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    for i, f_name in enumerate(eval_funcs):

        acc_f = EVAL_FUNCS[f_name]
        tot_acc, unlab_known_acc, unlab_unknown_acc = acc_f(y_true, y_pred, mask)

        if i == 0:
            to_return = (tot_acc, unlab_known_acc, unlab_unknown_acc)

    return to_return


def test_agglo(linked, targets, mask_lab, mask_unlab_known, args, base=0, rePred=False, onlyEnd=True):
    # base2id = {'lab': 0, 'tot': 1, 'unlab_known': 2, 'unlab_unknown': 3}

    mask_lab = mask_lab.astype(bool)
    mask_unlab_known = mask_unlab_known.astype(bool)
    level_acc = np.zeros((args.data_level, 4))

    base_level_k = [len(np.unique(targets[i][mask_lab])) for i in range(args.data_level)]
    best_level_k = np.zeros(args.data_level, dtype=int)

    level_predicts = [[] for _ in range(args.data_level)]

    for i in range(args.data_level):
        if i == 0:
            num_classes = base_level_k[i]
        else:
            #num_classes = base_level_k[i]
            num_classes = base_level_k[i] - base_level_k[i - 1] + best_level_k[i - 1]
        # if i !=args.data_level - 1:
        #     continue
        # num_classes = base_level_k[i]

        target = targets[i]
        dist = linked[:, 2][:-num_classes]
        # dist  = linked[:, 2][-gold_k_list[i]]
        # dist=[dist]
        tolerance = 0
        for d in reversed(dist):
            preds = fcluster(linked, t=d, criterion='distance')
            k = max(preds)
            lab_acc = cluster_acc(y_true=target[mask_lab], y_pred=preds[mask_lab])

            tot_acc, unlab_known_acc, unlab_unknown_acc = log_accs_from_preds(y_true=target[~mask_lab],
                                                                              y_pred=preds[~mask_lab],
                                                                              mask=mask_unlab_known[~mask_lab],
                                                                              eval_funcs=['v2'])
            tmp = [lab_acc, tot_acc, unlab_known_acc, unlab_unknown_acc]
            judge_acc = tmp[base]
            if judge_acc > level_acc[i][base]:  # save best labeled acc without knowing GT K
                level_acc[i] = [lab_acc, tot_acc, unlab_known_acc, unlab_unknown_acc]
                best_level_k[i] = k
                level_predicts[i] = preds
                tolerance = 0
            else:
                tolerance += 1

            if tolerance == 50:
                break
        tmp = best_level_k[i]
        if tmp == num_classes:
            print(f'level {i + 1} can\'t find new class,it is a problem')
    if rePred:
        if onlyEnd:
            return level_acc, best_level_k, level_predicts[-1]
        else:
            return level_acc, best_level_k, level_predicts
    else:
        return level_acc, best_level_k, None


# def test_agglo(linked, targets, mask_lab, mask_unlab_known, args, base=0, rePred=False, onlyEnd=True):
#     # base2id = {'lab': 0, 'tot': 1, 'unlab_known': 2, 'unlab_unknown': 3}
#     mask_lab = mask_lab.astype(bool)
#     mask_unlab_known = mask_unlab_known.astype(bool)
#     level_acc = np.zeros((args.data_level, 4))
#
#     base_level_k = [len(np.unique(targets[i][mask_lab])) for i in range(args.data_level)]
#     best_level_k = np.zeros(args.data_level, dtype=int)
#
#     level_predicts = [[] for _ in range(args.data_level)]
#
#     for i in range(args.data_level):
#         if i == 0:
#             num_classes = base_level_k[i]
#         else:
#             num_classes = base_level_k[i] - base_level_k[i - 1] + best_level_k[i - 1]
#         # if i !=args.data_level - 1:
#         #     continue
#         # num_classes = base_level_k[i]
#
#         target = targets[i]
#         dist = linked[:, 2][:-num_classes]
#
#         tolerance = 0
#         for d in reversed(dist):
#             preds = fcluster(linked, t=d, criterion='distance')
#             k = max(preds)
#             lab_acc = cluster_acc(y_true=target[mask_lab], y_pred=preds[mask_lab])
#
#             tot_acc, unlab_known_acc, unlab_unknown_acc = log_accs_from_preds(y_true=target[~mask_lab],
#                                                                               y_pred=preds[~mask_lab],
#                                                                               mask=mask_unlab_known[~mask_lab],
#                                                                               eval_funcs=['v2'])
#             tmp = [lab_acc, tot_acc, unlab_known_acc, unlab_unknown_acc]
#             judge_acc = tmp[base]
#             if judge_acc > level_acc[i][base]:  # save best labeled acc without knowing GT K
#                 level_acc[i] = [lab_acc, tot_acc, unlab_known_acc, unlab_unknown_acc]
#                 best_level_k[i] = k
#                 level_predicts[i] = preds
#                 tolerance = 0
#             else:
#                 tolerance += 1
#
#             if tolerance == 50:
#                 break
#         tmp = best_level_k[i]
#         if tmp == num_classes:
#             print(f'level {i + 1} can\'t find new class,it is a problem')
#     if rePred:
#         if onlyEnd:
#             return level_acc, best_level_k, level_predicts[-1]
#         else:
#             return level_acc, best_level_k, level_predicts
#     else:
#         return level_acc, best_level_k, None


def finch_acc(c, num_clust, feats, targets, mask_lab, mask_unlab_known, args, base=0, rePred=False):
    # base2id = {'lab': 0, 'tot': 1, 'unlab_known': 2, 'unlab_unknown': 3}
    mask_lab = mask_lab.astype(bool)
    mask_unlab_known = mask_unlab_known.astype(bool)
    level_acc = np.zeros((args.data_level, 4))

    base_level_k = [len(np.unique(targets[i][mask_lab])) for i in range(args.data_level)]
    best_level_k = np.zeros(args.data_level, dtype=int)

    predicts = None

    for i in range(args.data_level):
        if i == 0:
            num_classes = base_level_k[i]
        else:
            num_classes = base_level_k[i] - base_level_k[i - 1] + best_level_k[i - 1]
        target = targets[i]

        tolerance = 0
        for num in range(num_classes, num_clust[0]):
            print(f'num cluster {num}')
            preds = getFinchPred(c, num_clust, feats, num)
            k = max(preds) + 1
            lab_acc = cluster_acc(y_true=target[mask_lab], y_pred=preds[mask_lab])

            tot_acc, unlab_known_acc, unlab_unknown_acc = log_accs_from_preds(y_true=target[~mask_lab],
                                                                              y_pred=preds[~mask_lab],
                                                                              mask=mask_unlab_known[~mask_lab],
                                                                              eval_funcs=['v2'])
            tmp = [lab_acc, tot_acc, unlab_known_acc, unlab_unknown_acc]
            judge_acc = tmp[base]
            if judge_acc > level_acc[i][base]:  # save best labeled acc without knowing GT K
                level_acc[i] = [lab_acc, tot_acc, unlab_known_acc, unlab_unknown_acc]
                best_level_k[i] = k
                predicts = preds
                tolerance = 0
            else:
                tolerance += 1

            if tolerance == 20:
                break
        tmp = best_level_k[i]
        if tmp == num_classes:
            print(f'level {i + 1} can\'t find new class,it is a problem')
    if rePred:
        return level_acc, best_level_k, predicts
    else:
        return level_acc, best_level_k, None


def hdbscan_acc(H, targets, mask_lab, mask_unlab_known, args, base=0, rePred=False):
    # base2id = {'lab': 0, 'tot': 1, 'unlab_known': 2, 'unlab_unknown': 3}
    mask_lab = mask_lab.astype(bool)
    mask_unlab_known = mask_unlab_known.astype(bool)
    level_acc = np.zeros((args.data_level, 4))
    base_level_k = [len(np.unique(targets[i][mask_lab])) for i in range(args.data_level)]
    best_level_k = np.zeros(args.data_level, dtype=int)
    predicts = None

    cluster_num = H.cluster_num

    if cluster_num <= base_level_k[args.data_level - 1]:
        raise Exception('cluster num too small,less than base level k.')

    for i in range(args.data_level):
        if i == 0:
            num_classes = base_level_k[i]
        else:
            num_classes = base_level_k[i] - base_level_k[i - 1] + best_level_k[i - 1]
        target = targets[i]

        tolerance = 0
        for num in range(num_classes, cluster_num):
            preds = H.getClusterPred(num)
            k = max(preds)
            lab_acc = cluster_acc(y_true=target[mask_lab], y_pred=preds[mask_lab])

            tot_acc, unlab_known_acc, unlab_unknown_acc = log_accs_from_preds(y_true=target[~mask_lab],
                                                                              y_pred=preds[~mask_lab],
                                                                              mask=mask_unlab_known[~mask_lab],
                                                                              eval_funcs=['v2'])
            tmp = [lab_acc, tot_acc, unlab_known_acc, unlab_unknown_acc]
            judge_acc = tmp[base]
            if judge_acc > level_acc[i][base]:  # save best labeled acc without knowing GT K
                level_acc[i] = [lab_acc, tot_acc, unlab_known_acc, unlab_unknown_acc]
                best_level_k[i] = k
                predicts = preds
                tolerance = 0
            else:
                tolerance += 1

            if tolerance == 50:
                break
        tmp = best_level_k[i]
        if tmp == num_classes:
            print(f'level {i + 1} can\'t find new class,it is a problem')
    if rePred:
        return level_acc, best_level_k, predicts
    else:
        return level_acc, best_level_k, None
