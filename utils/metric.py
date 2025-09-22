import math

from sklearn import metrics


def b3TotalElementPrecisionAndRecall(predictedsets, groundtruthsets, tot_data_len, elem2gold):
    totalPrecision = 0.0
    totalRecall = 0.0
    for c in predictedsets:
        pred_set = predictedsets[c]
        for r in predictedsets[c]:
            gold_set = groundtruthsets[elem2gold[r]]
            intersection = pred_set.intersection(gold_set)
            if len(predictedsets[c]) == 0:
                totalPrecision += 0.0
            else:
                totalPrecision += len(intersection) / float(len(predictedsets[c]))
            if len(gold_set) == 0:
                totalRecall += 0.0
            else:
                totalRecall += len(intersection) / float(len(gold_set))

    return totalPrecision / float(tot_data_len), totalRecall / float(tot_data_len)


def getB3Eval(targetList, predList):
    # def getB3Eval(level_targets, preds, mask_lab, mask_unlab_known, args, unlab_split=False):
    # target = level_targets[args.data_level - 1]
    # # labeled
    # target_lab = target[mask_lab]
    # pred_lab = preds[mask_lab]
    # # unlabeled
    # target_unlab = target[~mask_lab]
    # pred_unlab = preds[~mask_lab]
    # mask_unlab_known = mask_unlab_known[~mask_lab]
    #
    # targetList = [target_lab, target_unlab]
    # predList = [pred_lab, pred_unlab]
    # tot_lens = [len(target_lab), len(target_unlab)]
    #
    # if unlab_split:
    #     target_unlab_known = target_unlab[mask_unlab_known]
    #     pred_unlab_known = pred_unlab[mask_unlab_known]
    #     target_unlab_unknown = target_unlab[~mask_unlab_known]
    #     pred_unlab_unknown = pred_unlab[~mask_unlab_known]
    #     targetList.extend([target_unlab_known, target_unlab_unknown])
    #     predList.extend(([pred_unlab_known, pred_unlab_unknown]))
    #     tot_lens.extend([len(target_unlab_known), len(target_unlab_unknown)])
    tot_lens = []
    for target in targetList:
        tot_lens.append(len(target))

    goldDict = dict()
    clusterDict = dict()
    elem2gold = dict()
    for i, target_ in enumerate(targetList):
        goldDict.setdefault(i, dict())
        elem2gold.setdefault(i, dict())
        for j, c in enumerate(target_):
            goldDict[i].setdefault(c, set()).add(j)
            elem2gold[i][j] = c
    for i, preds_ in enumerate(predList):
        clusterDict.setdefault(i, dict())
        for j, c in enumerate(preds_):
            clusterDict[i].setdefault(c, set()).add(j)

    b3Dict = dict()
    for i in goldDict:
        b3Dict.setdefault(i, list())
        prec, recall = b3TotalElementPrecisionAndRecall(clusterDict[i], goldDict[i], tot_lens[i], elem2gold[i])
        if prec == 0.0 and recall == 0.0:
            f1 = 0.0
        else:
            f1 = (2 * recall * prec) / (recall + prec)
        b3Dict[i] = [prec, recall, f1]

    return b3Dict


def getVmAndARIAndNMI(targetList, predList):
    # def getVmAndARIAndNMI(level_targets, preds, mask_lab, mask_unlab_known, args, unlab_split=False):
    # target = level_targets[args.data_level - 1]
    # # labeled
    # target_lab = target[mask_lab]
    # pred_lab = preds[mask_lab]
    # # unlabeled
    # target_unlab = target[~mask_lab]
    # pred_unlab = preds[~mask_lab]
    # mask_unlab_known = mask_unlab_known[~mask_lab]
    #
    # targetList = [target_lab, target_unlab]
    # predList = [pred_lab, pred_unlab]
    #
    # if unlab_split:
    #     target_unlab_known = target_unlab[mask_unlab_known]
    #     pred_unlab_known = pred_unlab[mask_unlab_known]
    #     target_unlab_unknown = target_unlab[~mask_unlab_known]
    #     pred_unlab_unknown = pred_unlab[~mask_unlab_known]
    #     targetList.extend([target_unlab_known, target_unlab_unknown])
    #     predList.extend(([pred_unlab_known, pred_unlab_unknown]))

    VmARINMIDict = dict()

    for i in range(len(targetList)):
        VmARINMIDict.setdefault(i, list())
        v_measure = metrics.v_measure_score(targetList[i], predList[i])
        homogeneity_score = metrics.homogeneity_score(targetList[i], predList[i])
        completeness_score = metrics.completeness_score(targetList[i], predList[i])
        ARI = metrics.adjusted_rand_score(targetList[i], predList[i])
        NMI = metrics.normalized_mutual_info_score(targetList[i], predList[i])
        VmARINMIDict[i] = [v_measure, homogeneity_score, completeness_score, ARI, NMI]

    return VmARINMIDict
