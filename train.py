import os
from collections import Counter
from datetime import datetime

from torch.nn.utils.rnn import pad_sequence

from utils.metric import getB3Eval, getVmAndARIAndNMI
from utils.visual import reduce_dimension, plot_embeddings

os.environ['PYTHONHASHSEED'] = '11'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np
import tqdm
from scipy.cluster.hierarchy import linkage
from torch.utils.data import DataLoader

from dataProcess.OWETDataset import OWETDataset, labeled_collate_fn
from dataProcess.loadData import loadDataset
from models.model import OWETModel, Cluster2Box, cluster2box
from pretrain import toVal, toEval
from utils.cluster_acc import test_agglo
from utils.forTrain import get_optimizer
from utils.hdbscanTools import compute_mutual_reachability
from utils.loss import lossManager, prototypical_logits, prototypical_logits_box_loss, \
    prototypical_logits_box_loss_BoxE, regularization_logits, volume_reg_loss, regularization_loss, \
    Box_Intersection, Type_box_intersection, Box_Intersection_Taxo, Box_Intersection_boxplus, Cross_box_intersection, \
    Box_intersection_all, prototypical_logits_box_loss_Cross
from utils.losshistory import lossHistory

import constant
import torch

torch.use_deterministic_algorithms(True)
from config import initConfig
from utils.utils import setSeed, getType2Id, getTypeNum, getLevelTarget, saveLossHistory, saveCheckConfig, \
    getTypesInput, getNegSampleList, getConceptCounts, getInstanceCounts, getPathMatch, maxMatchScore, logMatch, \
    getNegSampleList_Cross, cross_loss_weight, getTypesDesc, logClusterMatch, getBroCousinAndDepthWeight


def transform_type_inputs(type_inputs):
    input_ids = [item['input_ids'].squeeze(0) for item in type_inputs]
    attention_mask = [item['attention_mask'].squeeze(0) for item in type_inputs]
    token_type_ids = [item['token_type_ids'].squeeze(0) for item in type_inputs]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)

    return input_ids, attention_mask, token_type_ids


def are_params_equal(params1, params2):
    # 检查参数名称是否一致
    if params1.keys() != params2.keys():
        return False
    # 逐个比较张量数值
    result = np.zeros(len(params1), dtype=bool)
    for i, name in enumerate(params1):
        if torch.equal(params1[name], params2[name]):
            print(f'{name} not change')


def getclusterInfo(all_feats, preds, ratio=0.5, enable_threshold=False):
    assert len(all_feats) == len(preds), 'all feats len not equal preds len'
    assert min(preds) == 1, 'cluster not begin at 1'
    ins2cluster = np.zeros(len(preds), dtype=int)
    ins2pbl = np.zeros(len(all_feats), dtype=bool)
    centroids = []
    cluster_ins_re_dict={}
    k = max(preds)
    for i in range(1, k + 1):
        indexes = np.where(preds == i)[0]
        ins2cluster[indexes] = i - 1
        centroid = all_feats[indexes].mean(axis=0)
        cluster_ins_re_dict[i-1]=all_feats[indexes]
        centroids.append(centroid)
        if enable_threshold:
            dist = np.linalg.norm(all_feats[indexes] - centroid, axis=-1)
            tmp = zip(indexes, dist)
            tmp = sorted(tmp, key=lambda x: x[1])
            num = int(len(tmp) * ratio)
            num = num if num > 0 else 1
            selected = tmp[:num]
            idxes = [item[0] for item in selected]
            ins2pbl[idxes] = True
    return ins2cluster, centroids, ins2pbl,cluster_ins_re_dict


def getClusterResult(model, dataloader, device, id2type, args):
    model.eval()
    with torch.no_grad():
        all_feats = []
        all_idxes = []
        targets = []
        mask_lab = np.array([])
        mask_unlab_known = np.array([])
        bar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader),
                        desc='extra feature')
        for step, batch in bar:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, idxes = batch
            feats, _ = model(input_ids, segment_ids, input_mask, label_ids, mode="train")

            all_feats.extend(feats.cpu().numpy())
            all_idxes.extend(idxes.cpu().numpy().tolist())
            targets.extend(label_ids.cpu().numpy())
            del feats, input_mask, segment_ids
            mask_lab = np.append(mask_lab, np.array(
                [True if dataloader.dataset.data[i]['mention']['dtype'] == 'known_labeled' else False
                 for i in idxes]))
            mask_unlab_known = np.append(mask_unlab_known, np.array(
                [True if dataloader.dataset.data[i]['mention']['dtype'] == 'known_unlabeled' else False
                 for i in idxes]))

        targets = np.asarray(targets)
        all_feats = np.asarray(all_feats)
        level_target = getLevelTarget(targets, args.data_level, id2type)
        mask_lab = mask_lab.astype(bool)
        mask_unlab_known = mask_unlab_known.astype(bool)

        # cluster acc
        if args.cluster_type == 'HAC':
            linked = linkage(all_feats, method=args.cluster_method)
        elif args.cluster_type == 'HDBSCAN':
            dist_martix = compute_mutual_reachability(all_feats, min_samples=5)
            linked = linkage(dist_martix, method=args.cluster_method)
        else:
            raise Exception(f'args.cluster_type error {args.cluster_type}')

        clusterAccDict = dict()
        print('getting cluster result...')
        level_acc, best_level_k, level_preds = test_agglo(linked, level_target, mask_lab, mask_unlab_known, args,
                                                          rePred=True, onlyEnd=False)
        print(level_acc)
        print(best_level_k)
        preds = level_preds[-1]
        lab_acc, tot_acc, unlab_known_acc, unlab_unknown_acc = level_acc[args.data_level - 1]
        best_k = best_level_k[args.data_level - 1]
        clusterAccDict[0] = [tot_acc, unlab_known_acc, unlab_unknown_acc, lab_acc, best_k, best_level_k]

        target = level_target[args.data_level - 1]
        # labeled
        target_lab = target[mask_lab]
        pred_lab = preds[mask_lab]
        # unlabeled
        target_unlab = target[~mask_lab]
        pred_unlab = preds[~mask_lab]
        mask_unlab_known = mask_unlab_known[~mask_lab]

        targetList = [target_lab, target_unlab]
        predList = [pred_lab, pred_unlab]
        target_unlab_known = target_unlab[mask_unlab_known]
        pred_unlab_known = pred_unlab[mask_unlab_known]
        target_unlab_unknown = target_unlab[~mask_unlab_known]
        pred_unlab_unknown = pred_unlab[~mask_unlab_known]
        targetList.extend([target_unlab_known, target_unlab_unknown])
        predList.extend(([pred_unlab_known, pred_unlab_unknown]))

        # B3
        b3Dict = getB3Eval(targetList, predList)
        # V_measure and ARI
        VmARINMIDict = getVmAndARIAndNMI(targetList, predList)

        clusterResult = dict()
        print('getting cluster info...')
        ins2cluster, centroids, ins2pbl,_ = getclusterInfo(all_feats, preds, args.pbl_ratio, args.enable_pbl_ratio)
        clusterResult['ins2cluster'] = ins2cluster
        clusterResult['centroids'] = centroids
        clusterResult['ins2pbl'] = torch.from_numpy(ins2pbl).bool()
        return clusterResult, all_idxes, all_feats, level_preds, (clusterAccDict, b3Dict, VmARINMIDict), mask_lab

def getClusterResult_test(model, dataloader, device, id2type, args):
    model.eval()
    with torch.no_grad():
        all_feats = []
        all_idxes = []
        targets = []
        mask_lab = np.array([])
        mask_unlab_known = np.array([])
        bar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader),
                        desc='extra feature')
        for step, batch in bar:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, idxes = batch
            feats, _ = model(input_ids, segment_ids, input_mask, label_ids, mode="train")

            all_feats.extend(feats.cpu().numpy())
            all_idxes.extend(idxes.cpu().numpy().tolist())
            targets.extend(label_ids.cpu().numpy())
            del feats, input_mask, segment_ids
            mask_lab = np.append(mask_lab, np.array(
                [True if dataloader.dataset.data[i]['mention']['dtype'] == 'known_labeled' else False
                 for i in idxes]))
            mask_unlab_known = np.append(mask_unlab_known, np.array(
                [True if dataloader.dataset.data[i]['mention']['dtype'] == 'known_unlabeled' else False
                 for i in idxes]))

        targets = np.asarray(targets)
        all_feats = np.asarray(all_feats)
        level_target = getLevelTarget(targets, args.data_level, id2type)
        mask_lab = mask_lab.astype(bool)
        mask_unlab_known = mask_unlab_known.astype(bool)

        # cluster acc
        if args.cluster_type == 'HAC':
            linked = linkage(all_feats, method=args.cluster_method)
        elif args.cluster_type == 'HDBSCAN':
            dist_martix = compute_mutual_reachability(all_feats, min_samples=5)
            linked = linkage(dist_martix, method=args.cluster_method)
        else:
            raise Exception(f'args.cluster_type error {args.cluster_type}')

        clusterAccDict = dict()
        print('getting cluster result...')
        level_acc, best_level_k, level_preds = test_agglo(linked, level_target, mask_lab, mask_unlab_known, args,
                                                          rePred=True, onlyEnd=False)
        for level in range(len(level_acc)):
            preds = level_preds[level]
            lab_acc, tot_acc, unlab_known_acc, unlab_unknown_acc = level_acc[level]
            best_k = best_level_k[level]
            clusterAccDict[level] = [tot_acc, unlab_known_acc, unlab_unknown_acc, lab_acc, best_k]

        clusterResult = dict()
        print('getting cluster info...')
        ins2cluster, centroids, ins2pbl,_ = getclusterInfo(all_feats, preds, args.pbl_ratio, args.enable_pbl_ratio)
        clusterResult['ins2cluster'] = ins2cluster
        clusterResult['centroids'] = centroids
        clusterResult['ins2pbl'] = torch.from_numpy(ins2pbl).bool()
        return clusterResult, all_idxes, all_feats, level_preds, clusterAccDict, mask_lab
def getInBoxRate(feats, predicts, centers, offsets, path, epoch):
    tot_num = len(offsets) * len(offsets[0])
    zeros_num = (offsets <= 1.1e-6).sum()
    print(f'tot offset num: {tot_num}')
    print(f'zero offset num: {zeros_num}')
    print(f'rate: {zeros_num / tot_num}')
    feats_exp = feats.unsqueeze(1)  # (B, 1, D)
    centers_exp = centers.unsqueeze(0)  # (1, K, D)
    offsets_exp = offsets.unsqueeze(0)
    delta = (feats_exp - centers_exp).abs()
    judge = (delta <= (offsets_exp + 1e-6))
    inBox = judge.all(dim=2)  # (B,K)
    t = inBox.sum(dim=1)
    t2 = len(torch.where(t > 0)[0])
    print(f'tot {len(feats)}, {t2} in box')

    # tt = torch.where(t <= 0)[0]
    # pred = predicts[tt] - 1
    # tmp_cen = centers[pred]
    # tmp_off = offsets[pred]
    # t3 = judge[tt, pred]
    # t4 = delta[tt, pred]
    # for i in range(len(t3)):
    #     t5 = t4[i][~t3[i]]  # false delta dim value
    #     t6 = feats[tt[i]][~t3[i]]  # false feat dim value
    #     t7 = tmp_cen[i][~t3[i]]  # false cen dim value
    #     t8 = tmp_off[i][~t3[i]]  # false off dim value

    inBox = inBox.numpy()
    with open(path, 'a', encoding='utf-8') as f:
        f.write(f'\nepoch:{epoch}\n')
        for i in range(len(centers)):
            tmp = inBox[:, i]
            tot = tmp.sum()
            right = np.sum(predicts[tmp] == (i + 1))
            tot_should = np.sum(predicts == (i + 1))
            prec = right / tot if tot != 0 else 0.0
            recall = right / tot_should if tot_should != 0 else 0.0
            f.write(f'box {i},tot point:{tot}, right point:{right}, should point:{tot_should}, '
                    f'prec={prec:.6f}, recall={recall:.6f}\n')


def getInBoxRate_level(feats, level_predicts, level_centers, level_offsets, path, epoch):
    level_inboxRate = []
    device = level_centers[0].device
    for level in range(len(level_predicts)):
        inboxRate = []
        predicts = level_predicts[level]
        centers = level_centers[level]
        offsets = level_offsets[level]
        centers = centers.cpu()
        offsets = offsets.cpu()

        print(f'level {level}')
        tot_num = len(offsets) * len(offsets[0])
        zeros_num = (offsets <= 1.1e-6).sum()
        print(f'tot offset num: {tot_num}')
        print(f'zero offset num: {zeros_num}')
        print(f'rate: {zeros_num / tot_num}')
        feats_exp = feats.unsqueeze(1)  # (B, 1, D)
        centers_exp = centers.unsqueeze(0)  # (1, K, D)
        offsets_exp = offsets.unsqueeze(0)
        delta = (feats_exp - centers_exp).abs()
        judge = (delta <= (offsets_exp + 1e-6))
        inBox = judge.all(dim=2)  # (B,K)
        t = inBox.sum(dim=1)
        t2 = len(torch.where(t > 0)[0])
        print(f'tot {len(feats)}, {t2} in box')

        # tt = torch.where(t <= 0)[0]
        # pred = predicts[tt] - 1
        # tmp_cen = centers[pred]
        # tmp_off = offsets[pred]
        # t3 = judge[tt, pred]
        # t4 = delta[tt, pred]
        # for i in range(len(t3)):
        #     t5 = t4[i][~t3[i]]  # false delta dim value
        #     t6 = feats[tt[i]][~t3[i]]  # false feat dim value
        #     t7 = tmp_cen[i][~t3[i]]  # false cen dim value
        #     t8 = tmp_off[i][~t3[i]]  # false off dim value

        inBox = inBox.numpy()
        with open(path, 'a', encoding='utf-8') as f:
            if level == 0:
                f.write(f'\nepoch:{epoch}\n')
            for i in range(len(centers)):
                tmp = inBox[:, i]
                tot = tmp.sum()
                right = np.sum(predicts[tmp] == (i + 1))
                tot_should = np.sum(predicts == (i + 1))
                prec = right / tot if tot != 0 else 0.0
                recall = right / tot_should if tot_should != 0 else 0.0
                f1 = 2 * prec * recall / (prec + recall) if prec + recall != 0 else 0
                inboxRate.append(f1)
                f.write(f'level {level} box {i},\ttot point:{tot},\tright point:{right},\tshould point:{tot_should},\t'
                        f'prec={prec:.6f},\trecall={recall:.6f}\n')
        inboxRate = torch.tensor(inboxRate, dtype=torch.float32).to(device)
        level_inboxRate.append(inboxRate)

    return level_inboxRate


def train(model, device, optimizer, scheduler, labeled_dataloader, adapted_dataloader, lossM,
          eval_dataloader=None, eval_dataset=None, val_dataloader=None, id2type=None, losshistory=None, args=None,
          save_dir='log', box_model=None, box_optimizer=None, box_scheduler=None, types_input=None, neg_list=None,
          neg_list_cross=None, types=None, type2id=None):
    # if args.start_epoch == 0:
    #     clusterAccDict, b3Dict, VmARINMIDict = toEval(model, device, eval_dataloader, eval_dataset, id2type, 0,
    #                                                   args, mode='all')
    #     losshistory.append_eval(clusterAccDict, b3Dict, VmARINMIDict)

    start = args.start_epoch
    cur_epoch = start
    box_inter = Box_intersection_all(margin=args.box_taxo_margin, eps=args.box_taxo_eps)
    type_inter = Type_box_intersection(margin=args.type_taxo_margin, eps=args.type_taxo_eps)
    cross_inter = Cross_box_intersection(margin=args.cross_taxo_margin, eps=args.cross_taxo_eps,
                                         match_weight=args.match_weight)

    predicts = None
    result_ = None
    epoch_feats = None
    epoch5 = False
    pre = None

    type_batch = transform_type_inputs(types_input)
    type_batch = tuple(t.to(device) for t in type_batch)

    # if args.enable_cross_inter == 1:
    type_counts = getConceptCounts(types, type2id, labeled_dataloader.dataset.targets, level=args.data_level)
    for epoch in range(start, args.num_epoch):
        if args.cluster_rep == 'vec':
            clusterResult, all_idxes, _, _, re, _ = getClusterResult(model, eval_dataloader, device, id2type,
                                                                     args)
            losshistory.append_eval(re[0], re[1], re[2])
        elif args.cluster_rep == 'box':
            clusterResult, all_idxes, all_feats, level_preds, re, mask_lab = getClusterResult(model, eval_dataloader,
                                                                                              device, id2type,
                                                                                              args)
            preds = level_preds[-1]
            if predicts is None or not epoch5:
                predicts = preds
            if result_ is None or not epoch5:
                result_ = clusterResult
            torch_feats = torch.from_numpy(all_feats).to(device)
            losshistory.append_eval(re[0], re[1], re[2])
            # torch_feats = torch.nn.functional.normalize(torch_feats, dim=-1)
            # centers, offsets = model.cluster2box(torch_feats, predicts)
            torch_feats = torch.nn.functional.normalize(torch_feats, dim=-1)
            if epoch_feats is None or not epoch5:
                epoch_feats = torch_feats
            # tmp_feats = np.append(all_feats, np.asarray(result['centroids']), axis=0)
            # tmp_feats = np.append(tmp_feats, centers.detach().cpu().numpy(), axis=0)
            # tmp_preds = np.append(predicts, np.asarray(list(range(1, max(predicts) + 1)) * 2))
            # cls, count = np.unique(tmp_preds, return_counts=True)
            # tmp = [(cls[i], count[i]) for i in range(len(cls))]
            # tmp = sorted(tmp, key=lambda x: -x[1])
            # tmp_preds2 = np.zeros_like(tmp_preds)
            # for i, it in enumerate(tmp):
            #     pos = np.where(tmp_preds == i + 1)[0]
            #     tmp_preds2[pos] = i
            # low_dim_feats = reduce_dimension(tmp_feats, method='tsne')
            # plot_embeddings(low_dim_feats, tmp_preds2, len(predicts), 'box', f'epoch {epoch}',
            #                 f'{losshistory.save_dir}/visual/epoch{epoch}.png')
            print('getting inbox info...')
            with torch.no_grad():
                if args.single_box_model == 0:
                    level_centers, level_offsets, _ = model.cluster2box_level(epoch_feats, level_preds)
                else:
                    raise ValueError('not support')
                temp = torch_feats.cpu()
                # centers = torch.nn.functional.normalize(centers, dim=-1)
                # offsets = torch.nn.functional.normalize(offsets, dim=-1)
                level_inboxRate = getInBoxRate_level(temp, level_preds, level_centers, level_offsets
                                                     , f'{losshistory.save_dir}/inbox.txt', epoch)
            if epoch % 5 == 0:
                if epoch5:
                    predicts = preds
                    result_ = clusterResult
                    epoch_feats = torch_feats
        else:
            raise ValueError(f'not support {args.cluster_rep}')

        labelediter = iter(labeled_dataloader)
        model.train()
        if args.single_box_model == 1:
            raise ValueError('not support')
        train_loss = 0
        sup_loss = 0
        pcl_loss = 0
        re_loss = 0
        inter_loss = 0
        type_loss = 0
        cross_loss = 0
        train_steps = 0
        re_type_loss = 0
        c_pcl_loss = 0

        if args.single_box_model == 0:
            cur = {name: param.detach().clone() for name, param in model.Cluster2Box.named_parameters()}
        else:
            raise ValueError('not support')
        if pre is None:
            pre = cur
        if epoch > 0:
            are_params_equal(pre, cur)
            pre = cur
        # if args.enable_cross_inter == 1:
        ins_counts, type_count_all = getInstanceCounts(eval_dataset.targets, all_idxes, level_preds, mask_lab, types,
                                                       type2id,
                                                       id2type)
        matrix, level_matrix = getPathMatch(type_counts, ins_counts)
        match_result, match_score = maxMatchScore(matrix)
        concept_type_path, ins_type_path = type_counts['type_path'], ins_counts['type_path']

        logMatch(level_matrix, match_result, concept_type_path, ins_type_path, ins_counts['type_count'], id2type,
                 f'{losshistory.save_dir}/match_log.txt', epoch)

        logClusterMatch(match_result, concept_type_path, ins_type_path, ins_counts['type_count'], id2type,
                        f'{losshistory.save_dir}/ClusterMatch.txt', epoch)
        logClusterMatch(match_result, concept_type_path, ins_type_path, type_count_all, id2type,
                        f'{losshistory.save_dir}/ClusterMatch.txt', epoch)

        Bro_dict, Cousin_dict, Depth_weight = getBroCousinAndDepthWeight(match_result, concept_type_path, ins_type_path,
                                                                         neg_list)
        Depth_weight = Depth_weight.to(device)
        ins2pbl = clusterResult['ins2pbl'].to(device)
        ins2pbl_cnp = ins2pbl.cpu().numpy()

        bar = tqdm.tqdm(enumerate(adapted_dataloader), total=len(adapted_dataloader), desc='pretrain one epoch')

        if args.enable_pbl_ratio == 1:
            epoch_feats = epoch_feats[ins2pbl]
            predicts = predicts[ins2pbl_cnp]
            level_preds = [pred[ins2pbl_cnp] for pred in level_preds]

        for step, batch in bar:
            if args.cluster_rep == 'box':
                if args.single_box_model == 0:
                    centers, offsets = model.cluster2box(epoch_feats, predicts)
                    level_centers, level_offsets, level_info = model.cluster2box_level(epoch_feats, level_preds)
                    input_ids, input_mask, segment_ids = type_batch
                    type_centers, type_offsets = model(input_ids, segment_ids, input_mask, mode='type',
                                                       generator=args.gpu_g)

                else:
                    raise ValueError('not support')
            postDict = dict()
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, idxes = batch
            feats, _ = model(input_ids, segment_ids, input_mask, mode='train', generator=args.gpu_g)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            idxes = idxes.cpu().numpy().tolist()
            if args.enable_pbl_ratio == 1:
                uq_idxs = torch.from_numpy(np.array([all_idxes.index(item) for item in idxes])).long().to(feats.device)
                pbl_mask = ins2pbl[uq_idxs]
                feats = feats[pbl_mask]
                idxes = np.asarray(idxes)[pbl_mask.cpu().numpy()].tolist()

            if args.cluster_rep == 'vec':
                labels_proto, logits_proto = prototypical_logits(feats, result_, idxes, all_idxes)
                loss_pcl = lossM.ce_loss(logits_proto, labels_proto)
                postDict['pcl loss'] = loss_pcl.item()
                loss_re = torch.tensor(0)
                loss_inter = torch.tensor(0)
            elif args.cluster_rep == 'box':
                # if epoch % 3!=1:
                if args.enable_cross_pb == 0:
                    loss_pcl = prototypical_logits_box_loss_BoxE(feats, result_, idxes, all_idxes, centers, offsets,
                                                                 args)
                    postDict['pcl loss'] = loss_pcl.item()
                else:
                    loss_pcl = torch.tensor(0)
                # print(offsets)

                if args.enable_box_inter == 1 and epoch >= args.box_inter_start:
                    result = box_inter(level_centers, level_offsets, level_info, level_inboxRate, args.inter_n_neg,
                                       args.gumbel_beta, args.tanh, args.box_taxo_alpha, args.box_taxo_extra,
                                       args.IoU_margin,
                                       mode=args.box_inter_type, IoU_mode=args.IoU_mode, d_alpha=args.d_alpha)
                    if args.box_inter_type == 'B4T':
                        loss_inter = 0.5 * lossM.inter_loss(result[0], result[1])
                    elif args.box_inter_type == 'taxo':
                        loss_inter = result[0]
                    elif args.box_inter_type == 'box+':
                        loss_inter = result[0]
                    elif args.box_inter_type == 'IoU':
                        loss_inter = result

                    postDict['inter loss'] = loss_inter.item()
                    if args.box_inter_type == 'box+':
                        postDict['lap'] = result[1].item()
                        postDict['out'] = result[2].item()
                    elif args.box_inter_type == 'taxo':
                        postDict['pc'] = result[1].item()
                        postDict['pp'] = result[2].item()
                        # postDict['nc'] = result[3].item()
                        # postDict['np'] = result[4].item()
                else:
                    loss_inter = torch.tensor(0)

                pred_logits1, mini_logits1 = regularization_logits(type_offsets, args.mini_type_size)
                pred_logits, mini_logits = regularization_logits(offsets, args.mini_ins_size)
                if True:
                    loss_re = lossM.re_loss(pred_logits, mini_logits)
                    postDict['re loss'] = loss_re.item()
                    #loss_re = torch.tensor(0)
                else:
                    loss_re = torch.tensor(0)
                if args.enable_type_inter == 1:
                    loss_re_type = lossM.re_loss(pred_logits1, mini_logits1)
                    postDict['re type loss'] = loss_re_type.item()
                    #loss_re_type = torch.tensor(0)
                else:
                    loss_re_type = torch.tensor(0)
                # loss_re = regularization_loss(offsets, args.mini_size)
                # loss_re = torch.tensor(0)
                # print(offsets)
                # cluster_volume,loss_re = volume_reg_loss(centers,offsets)
                # print(cluster_volume)

                if args.enable_type_inter == 1:
                    result = type_inter(type_centers, type_offsets, args.type_n_neg, neg_list,
                                        args.gumbel_beta, args.tanh, args.type_taxo_alpha, args.type_taxo_extra,
                                        mode=args.type_inter_type, IoU_mode=args.type_IoU_mode,
                                        IoU_margin=args.type_IoU_margin, d_alpha=args.type_d_alpha)
                    if args.type_inter_type == 'B4T':
                        loss_type = 0.5 * lossM.type_loss(result[0], result[1])
                    elif args.type_inter_type == 'taxo':
                        loss_type = result[0]
                    elif args.type_inter_type == 'box+':
                        loss_type = result[0]
                    elif args.type_inter_type == 'IoU':
                        loss_type = result
                    elif args.type_inter_type == 'cbox':
                        loss_type = result
                    postDict['type loss'] = loss_type.item()
                    # if args.type_inter_type == 'box+':
                    #     postDict['lap'] = result[1].item()
                    #     postDict['out'] = result[2].item()
                    # elif args.type_inter_type == 'taxo':
                    #     postDict['pc'] = result[1].item()
                    #     postDict['pp'] = result[2].item()
                    #     postDict['nc'] = result[3].item()
                    #     postDict['np'] = result[4].item()
                else:
                    loss_type = torch.tensor(0)

                if args.enable_cross_inter == 1 and epoch >= args.cross_inter_start:
                    if args.enable_cross_pb == 1:
                        loss_c_pcl = prototypical_logits_box_loss_Cross(feats, result_, idxes, all_idxes, centers,
                                                                        offsets, Bro_dict, Cousin_dict, Depth_weight,
                                                                        args)
                        postDict['c_pcl loss'] = loss_c_pcl.item()
                    else:
                        loss_c_pcl = torch.tensor(0)

                    if args.proj_ins2type == 1:
                        level_centers, level_offsets = model.ins2type_proj(level_cen=level_centers,
                                                                           level_off=level_offsets)
                    if args.proj_type2ins == 1:
                        type_centers, type_offsets = model.type2ins_proj(type_centers, type_offsets)
                    result = cross_inter(type_centers, type_offsets, args.cross_n_neg, neg_list_cross, level_centers,
                                         level_offsets, match_result, level_matrix, concept_type_path, ins_type_path,
                                         args.gumbel_beta, args.tanh, args.cross_taxo_alpha, args.cross_taxo_extra,
                                         mode=args.cross_inter_type, IoU_mode=args.cross_IoU_mode,
                                         IoU_margin=args.cross_IoU_margin, d_alpha=args.cross_d_alpha,
                                         match_weight=args.match_weight,
                                         onlyBottom=args.onlyBottom, self_adv=args.cross_self_adv,
                                         adv_temp=args.cross_adv_temp, dist_margin=args.cross_dist_margin)
                    if args.cross_inter_type == 'B4T':
                        loss_cross = lossM.cross_loss(result[0], result[1])
                    elif args.cross_inter_type == 'taxo':
                        loss_cross = result[0]
                    elif args.cross_inter_type == 'box+':
                        loss_cross = result[0]
                    elif args.cross_inter_type == 'IoU':
                        loss_cross = result
                    elif args.cross_inter_type == 'dist':
                        loss_cross = result
                    elif args.cross_inter_type == 'cbox':
                        loss_cross = result
                    loss_cross = loss_cross * args.cross_weight
                    postDict['cross loss'] = loss_cross.item()
                    # if args.cross_inter_type == 'box+':
                    #     postDict['lap'] = result[1].item()
                    #     postDict['out'] = result[2].item()
                    # elif args.cross_inter_type == 'taxo':
                    #     postDict['pc'] = result[1].item()
                    #     postDict['pp'] = result[2].item()
                    #     postDict['nc'] = result[3].item()
                    #     postDict['np'] = result[4].item()
                else:
                    loss_cross = torch.tensor(0)
                    loss_c_pcl = torch.tensor(0)
            else:
                loss_pcl = torch.tensor(0)
                loss_re = torch.tensor(0)
                loss_inter = torch.tensor(0)
                loss_type = torch.tensor(0)
                loss_cross = torch.tensor(0)
                loss_re_type = torch.tensor(0)
                loss_c_pcl = torch.tensor(0)

            if loss_pcl.isnan():
                print('loss pcl')
                return
            if loss_re.isnan():
                print('loss re')
                return
            if loss_inter.isnan():
                print('loss inter')
                return
            if loss_type.isnan():
                print('loss type')
                return
            if loss_cross.isnan():
                print('loss cross')
                return

            if args.combine_type != 'None':
                try:
                    batch = next(labelediter)
                except StopIteration:
                    labelediter = iter(labeled_dataloader)
                    batch = next(labelediter)

                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, _ = batch
                feats, _ = model(input_ids, segment_ids, input_mask, mode='train', generator=args.gpu_g)
                loss_sup = lossM.sup_loss(feats, device, label_ids)
                postDict['sup loss'] = loss_sup.item()
            else:
                loss_sup = torch.tensor(0)

            if args.cross_dynamic_weight == 1:
                wc = cross_loss_weight(epoch, args.num_epoch)
            else:
                wc = 1.0
            loss = loss_sup + loss_pcl + loss_re + +loss_re_type + loss_inter + loss_type + wc * loss_cross + loss_c_pcl

            train_loss += loss.item()
            sup_loss += loss_sup.item()
            pcl_loss += loss_pcl.item()
            re_loss += loss_re.item()
            re_type_loss += loss_re_type.item()
            inter_loss += loss_inter.item()
            type_loss += loss_type.item()
            cross_loss += wc * loss_cross.item()
            c_pcl_loss += loss_c_pcl.item()

            optimizer.zero_grad()
            if args.single_box_model == 1:
                box_optimizer.zero_grad()
            loss.backward()
            # if args.single_box_model == 1:
            #     for name, param in box_model.named_parameters():
            #         if param.requires_grad:
            #             print(f'{name}:grad {param.grad.norm().item() if param.grad is not None else 0.0:.16f} ')
            # else:
            #     for name, param in model.Cluster2Box.named_parameters():
            #         if param.requires_grad:
            #             print(f'{name}:grad {param.grad.norm().item() if param.grad is not None else 0.0:.16f} ')
            optimizer.step()
            scheduler.step()
            if args.single_box_model == 1:
                box_optimizer.step()
                box_scheduler.step()

            train_steps += 1
            bar.set_postfix(postDict)

        loss = train_loss / train_steps
        sup_loss = sup_loss / train_steps
        pcl_loss = pcl_loss / train_steps
        re_loss = re_loss / train_steps
        inter_loss = inter_loss / train_steps
        type_loss = type_loss / train_steps
        cross_loss = cross_loss / train_steps
        re_type_loss = re_type_loss / train_steps
        c_pcl_loss = c_pcl_loss / train_steps
        print(
            f'Epoch {epoch} train_loss: {loss} sup loss: {sup_loss} pcl loss: {pcl_loss} re loss: {re_loss} '
            f're_type_loss: {re_type_loss} inter loss: {inter_loss} type loss: {type_loss} cross loss: {cross_loss} '
            f'c_pcl_loss: {c_pcl_loss}')
        if args.enable_cross_inter == 1 and args.enable_cross_pb == 1:
            losshistory.append_loss(loss, sup_loss, c_pcl_loss, re_loss, inter_loss, type_loss, cross_loss)
        else:
            losshistory.append_loss(loss, sup_loss, pcl_loss, re_loss, inter_loss, type_loss, cross_loss)
        cur_epoch += 1

        # if args.split_val == 1:
        #     lab_acc, b3Dict, VmARINMIDict = toVal(model, device, val_dataloader, id2type, epoch, args)
        #     losshistory.append_val(lab_acc, b3Dict, VmARINMIDict)

        # if cur_epoch % 10 == 0:
        #     clusterAccDict, b3Dict, VmARINMIDict = toEval(model, device, eval_dataloader,
        #                                                   eval_dataset, id2type, epoch, args,
        #                                                   mode='all')
        #     losshistory.append_eval(clusterAccDict, b3Dict, VmARINMIDict)

        if cur_epoch % 10 == 0:
            if args.single_box_model == 0:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }
            else:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'box_model_state_dict': box_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }
            torch.save(checkpoint, f'{save_dir}/checkpoint_{args.dataset}_lr_{args.lr}_latest.pth')
            saveLossHistory(lossHistory, f'{save_dir}/lossHistory.pkl')
            args.start_epoch = cur_epoch
            saveCheckConfig(args, f'{save_dir}/config.pkl')
            losshistory.loss_plot()
            losshistory.record()
        if cur_epoch == args.stop_epoch:
            return

    if args.cluster_rep == 'vec':
        clusterResult, all_idxes, all_feats, preds, re, _ = getClusterResult(model, eval_dataloader, device, id2type,
                                                                             args)
        losshistory.append_eval(re[0], re[1], re[2])
    elif args.cluster_rep == 'box':
        clusterResult, all_idxes, all_feats, preds, re, _ = getClusterResult(model, eval_dataloader,
                                                                             device, id2type, args)
        # torch_feats = torch.from_numpy(all_feats).to(device)
        losshistory.append_eval(re[0], re[1], re[2])
        # torch_feats = torch.nn.functional.normalize(torch_feats, dim=-1)
        # centers, offsets = model.cluster2box(torch_feats, preds)
        # torch_feats = torch.nn.functional.normalize(torch_feats, dim=-1)

        # tmp_feats = np.append(all_feats, np.asarray(clusterResult['centroids']), axis=0)
        # tmp_feats = np.append(tmp_feats, centers.detach().cpu().numpy(), axis=0)
        # tmp_preds = np.append(preds, np.asarray(list(range(1, max(preds) + 1)) * 2))
        # cls, count = np.unique(tmp_preds, return_counts=True)
        # tmp = [(cls[i], count[i]) for i in range(len(cls))]
        # tmp = sorted(tmp, key=lambda x: -x[1])
        # tmp_preds2 = np.zeros_like(tmp_preds)
        # for i, it in enumerate(tmp):
        #     pos = np.where(tmp_preds == i + 1)[0]
        #     tmp_preds2[pos] = i
        # low_dim_feats = reduce_dimension(tmp_feats, method='tsne')
        # plot_embeddings(low_dim_feats, tmp_preds2, len(preds), 'box', f'epoch {200}',
        #                 f'{losshistory.save_dir}/visual/epoch{200}.png')
        #
        # print('getting inbox info...')
        # centers, offsets = model.cluster2box(torch_feats, preds)
        # temp = torch_feats.detach().cpu()
        # getInBoxRate(temp, preds, centers.detach().cpu(), offsets.detach().cpu(),
        #              f'{losshistory.save_dir}/inbox.txt', 200)

    losshistory.loss_plot()
    losshistory.record()


def main():
    args = initConfig()
    setSeed(args.seed, args.n_gpu)
    args.cpu_g = torch.Generator(device='cpu').manual_seed(args.seed)
    args.gpu_g = torch.Generator(device=f'cuda:{args.device}').manual_seed(args.seed)

    save_dir = 'log'
    args.start_epoch = 0
    # 初始化一些参数
    args.num_known_class = constant.type_nums[args.dataset]['pad']['known']
    args.num_unknown_class = constant.type_nums[args.dataset]['pad']['unknown']
    args.data_level = constant.level_num[args.dataset]
    args.gold_k_list = constant.dataset_gold_k[args.dataset]

    type2id = getType2Id(constant.type2id_path[args.dataset][1])
    id2type = {v: k for k, v in type2id.items()}

    model = OWETModel(args)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.single_box_model == 1:
        raise ValueError('not support')
    else:
        box_model = None

    if args.pretrain_path is not None:
        checkpoint = torch.load(args.pretrain_path, weights_only=True,map_location=device)
        print(checkpoint.keys())
        # del checkpoint['model_state_dict']['proj.proj.nonlinear.0.weight']
        # del checkpoint['model_state_dict']['proj.proj.nonlinear.0.bias']
        # del checkpoint['model_state_dict']['proj.proj.nonlinear.1.weight']
        # del checkpoint['model_state_dict']['proj.proj.nonlinear.1.bias']
        # del checkpoint['model_state_dict']['proj.proj.gate.0.weight']
        # del checkpoint['model_state_dict']['proj.proj.gate.0.bias']
        # del checkpoint['model_state_dict']['proj.proj.gate.1.weight']
        # del checkpoint['model_state_dict']['proj.proj.gate.1.bias']
        # del checkpoint['model_state_dict']['proj.proj.final_linear_layer.weight']
        # for key in list(checkpoint['model_state_dict'].keys()):
        #     if 'type_model' in key:
        #         del checkpoint['model_state_dict'][key]
        #         print(key)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # 加载数据
    train_data, labeled_data, val_data = loadDataset(args.dataset, args.seed, args.split_val, args.val_ratio)
    labeled_dataset = OWETDataset(labeled_data, type2id, model.tokenizer, num_classes=args.num_known_class, args=args,
                                  mode='labeled')
    labeled_g = torch.Generator().manual_seed(args.seed)
    args.labeled_g = labeled_g
    sampler = torch.utils.data.WeightedRandomSampler(labeled_dataset.sample_weights, num_samples=len(labeled_dataset),
                                                     generator=labeled_g)
    labeled_dataloader = DataLoader(labeled_dataset, batch_size=args.labeled_batch_size, shuffle=False, sampler=sampler,
                                    drop_last=True, collate_fn=labeled_collate_fn, num_workers=0, generator=labeled_g)
    if args.split_val == 1:
        args.num_val_class = getTypeNum(val_data)
        val_dataset = OWETDataset(val_data, type2id, model.tokenizer, num_classes=args.num_known_class, args=args,
                                  mode='labeled')
        sampler = torch.utils.data.SequentialSampler(val_dataset)
        val_g = torch.Generator().manual_seed(args.seed)
        args.val_g = val_g
        val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                    drop_last=False, collate_fn=labeled_collate_fn, sampler=sampler, num_workers=0,
                                    generator=val_g)
    else:
        val_dataloader = None

    adapted_data = train_data + labeled_data
    adapted_dataset = OWETDataset(adapted_data, type2id, model.tokenizer, num_classes=args.num_known_class,
                                  args=args,
                                  mode='unlabeled')
    adapted_g = torch.Generator().manual_seed(args.seed)
    args.adapted_g = adapted_g
    sampler = torch.utils.data.RandomSampler(adapted_dataset, num_samples=len(adapted_dataset), generator=adapted_g)
    adapted_dataloader = DataLoader(adapted_dataset, batch_size=args.mlm_batch_size, shuffle=False,
                                    drop_last=True, collate_fn=labeled_collate_fn, sampler=sampler, num_workers=0,
                                    generator=adapted_g)

    # eval的dataloader
    total_data = train_data + labeled_data
    eval_dataset = OWETDataset(total_data, type2id, model.tokenizer,
                               num_classes=args.num_known_class + args.num_unknown_class, args=args,
                               mode='labeled')
    sampler = torch.utils.data.SequentialSampler(eval_dataset)
    eval_g = torch.Generator().manual_seed(args.seed)
    args.eval_g = eval_g
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                 drop_last=False, collate_fn=labeled_collate_fn, sampler=sampler, num_workers=0,
                                 generator=eval_g)

    # 获取相关
    args.epoch_steps = len(adapted_dataloader)
    # args.warmup_epoch = 0.1*args.num_epoch
    optimizer, scheduler = get_optimizer(model, args, mode='bert')
    if args.single_box_model == 1:
        raise ValueError('not support')
    else:
        box_optimizer, box_scheduler = None, None
    lossM = lossManager(args)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'log/train_{args.dataset}_{current_time}'
    losshistory = lossHistory(save_dir, args.contra_type, args.train_type)
    losshistory.gold_k = constant.dataset_gold_k[args.dataset][0]

    for key, value in vars(args).items():
        print(f"{key}:{value}")

    if args.enable_type_desc == 1:
        type_desc = getTypesDesc(constant.type_description[args.dataset])
    else:
        type_desc = None

    types, types_input = getTypesInput(constant.type2id_path[args.dataset][1], args.num_known_class, model.tokenizer,
                                       type_desc)
    neg_list = getNegSampleList(types, type2id)
    neg_list_cross = getNegSampleList_Cross(types, type2id)

    train(model, device, optimizer, scheduler, labeled_dataloader, adapted_dataloader, lossM,
          eval_dataloader, eval_dataset, val_dataloader, id2type, losshistory, args, save_dir, box_model, box_optimizer,
          box_scheduler, types_input, neg_list, neg_list_cross, types, type2id)


if __name__ == '__main__':
    main()
