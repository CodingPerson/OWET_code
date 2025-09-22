import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class lossManager:
    def __init__(self, args):
        if args.contra_type == 'sup':
            self.sup_loss = SupConLoss1()
            self.ce_loss = torch.nn.CrossEntropyLoss()
        elif args.contra_type in ['self', 'test']:
            self.sup_loss = SupConLoss1()
            self.self_loss = SupConLoss2()
            self.ce_loss = torch.nn.CrossEntropyLoss()

        if args.train_type == 'train' or 'pretrain':
            self.sup_loss = SupConLoss1()
            self.ce_loss = torch.nn.CrossEntropyLoss()
            self.re_loss = torch.nn.MSELoss()
            self.inter_loss = torch.nn.BCEWithLogitsLoss()
            self.type_loss = torch.nn.BCEWithLogitsLoss()


"""
对比损失1
"""


# 有监督对比损失
class SupConLoss1(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss1, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, device, labels):
        features = torch.nn.functional.normalize(features, dim=-1)

        batch_size = features.shape[0]

        if len(labels.shape) != 2:
            raise ValueError('`labels` 的形状应为 [bsz, num_classes]')
            # 生成掩码：标签完全相同的样本对
        mask = torch.all(torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)), dim=2).float().to(device)

        contrast_feature = features

        anchor_feature = contrast_feature

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size).view(-1, 1).to(device), 0)

        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(batch_size).mean()

        return loss


# 半监督/自监督对比损失
class SupConLoss2(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, contrast_mode='all'):
        super(SupConLoss2, self).__init__()
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None, temperature=0.07, device=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            temperature)
        torch.set_printoptions(threshold=sys.maxsize)
        # print('dot')
        # print( anchor_dot_contrast)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # print('logits')
        # print(logits)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # print('exp logits')
        # print(exp_logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print('log prob')
        # print(log_prob)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # print((mask * log_prob).sum(1))
        # print(mask.sum(1))
        # print(mean_log_prob_pos)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        # if torch.isnan(loss).any():
        #     print('have nan in self loss')
        return loss


"""
对比损失2
"""


def pair_cosine_similarity(x, x_adv, eps=1e-8):
    n = x.norm(p=2, dim=1, keepdim=True)
    n_adv = x_adv.norm(p=2, dim=1, keepdim=True)
    return (x @ x.t()) / (n * n.t()).clamp(min=eps), (x_adv @ x_adv.t()) / (n_adv * n_adv.t()).clamp(min=eps), (
            x @ x_adv.t()) / (n * n_adv.t()).clamp(min=eps)


def nt_xent(x, x_adv, mask, cuda=True, t=0.07):
    x, x_adv, x_c = pair_cosine_similarity(x, x_adv)
    x = torch.exp(x / t)
    x_adv = torch.exp(x_adv / t)
    x_c = torch.exp(x_c / t)
    mask_count = mask.sum(1)
    mask_reverse = (~(mask.bool())).long()

    if cuda:
        dis = (x * (mask - torch.eye(x.size(0)).long().cuda()) + x_c * mask) / (
                x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + mask_reverse
        dis_adv = (x_adv * (mask - torch.eye(x.size(0)).long().cuda()) + x_c.T * mask) / (
                x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t))) + mask_reverse
    else:
        dis = (x * (mask - torch.eye(x.size(0)).long()) + x_c * mask) / (
                x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + mask_reverse
        dis_adv = (x_adv * (mask - torch.eye(x.size(0)).long()) + x_c.T * mask) / (
                x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t))) + mask_reverse

    loss = (torch.log(dis).sum(1) + torch.log(dis_adv).sum(1)) / mask_count
    return -loss.mean()


"""
PCL损失
"""


def prototypical_logits(feats, cluster_result, uq_idxs, all_uq_idxs, temperature=0.07):
    # uq_idxs is used to do label assign
    uq_idxs = torch.from_numpy(np.array([all_uq_idxs.index(item) for item in uq_idxs])).long().cuda()

    ins2cluster = torch.from_numpy(np.asarray(cluster_result['ins2cluster'])).long().cuda()
    prototypes = torch.from_numpy(np.asarray(cluster_result['centroids'])).cuda()

    logits_proto = torch.mm(feats, prototypes.t())
    logits_proto = torch.div(logits_proto, temperature)
    labels_proto = ins2cluster[uq_idxs]  # 当前样本对应的原型 ID

    return labels_proto, logits_proto


def prototypical_logits_box_loss(feats, cluster_result, uq_idxs, all_uq_idxs, centers, offsets, args=None):
    # negative sampling loss Query2Box dist
    uq_idxs = torch.from_numpy(np.array([all_uq_idxs.index(item) for item in uq_idxs])).long().cuda()
    ins2cluster = torch.from_numpy(np.asarray(cluster_result['ins2cluster'])).long().cuda()
    if args.enable_weight == 1:
        cluster_ids, cluster_counts = torch.unique(ins2cluster, return_counts=True, sorted=True)
        tot_subsampling_weight = torch.sqrt(1 / cluster_counts)
        assert cluster_ids[0] == 0, 'cluster not begin at 0'

    boxes_max = centers + offsets
    boxes_min = centers - offsets

    # 维度扩展以支持广播计算
    feats_exp = feats.unsqueeze(1)  # (B, 1, D)
    boxes_max_exp = boxes_max.unsqueeze(0)  # (1, K, D)
    boxes_min_exp = boxes_min.unsqueeze(0)  # (1, K, D)
    centers_exp = centers.unsqueeze(0)  # (1, K, D)

    # 计算边界超出量（类似原代码中的score_offset）
    below_min = F.relu(boxes_min_exp - feats_exp)  # 特征小于min的部分
    above_max = F.relu(feats_exp - boxes_max_exp)  # 特征大于max的部分
    dist_out = below_min + above_max  # (B, K, D)

    # 计算中心调整值（类似原代码中的score_center_plus）
    clipped_feats = torch.minimum(
        boxes_max_exp,
        torch.maximum(boxes_min_exp, feats_exp)
    )
    dist_in = clipped_feats - centers_exp  # (B, K, D)

    # L_1
    dist = torch.norm(dist_out, p=1, dim=-1) + args.alpha * torch.norm(dist_in, p=1, dim=-1)
    # L_2^2
    # dist = torch.sqrt(torch.sum(dist_out ** 2, dim=-1)) + alpha * torch.sqrt(torch.sum(dist_in ** 2, dim=-1))
    dist = args.margin - dist

    labels_proto = ins2cluster[uq_idxs]  # 当前样本对应的原型 ID
    if args.enable_weight == 1:
        subsampling_weight = tot_subsampling_weight[labels_proto]

    # 创建正样本的掩码
    mask_pos = torch.zeros_like(dist, dtype=torch.bool).cuda()
    mask_pos.scatter_(1, labels_proto.unsqueeze(1), True)  # (B, K)

    # 提取正负样本的距离
    dist_pos = dist[mask_pos]  # (B,)
    # print(dist_pos.shape)
    dist_neg = dist[~mask_pos].view(dist.size(0), -1)  # (B, K-1)
    # print(dist_neg.shape)

    # 计算正样本损失项
    pos_loss = F.logsigmoid(dist_pos)
    # print(pos_loss.shape)

    # 计算负样本损失项（取负并平均）
    neg_loss = F.logsigmoid(-dist_neg).mean(dim=1)
    # print(neg_loss.shape)
    # print('pos loss')
    # print(-pos_loss/len(pos_loss))
    # print('neg loss')
    # print(-neg_loss/len(neg_loss))

    if args.enable_weight == 1:
        # pos_loss = subsampling_weight * pos_loss / subsampling_weight.sum()
        # neg_loss = subsampling_weight * neg_loss / subsampling_weight.sum()
        # print('weight pos loss')
        # print(-pos_loss)
        # print('weight neg loss')
        # print(-neg_loss)
        pos_loss = -(subsampling_weight * pos_loss).sum()
        neg_loss = -(subsampling_weight * neg_loss).sum()
        # pos_loss = - pos_loss.sum()
        # neg_loss = - neg_loss.sum()
        pos_loss /= subsampling_weight.sum()
        neg_loss /= subsampling_weight.sum()
    else:
        pos_loss = -pos_loss.mean()
        neg_loss = -neg_loss.mean()

    total_loss = (pos_loss + neg_loss) / 2

    return total_loss


def regularization_logits(offsets, min_value=0.03):
    mini_size = torch.ones_like(offsets) * min_value
    mask = torch.where(offsets < mini_size, 1, 0)
    pred = torch.mul(offsets, mask)
    true_ = torch.mul(mini_size, mask)
    return pred, true_


def regularization_loss(offsets, min_value=0.03):
    mini_size = torch.ones_like(offsets) * min_value
    mask = torch.where(offsets < mini_size, 1, 0)
    temp = (offsets <= 1e-6).sum(dim=-1)
    base = torch.where(temp == 0, 1.0, 3.0)
    add = temp / offsets.shape[-1] * 3
    tot = add + base
    tot = tot.unsqueeze(1)
    pred = torch.mul(offsets, mask)
    true_ = torch.mul(mini_size, mask)
    delta = true_ - pred
    loss = (delta ** 2) * tot
    reg_loss = loss.mean()
    return reg_loss


def prototypical_logits_box_loss_BoxE(feats, cluster_result, uq_idxs, all_uq_idxs, centers, offsets, args=None):
    # uq_idxs is used to do label assign
    uq_idxs = torch.from_numpy(np.array([all_uq_idxs.index(item) for item in uq_idxs])).long().to(feats.device)
    # print(cluster_result)
    ins2cluster = torch.from_numpy(np.asarray(cluster_result['ins2cluster'])).long().to(feats.device)

    neg_num = centers.shape[-1] - 1

    if args.dist_type == 'BoxE':
        width_p = offsets * 2 + 1
        # 维度扩展以支持广播计算
        feats_exp = feats.unsqueeze(1)  # (B, 1, D)
        centers_exp = centers.unsqueeze(0)  # (1, K, D)
        offsets_exp = offsets.unsqueeze(0)  # (1, K, D)
        width_p_exp = width_p.unsqueeze(0)  # (1, K, D)

        delta = (feats_exp - centers_exp).abs()
        dist_in = delta / width_p
        dist_out = width_p * delta - offsets_exp * (width_p_exp - 1) / width_p_exp
        dists = torch.where(delta <= offsets_exp, dist_in, dist_out)

        dist = torch.norm(dists, dim=-1, p=2)
    elif args.dist_type == 'Q2B':
        # 维度扩展以支持广播计算
        feats_exp = feats.unsqueeze(1)  # (B, 1, D)
        centers_exp = centers.unsqueeze(0)  # (1, K, D)
        offsets_exp = offsets.unsqueeze(0)  # (1, K, D)

        delta = (feats_exp - centers_exp).abs()
        dist_out = torch.relu(delta - offsets_exp)  # (B, K, D)
        dist_in = torch.min(delta, offsets_exp)

        dist = torch.norm(dist_out, dim=-1, p=1) + args.alpha * torch.norm(dist_in, dim=-1, p=1)

    labels_proto = ins2cluster[uq_idxs]  # 当前样本对应的原型 ID
    # 创建正样本的掩码
    mask_pos = torch.zeros_like(dist, dtype=torch.bool).to(feats.device)
    mask_pos.scatter_(1, labels_proto.unsqueeze(1), True)  # (B, K)

    # 提取正负样本的距离
    dist_pos = dist[mask_pos]  # (B,)
    dist_neg = dist[~mask_pos].view(dist.size(0), -1)  # (B, K-1)
    # print(dist_pos)
    # print(dist_neg)
    if args.box_loss_type == 'neg_sample':
        pos_loss = F.logsigmoid(args.margin - dist_pos)
        # pos_loss = pos_loss.sum()
        neg_loss = F.logsigmoid(dist_neg - args.margin)
        if args.box_self_adv == 1:
            score = - dist_neg * args.adv_temp
            att = torch.softmax(score, dim=-1).detach()
            neg_loss = att * neg_loss
            neg_loss = neg_loss.mean(dim=-1)
        else:
            neg_loss = neg_loss.mean(dim=-1)
        pos_loss = pos_loss.mean()
        neg_loss = neg_loss.mean()
        # print(f'pos_loss={pos_loss}')
        # print(f'neg_loss={neg_loss}')
        tot_loss = - neg_loss - pos_loss
    elif args.box_loss_type == 'margin_base':
        pos_loss = dist_pos
        if args.box_self_adv == 1:
            pos_loss = pos_loss.unsqueeze(1)
            score = - dist_neg * args.adv_temp
            att = torch.softmax(score, dim=-1).detach()
            neg_loss = att * dist_neg
        else:
            neg_loss = dist_neg.mean(dim=-1)
        tot_loss = torch.relu(args.margin + pos_loss - neg_loss)
        if len(tot_loss.shape) == 2:
            tot_loss = tot_loss.mean(dim=-1)
        tot_loss = tot_loss.mean()

    return tot_loss


def prototypical_logits_box_loss_Cross(feats, cluster_result, uq_idxs, all_uq_idxs, centers, offsets, Bro_dict,
                                       Cousin_dict, Depth_weight, args=None):
    # uq_idxs is used to do label assign
    uq_idxs = torch.from_numpy(np.array([all_uq_idxs.index(item) for item in uq_idxs])).long().to(feats.device)
    # print(cluster_result)
    ins2cluster = torch.from_numpy(np.asarray(cluster_result['ins2cluster'])).long().to(feats.device)

    if args.dist_type == 'BoxE':
        width_p = offsets * 2 + 1
        # 维度扩展以支持广播计算
        feats_exp = feats.unsqueeze(1)  # (B, 1, D)
        centers_exp = centers.unsqueeze(0)  # (1, K, D)
        offsets_exp = offsets.unsqueeze(0)  # (1, K, D)
        width_p_exp = width_p.unsqueeze(0)  # (1, K, D)

        delta = (feats_exp - centers_exp).abs()
        dist_in = delta / width_p
        dist_out = width_p * delta - offsets_exp * (width_p_exp - 1) / width_p_exp
        dists = torch.where(delta <= offsets_exp, dist_in, dist_out)

        dist = torch.norm(dists, dim=-1, p=2)
    elif args.dist_type == 'Q2B':
        # 维度扩展以支持广播计算
        feats_exp = feats.unsqueeze(1)  # (B, 1, D)
        centers_exp = centers.unsqueeze(0)  # (1, K, D)
        offsets_exp = offsets.unsqueeze(0)  # (1, K, D)

        delta = (feats_exp - centers_exp).abs()
        dist_out = torch.relu(delta - offsets_exp)  # (B, K, D)
        dist_in = torch.min(delta, offsets_exp)

        dist = torch.norm(dist_out, dim=-1, p=1) + args.alpha * torch.norm(dist_in, dim=-1, p=1)

    cos_dist = 1 - F.cosine_similarity(feats_exp, centers_exp, dim=-1)

    labels_proto = ins2cluster[uq_idxs]  # 当前样本对应的原型 ID
    # 创建正样本的掩码
    mask_pos = torch.zeros_like(dist, dtype=torch.bool).to(feats.device)
    mask_pos.scatter_(1, labels_proto.unsqueeze(1), True)  # (B, K)

    # 提取正负样本的距离
    dist_pos = dist[mask_pos]  # (B,)
    dist_neg = dist  # (B, K)
    if args.box_loss_type == 'neg_sample':
        pos_loss = F.logsigmoid(args.margin - dist_pos)
        neg_losses = []
        if args.dynomic_pb_margin == 1:
            for i, i_neg_loss in enumerate(dist_neg):
                weight = Depth_weight[labels_proto[i]]
                margin = weight * args.margin
                neg_loss = F.logsigmoid(i_neg_loss - margin)
                mask_neg = torch.zeros_like(neg_loss, dtype=torch.bool).to(feats.device)
                mask_neg[labels_proto[i]] = True
                neg_loss_ = neg_loss[~mask_neg]
                d_neg = dist_neg[i][~mask_neg]

                if args.box_self_adv == 1:
                    score = - d_neg * args.adv_temp
                    att = torch.softmax(score, dim=-1).detach()
                    neg_loss_ = att * neg_loss_
                neg_loss_ = neg_loss_.mean()
                neg_losses.append(neg_loss_)
            neg_losses = torch.stack(neg_losses)
            pos_loss = pos_loss.mean()
            neg_losses = neg_losses.mean()
            tot_loss = - neg_losses - pos_loss
        else:
            for i, i_neg_loss in enumerate(dist_neg):
                neg_loss = F.logsigmoid(i_neg_loss - args.margin)
                bro = Bro_dict[labels_proto[i].item()]
                mask_neg = torch.zeros_like(neg_loss, dtype=torch.bool).to(feats.device)
                mask_neg[bro] = True
                neg_loss_ = neg_loss[~mask_neg]
                d_neg = dist_neg[i][~mask_neg]

                if args.box_self_adv == 1:
                    score = - d_neg * args.adv_temp
                    att = torch.softmax(score, dim=-1).detach()
                    neg_loss_ = att * neg_loss_
                neg_loss_ = neg_loss_.mean()
                neg_losses.append(neg_loss_)
            neg_losses = torch.stack(neg_losses)
            pos_loss = pos_loss.mean()
            neg_losses = neg_losses.mean()
            tot_loss = - neg_losses - pos_loss
    else:
        raise ValueError(f'{args.box_loss_type} not support')

    return tot_loss


def prototypical_logits_box_hinge_loss(feats, cluster_result, uq_idxs, all_uq_idxs, centers, offsets, margin=24,
                                       alpha=0.2):
    # uq_idxs is used to do label assign
    uq_idxs = torch.from_numpy(np.array([all_uq_idxs.index(item) for item in uq_idxs])).long().cuda()
    ins2cluster = torch.from_numpy(np.asarray(cluster_result['ins2cluster'])).long().cuda()

    boxes_max = centers + offsets
    boxes_min = centers - offsets

    # 维度扩展以支持广播计算
    feats_exp = feats.unsqueeze(1)  # (B, 1, D)
    boxes_max_exp = boxes_max.unsqueeze(0)  # (1, K, D)
    boxes_min_exp = boxes_min.unsqueeze(0)  # (1, K, D)
    centers_exp = centers.unsqueeze(0)  # (1, K, D)

    # 计算边界超出量（类似原代码中的score_offset）
    below_min = F.relu(boxes_min_exp - feats_exp)  # 特征小于min的部分
    above_max = F.relu(feats_exp - boxes_max_exp)  # 特征大于max的部分
    dist_out = below_min + above_max  # (B, K, D)

    # 计算中心调整值（类似原代码中的score_center_plus）
    clipped_feats = torch.minimum(
        boxes_max_exp,
        torch.maximum(boxes_min_exp, feats_exp)
    )
    dist_in = clipped_feats - centers_exp  # (B, K, D)

    # L_1
    dist = torch.norm(dist_out, p=1, dim=-1) + alpha * torch.norm(dist_in, p=1, dim=-1)
    # L_2^2
    # dist = torch.sqrt(torch.sum(dist_out ** 2, dim=-1)) + alpha * torch.sqrt(torch.sum(dist_in ** 2, dim=-1))

    labels_proto = ins2cluster[uq_idxs]  # 当前样本对应的原型 ID
    # 创建正样本的掩码
    mask_pos = torch.zeros_like(dist, dtype=torch.bool).cuda()
    mask_pos.scatter_(1, labels_proto.unsqueeze(1), True)  # (B, K)

    # 提取正负样本的距离
    dist_pos = dist[mask_pos]  # (B,)
    dist_pos = dist_pos.unsqueeze(1)
    # print(dist_pos.shape)
    dist_neg = dist[~mask_pos].view(dist.size(0), -1)  # (B, K-1)
    # print(dist_neg.shape)

    loss = torch.relu(dist_pos - dist_neg + margin)

    return loss.mean()


class Box_Intersection(nn.Module):
    def __init__(self, level_centers, level_offsets, level_info, n_neg=10, gumbel_beta=0.99):
        super(Box_Intersection, self).__init__()
        self.level_centers = level_centers
        self.level_offsets = level_offsets
        self.child = level_info['child']
        self.parent = level_info['parent']
        self.n_neg = n_neg
        self.rest_neg = n_neg
        self.gumbel_beta = gumbel_beta
        self.tanh = 0

        self.box1_center, self.box1_offset, self.box2_center, self.box2_offset = [], [], [], []

    def append(self, a, b, c, d, num):
        self.box1_center.extend([self.level_centers[a][b] for _ in range(num)])
        self.box1_offset.extend([self.level_offsets[a][b] for _ in range(num)])
        self.box2_center.extend(self.level_centers[c][d])
        self.box2_offset.extend(self.level_offsets[c][d])

    def forward(self, level_center=None, level_offsets=None, level_info=None, n_neg=None, gumbel_beta=None, tanh=None,
                inter_type='BCE'):
        if level_center is not None:
            self.level_centers = level_center
        if level_offsets is not None:
            self.level_offsets = level_offsets
        if level_info is not None:
            self.child = level_info['child']
            self.parent = level_info['parent']
        if n_neg is not None:
            self.n_neg = n_neg
            self.rest_neg = n_neg
        if gumbel_beta is not None:
            self.gumbel_beta = gumbel_beta
        if tanh is not None:
            self.tanh = tanh
        self.box1_center, self.box1_offset, self.box2_center, self.box2_offset = [], [], [], []

        level = len(self.level_centers)
        chunk_num = 0
        for cur_level in reversed(range(1, level)):
            for k, v in self.parent[cur_level].items():
                self.rest_neg = self.n_neg
                chunk_num += 1
                self.box1_center.append(self.level_centers[cur_level][k])
                self.box1_offset.append(self.level_offsets[cur_level][k])
                self.box2_center.append(self.level_centers[cur_level - 1][v])
                self.box2_offset.append(self.level_offsets[cur_level - 1][v])

                bro_num = len(self.child[cur_level - 1][v]) - 1
                tmp_idxes = np.where(self.child[cur_level - 1][v] != k)[0]
                bro = self.child[cur_level - 1][v][tmp_idxes]
                if cur_level >= 2:
                    p_parent = self.parent[cur_level - 1][v]
                    uncle_num = len(self.child[cur_level - 2][p_parent]) - 1
                    tmp = self.child[cur_level - 2][p_parent]
                    tmp_idxes = np.where(tmp != v)[0]
                    uncle = tmp[tmp_idxes]
                    cousin = []
                    for i in uncle:
                        cousin.extend(self.child[cur_level - 1][i])
                    cousin_num = len(cousin)
                else:
                    uncle = []
                    cousin = []
                    uncle_num = 0
                    cousin_num = 0

                num = [bro_num, uncle_num, cousin_num]
                indexes = [bro, uncle, cousin]
                add_level = [cur_level, cur_level - 1, cur_level]

                for i, i_num, in enumerate(num):
                    if i_num >= self.rest_neg:
                        select_num = self.rest_neg
                        self.rest_neg = 0
                    elif i_num > 0:
                        select_num = i_num
                        self.rest_neg -= i_num
                    else:
                        break

                    if select_num > 0:
                        select_idxes = np.random.choice(indexes[i], size=select_num, replace=False)
                        self.append(cur_level, k, add_level[i], select_idxes, select_num)
                    if self.rest_neg == 0:
                        break

                if self.rest_neg > 0:
                    tmp_level = cur_level - 1
                    tmp_p = v
                    while tmp_level > 0:
                        tmp_p = self.parent[tmp_level][tmp_p]
                        tmp_level -= 1
                        if tmp_level == 0:
                            break
                    start = []
                    for kk, vv in self.child[tmp_level].items():
                        if kk != tmp_p:
                            start.append(kk)
                    tmp_num = len(start)
                    while True:
                        if tmp_num >= self.rest_neg:
                            select_num = self.rest_neg
                            self.rest_neg = 0
                        elif tmp_num > 0:
                            select_num = tmp_num
                            self.rest_neg -= tmp_num
                        else:
                            break
                        if select_num > 0:
                            select_idxes = np.random.choice(start, size=select_num, replace=False)
                            self.append(cur_level, k, tmp_level, select_idxes, select_num)
                        if self.rest_neg == 0:
                            break
                        if tmp_level == level - 1:
                            break
                        tmp_start = []
                        for pos in start:
                            tmp_start.extend(self.child[tmp_level][pos])
                        start = tmp_start
                        tmp_level += 1
                        tmp_num = len(start)
                    if self.rest_neg > 0:
                        select_idxes = np.random.choice(list(range(self.n_neg - self.rest_neg)),
                                                        size=self.rest_neg, replace=True)
                        select_idxes = - select_idxes - 1
                        self.box1_center.extend([self.level_centers[cur_level][k] for _ in range(self.rest_neg)])
                        self.box1_offset.extend([self.level_offsets[cur_level][k] for _ in range(self.rest_neg)])
                        for ad, idx in enumerate(select_idxes):
                            self.box2_center.append(self.box2_center[idx - ad])
                            self.box2_offset.append(self.box2_offset[idx - ad])

        self.box1_center = torch.stack(self.box1_center)
        self.box1_offset = torch.stack(self.box1_offset)
        self.box2_center = torch.stack(self.box2_center)
        self.box2_offset = torch.stack(self.box2_offset)
        inter_vol = cal_logit_box(self.box1_center, self.box1_offset, self.box2_center, self.box2_offset,
                                  self.gumbel_beta, self.tanh)
        assert len(inter_vol) % (1 + self.n_neg) == 0, 'probs cant div by (1+n_neg)'
        if inter_type == 'BPR':
            chunk_vol = inter_vol.view(chunk_num, -1)
            pos_vol = chunk_vol[:, 0]
            pos_vol = pos_vol.unsqueeze(1)
            neg_vol = chunk_vol[:, 1:]

            loss = -1 * torch.mean(torch.nn.functional.logsigmoid(pos_vol - neg_vol))
            return loss
        elif inter_type == 'BCE':
            if self.tanh == 0:
                max_x = self.box2_center + self.box2_offset
                min_x = self.box2_center - self.box2_offset
            else:
                max_x = torch.tanh(self.box2_center + self.box2_offset)
                min_x = torch.tanh(self.box2_center - self.box2_offset)

            euler_gamma = 0.57721566490153286060
            x_vol = torch.sum(
                torch.log(F.softplus(max_x - min_x - 2 * euler_gamma * self.gumbel_beta,
                                     beta=1 / self.gumbel_beta) + 1e-23
                          ),
                dim=-1,
            )
            probs = inter_vol - x_vol
            probs = probs.view(chunk_num, -1)
            target = torch.zeros_like(probs, dtype=torch.float)
            target[:, 0] = 1
            return probs, target


class Box_Intersection_Taxo(nn.Module):
    def __init__(self, level_centers, level_offsets, level_info, n_neg=10, margin=0.01, eps=-0.01):
        super(Box_Intersection_Taxo, self).__init__()
        self.level_centers = level_centers
        self.level_offsets = level_offsets
        self.child = level_info['child']
        self.parent = level_info['parent']
        self.n_neg = n_neg
        self.rest_neg = n_neg
        self.tanh = 0
        self.margin = margin
        self.eps = eps

        self.box1_center, self.box1_offset, self.box2_center, self.box2_offset = [], [], [], []

        self.par_chd_left_loss = nn.MSELoss()
        self.par_chd_right_loss = nn.MSELoss()
        self.par_chd_negative_loss = nn.MSELoss()
        self.positive_prob_loss = nn.MSELoss()
        self.negative_prob_loss = nn.MSELoss()

    def box_volumn(self, delta):
        flag = torch.sum(delta <= 0, -1)
        product = torch.prod(delta, -1)
        zeros = torch.zeros_like(product)
        ones = torch.ones_like(product)
        mask = torch.where(flag == 0, ones, zeros)
        volumn = torch.mul(product, mask)

        return volumn

    def condition_score(self, child_center, child_delta, parent_center, parent_delta):
        inter_left, inter_right = self.box_intersection(child_center, child_delta, parent_center, parent_delta)
        inter_delta = (inter_right - inter_left) / 2
        flag = (inter_delta <= 0)
        zeros = torch.zeros_like(flag)
        ones = torch.ones_like(flag)
        mask = torch.where(flag == False, ones, zeros)
        masked_inter_delta = torch.mul(inter_delta, mask)

        score_pre = torch.div(masked_inter_delta, child_delta + 1e-6)
        score = torch.prod(score_pre, -1)

        parent_volumn = self.box_volumn(parent_delta)

        return score.squeeze(), parent_volumn.squeeze()

    def parent_child_contain_loss(self, parent_center, parent_delta, child_center, child_delta):
        parent_left = parent_center - parent_delta
        parent_right = parent_center + parent_delta

        child_left = child_center - child_delta
        child_right = child_center + child_delta

        diff_left = child_left - parent_left
        zeros = torch.zeros_like(diff_left)
        ones = torch.ones_like(diff_left)
        margins = torch.ones_like(diff_left) * self.margin
        left_mask = torch.where(diff_left < self.margin, ones, zeros)
        left_loss = self.par_chd_left_loss(torch.mul(diff_left, left_mask), torch.mul(margins, left_mask))

        diff_right = parent_right - child_right
        zeros = torch.zeros_like(diff_right)
        ones = torch.ones_like(diff_right)
        margins = torch.ones_like(diff_right) * self.margin
        right_mask = torch.where(diff_right < self.margin, ones, zeros)
        right_loss = self.par_chd_right_loss(torch.mul(diff_right, right_mask), torch.mul(margins, right_mask))

        return (left_loss + right_loss) / 2

    def parent_child_contain_loss_prob(self, parent_center, parent_delta, child_center, child_delta):
        score, _ = self.condition_score(child_center, child_delta, parent_center, parent_delta)
        ones = torch.ones_like(score)
        loss = self.positive_prob_loss(score, ones)

        return loss

    def box_intersection(self, center1, delta1, center2, delta2):
        left1 = center1 - delta1
        right1 = center1 + delta1
        left2 = center2 - delta2
        right2 = center2 + delta2
        inter_left = torch.max(left1, left2)
        inter_right = torch.min(right1, right2)

        return inter_left, inter_right

    def negative_contain_loss(self, child_center, child_delta, neg_parent_center, neg_parent_delta):
        inter_left, inter_right = self.box_intersection(child_center, child_delta, neg_parent_center, neg_parent_delta)

        inter_delta = (inter_right - inter_left) / 2
        zeros = torch.zeros_like(inter_delta)
        ones = torch.ones_like(inter_delta)
        epsilon = torch.ones_like(inter_delta) * self.eps
        inter_mask = torch.where(inter_delta > self.eps, ones, zeros)
        inter_loss = self.par_chd_negative_loss(torch.mul(inter_delta, inter_mask), torch.mul(epsilon, inter_mask))

        return inter_loss

    def negative_contain_loss_prob(self, child_center, child_delta, neg_parent_center, neg_parent_delta):
        score, _ = self.condition_score(child_center, child_delta, neg_parent_center, neg_parent_delta)
        zeros = torch.zeros_like(score)
        loss = self.negative_prob_loss(score, zeros)

        return loss

    def append(self, c, d):
        self.box2_center.extend(self.level_centers[c][d])
        self.box2_offset.extend(self.level_offsets[c][d])

    def forward(self, level_center=None, level_offsets=None, level_info=None, n_neg=None, alpha=1.0, extra=1.0):
        if level_center is not None:
            self.level_centers = level_center
        if level_offsets is not None:
            self.level_offsets = level_offsets
        if level_info is not None:
            self.child = level_info['child']
            self.parent = level_info['parent']
        if n_neg is not None:
            self.n_neg = n_neg
            self.rest_neg = n_neg
        self.box1_center, self.box1_offset, self.box2_center, self.box2_offset = [], [], [], []

        level = len(self.level_centers)
        chunk_num = 0
        for cur_level in reversed(range(1, level)):
            for k, v in self.parent[cur_level].items():
                self.rest_neg = self.n_neg
                chunk_num += 1
                self.box1_center.append(self.level_centers[cur_level][k])
                self.box1_offset.append(self.level_offsets[cur_level][k])
                self.box2_center.append(self.level_centers[cur_level - 1][v])
                self.box2_offset.append(self.level_offsets[cur_level - 1][v])

                bro_num = len(self.child[cur_level - 1][v]) - 1
                tmp_idxes = np.where(self.child[cur_level - 1][v] != k)[0]
                bro = self.child[cur_level - 1][v][tmp_idxes]
                if cur_level >= 2:
                    p_parent = self.parent[cur_level - 1][v]
                    uncle_num = len(self.child[cur_level - 2][p_parent]) - 1
                    tmp = self.child[cur_level - 2][p_parent]
                    tmp_idxes = np.where(tmp != v)[0]
                    uncle = tmp[tmp_idxes]
                    cousin = []
                    for i in uncle:
                        cousin.extend(self.child[cur_level - 1][i])
                    cousin_num = len(cousin)
                else:
                    uncle = []
                    cousin = []
                    uncle_num = 0
                    cousin_num = 0

                num = [bro_num, uncle_num, cousin_num]
                indexes = [bro, uncle, cousin]
                add_level = [cur_level, cur_level - 1, cur_level]

                for i, i_num, in enumerate(num):
                    if i_num >= self.rest_neg:
                        select_num = self.rest_neg
                        self.rest_neg = 0
                    elif i_num > 0:
                        select_num = i_num
                        self.rest_neg -= i_num
                    else:
                        break

                    if select_num > 0:
                        select_idxes = np.random.choice(indexes[i], size=select_num, replace=False)
                        self.append(add_level[i], select_idxes)
                    if self.rest_neg == 0:
                        break

                if self.rest_neg > 0:
                    tmp_level = cur_level - 1
                    tmp_p = v
                    while tmp_level > 0:
                        tmp_p = self.parent[tmp_level][tmp_p]
                        tmp_level -= 1
                        if tmp_level == 0:
                            break
                    start = []
                    for kk, vv in self.child[tmp_level].items():
                        if kk != tmp_p:
                            start.append(kk)
                    tmp_num = len(start)
                    while True:
                        if tmp_num >= self.rest_neg:
                            select_num = self.rest_neg
                            self.rest_neg = 0
                        elif tmp_num > 0:
                            select_num = tmp_num
                            self.rest_neg -= tmp_num
                        else:
                            break
                        if select_num > 0:
                            select_idxes = np.random.choice(start, size=select_num, replace=False)
                            self.append(tmp_level, select_idxes)
                        if self.rest_neg == 0:
                            break
                        if tmp_level == level - 1:
                            break
                        tmp_start = []
                        for pos in start:
                            tmp_start.extend(self.child[tmp_level][pos])
                        start = tmp_start
                        tmp_level += 1
                        tmp_num = len(start)
                    if self.rest_neg > 0:
                        select_idxes = np.random.choice(list(range(self.n_neg - self.rest_neg)),
                                                        size=self.rest_neg, replace=True)
                        select_idxes = - select_idxes - 1
                        for ad, idx in enumerate(select_idxes):
                            self.box2_center.append(self.box2_center[idx - ad])
                            self.box2_offset.append(self.box2_offset[idx - ad])

        self.box1_center = torch.stack(self.box1_center)
        self.box1_offset = torch.stack(self.box1_offset)
        self.box2_center = torch.stack(self.box2_center)
        self.box2_offset = torch.stack(self.box2_offset)
        self.box2_center = self.box2_center.view(chunk_num, (1 + self.n_neg), -1)
        self.box2_offset = self.box2_offset.view(chunk_num, (1 + self.n_neg), -1)
        parent_center = self.box2_center[:, 0]
        parent_delta = self.box2_offset[:, 0]
        neg_parent_center = self.box2_center[:, 1:]
        neg_parent_delta = self.box2_offset[:, 1:]
        parent_child_contain_loss = self.parent_child_contain_loss(parent_center, parent_delta, self.box1_center,
                                                                   self.box1_offset)
        parent_child_contain_loss_prob = self.parent_child_contain_loss_prob(parent_center, parent_delta,
                                                                             self.box1_center, self.box1_offset)

        child_parent_negative_loss = self.negative_contain_loss(self.box1_center.unsqueeze(1),
                                                                self.box1_offset.unsqueeze(1), neg_parent_center,
                                                                neg_parent_delta)
        child_parent_negative_loss_prob = self.negative_contain_loss_prob(self.box1_center.unsqueeze(1),
                                                                          self.box1_offset.unsqueeze(1),
                                                                          neg_parent_center, neg_parent_delta)

        loss = alpha * (parent_child_contain_loss + child_parent_negative_loss) + extra * (
                parent_child_contain_loss_prob + child_parent_negative_loss_prob)
        return loss, parent_child_contain_loss, parent_child_contain_loss_prob, child_parent_negative_loss, child_parent_negative_loss_prob


class GradNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)  # [batch, dim]
        return torch.sum(torch.log(input), dim=-1) + torch.log(torch.tensor(1.0))

    @staticmethod
    def backward(ctx, grad_chain):
        input, = ctx.saved_tensors
        # = 1 / X_i [batch, dim]
        log_FX = torch.sum(torch.log(input), dim=-1) + torch.log(torch.tensor(1.0))
        FX = torch.exp(log_FX)  # [batch, ]
        grad_input = 1. / input
        max_grad_input = torch.max(grad_input, 1).values.reshape(-1, 1).repeat(1, input.shape[1])
        norm_grad_input = (grad_input / max_grad_input) * (1e-2 / FX).reshape(-1, 1).repeat(1, input.shape[1])

        # print(norm_grad_input.shape, grad_chain.shape)
        return norm_grad_input * grad_chain.reshape(-1, 1).repeat(1, input.shape[1])


class Box_Intersection_boxplus(nn.Module):
    def __init__(self, level_centers, level_offsets, level_info, n_neg=10):
        super(Box_Intersection_boxplus, self).__init__()
        self.level_centers = level_centers
        self.level_offsets = level_offsets
        self.child = level_info['child']
        self.parent = level_info['parent']
        self.n_neg = n_neg
        self.rest_neg = n_neg
        self.gradnorm = GradNorm.apply
        self.overlap_loss = torch.nn.CrossEntropyLoss()

        self.box1_center, self.box1_offset, self.box2_center, self.box2_offset = [], [], [], []

    def append(self, c, d):
        self.box2_center.extend(self.level_centers[c][d])
        self.box2_offset.extend(self.level_offsets[c][d])

    def get_cond_probs(self, boxes1_center, boxes1_offset, boxes2_center, boxes2_offset):
        left, right = self.intersection(boxes1_center, boxes1_offset, boxes2_center, boxes2_offset)
        Index_overlap, Index_disjoint, log_inter_measure = self.volumes(left, right)
        left = boxes1_center - boxes1_offset
        right = boxes1_center + boxes1_offset
        _, _, log_box1 = self.volumes(left, right)
        log_box1 = torch.repeat_interleave(log_box1, repeats=int(len(log_inter_measure) / len(log_box1)), dim=0)
        measures = torch.zeros_like(log_inter_measure)
        if Index_overlap.shape[0] > 0:
            measures[Index_overlap] = torch.exp(log_inter_measure[Index_overlap] - log_box1[Index_overlap])
        if Index_disjoint.shape[0] > 0:
            measures[Index_disjoint] = log_inter_measure[Index_disjoint]

        return Index_overlap, Index_disjoint, measures

    def volumes(self, left, right):
        eps = torch.finfo(left.dtype).tiny  # type: ignore

        # overlap mark
        boxLens = right - left

        Tag = (torch.min(boxLens, dim=-1).values > 0.0)
        tmp_tag = Tag.view(-1)
        Index_overlap = torch.where(tmp_tag == True)[0]
        Index_disjoint = torch.where(tmp_tag == False)[0]
        pnorm = 1
        measures = torch.zeros_like(tmp_tag, dtype=torch.float)

        boxLens = boxLens.view(boxLens.shape[0] * boxLens.shape[1], -1)
        zeros = torch.zeros_like(boxLens)
        if Index_overlap.shape[0] > 0:
            measures[Index_overlap] = self.gradnorm(
                torch.max(zeros[Index_overlap], boxLens[Index_overlap]).clamp_min(eps))
        if Index_disjoint.shape[0] > 0:
            measures[Index_disjoint] = torch.norm(torch.max(zeros[Index_disjoint], -boxLens[Index_disjoint]), p=pnorm,
                                                  dim=-1)

        return Index_overlap, Index_disjoint, measures

    def intersection(self, center1, delta1, center2, delta2):
        left1 = center1 - delta1
        right1 = center1 + delta1
        left2 = center2 - delta2
        right2 = center2 + delta2
        inter_left = torch.max(left1, left2)
        inter_right = torch.min(right1, right2)

        return inter_left, inter_right

    def forward(self, level_center=None, level_offsets=None, level_info=None, n_neg=None):
        if level_center is not None:
            self.level_centers = level_center
        if level_offsets is not None:
            self.level_offsets = level_offsets
        if level_info is not None:
            self.child = level_info['child']
            self.parent = level_info['parent']
        if n_neg is not None:
            self.n_neg = n_neg
            self.rest_neg = n_neg
        self.box1_center, self.box1_offset, self.box2_center, self.box2_offset = [], [], [], []

        level = len(self.level_centers)
        chunk_num = 0
        for cur_level in reversed(range(1, level)):
            for k, v in self.parent[cur_level].items():
                self.rest_neg = self.n_neg
                chunk_num += 1
                self.box1_center.append(self.level_centers[cur_level][k])
                self.box1_offset.append(self.level_offsets[cur_level][k])
                self.box2_center.append(self.level_centers[cur_level - 1][v])
                self.box2_offset.append(self.level_offsets[cur_level - 1][v])

                bro_num = len(self.child[cur_level - 1][v]) - 1
                tmp_idxes = np.where(self.child[cur_level - 1][v] != k)[0]
                bro = self.child[cur_level - 1][v][tmp_idxes]
                if cur_level >= 2:
                    p_parent = self.parent[cur_level - 1][v]
                    uncle_num = len(self.child[cur_level - 2][p_parent]) - 1
                    tmp = self.child[cur_level - 2][p_parent]
                    tmp_idxes = np.where(tmp != v)[0]
                    uncle = tmp[tmp_idxes]
                    cousin = []
                    for i in uncle:
                        cousin.extend(self.child[cur_level - 1][i])
                    cousin_num = len(cousin)
                else:
                    uncle = []
                    cousin = []
                    uncle_num = 0
                    cousin_num = 0

                num = [bro_num, uncle_num, cousin_num]
                indexes = [bro, uncle, cousin]
                add_level = [cur_level, cur_level - 1, cur_level]

                for i, i_num, in enumerate(num):
                    if i_num >= self.rest_neg:
                        select_num = self.rest_neg
                        self.rest_neg = 0
                    elif i_num > 0:
                        select_num = i_num
                        self.rest_neg -= i_num
                    else:
                        break

                    if select_num > 0:
                        select_idxes = np.random.choice(indexes[i], size=select_num, replace=False)
                        self.append(add_level[i], select_idxes)
                    if self.rest_neg == 0:
                        break

                if self.rest_neg > 0:
                    tmp_level = cur_level - 1
                    tmp_p = v
                    while tmp_level > 0:
                        tmp_p = self.parent[tmp_level][tmp_p]
                        tmp_level -= 1
                        if tmp_level == 0:
                            break
                    start = []
                    for kk, vv in self.child[tmp_level].items():
                        if kk != tmp_p:
                            start.append(kk)
                    tmp_num = len(start)
                    while True:
                        if tmp_num >= self.rest_neg:
                            select_num = self.rest_neg
                            self.rest_neg = 0
                        elif tmp_num > 0:
                            select_num = tmp_num
                            self.rest_neg -= tmp_num
                        else:
                            break
                        if select_num > 0:
                            select_idxes = np.random.choice(start, size=select_num, replace=False)
                            self.append(tmp_level, select_idxes)
                        if self.rest_neg == 0:
                            break
                        if tmp_level == level - 1:
                            break
                        tmp_start = []
                        for pos in start:
                            tmp_start.extend(self.child[tmp_level][pos])
                        start = tmp_start
                        tmp_level += 1
                        tmp_num = len(start)
                    if self.rest_neg > 0:
                        select_idxes = np.random.choice(list(range(self.n_neg - self.rest_neg)),
                                                        size=self.rest_neg, replace=True)
                        select_idxes = - select_idxes - 1
                        for ad, idx in enumerate(select_idxes):
                            self.box2_center.append(self.box2_center[idx - ad])
                            self.box2_offset.append(self.box2_offset[idx - ad])

        self.box1_center = torch.stack(self.box1_center)
        self.box1_offset = torch.stack(self.box1_offset)
        self.box2_center = torch.stack(self.box2_center)
        self.box2_offset = torch.stack(self.box2_offset)

        self.box1_center = self.box1_center.unsqueeze(1)
        self.box1_offset = self.box1_offset.unsqueeze(1)
        self.box2_center = self.box2_center.view(chunk_num, (1 + self.n_neg), -1)
        self.box2_offset = self.box2_offset.view(chunk_num, (1 + self.n_neg), -1)

        Index_overlap, Index_disjoint, pos_predictions = self.get_cond_probs(self.box1_center, self.box1_offset,
                                                                             self.box2_center, self.box2_offset)
        input_condprob_step = torch.zeros_like(pos_predictions, dtype=torch.long)
        input_condprob_step = input_condprob_step.view(chunk_num, -1)
        input_condprob_step[:, 0] = 1
        input_condprob_step = input_condprob_step.view(-1)

        neg_prediction = torch.ones_like(pos_predictions) - pos_predictions
        prediction = torch.stack([neg_prediction, pos_predictions], dim=-1)

        if Index_overlap.shape[0] > 0:
            loss_overlap = self.overlap_loss(prediction[Index_overlap], input_condprob_step[Index_overlap])
        else:
            loss_overlap = torch.zeros(1).to(level_center[0].device)
        if Index_disjoint.shape[0] > 0:
            loss_disjoint = (input_condprob_step[Index_disjoint] * prediction[Index_disjoint][:, 1]).mean()
        else:
            loss_disjoint = torch.zeros(1).to(level_center[0].device)

        loss = loss_overlap + 2 * loss_disjoint
        return loss, loss_overlap, loss_disjoint


class Box_intersection_all(nn.Module):
    def __init__(self, margin=0.01, eps=-0.01):
        super(Box_intersection_all, self).__init__()
        self.margin = margin
        self.eps = eps

        self.par_chd_left_loss = nn.MSELoss()  # reduction='none'
        self.par_chd_right_loss = nn.MSELoss()
        self.par_chd_negative_loss = nn.MSELoss()
        self.positive_prob_loss = nn.MSELoss()
        self.negative_prob_loss = nn.MSELoss()

        self.gradnorm = GradNorm.apply
        self.overlap_loss = torch.nn.CrossEntropyLoss()

    def get_cond_probs(self, boxes1_center, boxes1_offset, boxes2_center, boxes2_offset):
        left, right = self.box_intersection(boxes1_center, boxes1_offset, boxes2_center, boxes2_offset)
        Index_overlap, Index_disjoint, log_inter_measure = self.volumes(left, right)
        left = boxes1_center - boxes1_offset
        right = boxes1_center + boxes1_offset
        _, _, log_box1 = self.volumes(left, right)
        log_box1 = torch.repeat_interleave(log_box1, repeats=int(len(log_inter_measure) / len(log_box1)), dim=0)
        measures = torch.zeros_like(log_inter_measure)
        if Index_overlap.shape[0] > 0:
            measures[Index_overlap] = torch.exp(log_inter_measure[Index_overlap] - log_box1[Index_overlap])
        if Index_disjoint.shape[0] > 0:
            measures[Index_disjoint] = log_inter_measure[Index_disjoint]

        return Index_overlap, Index_disjoint, measures

    def volumes(self, left, right):
        eps = torch.finfo(left.dtype).tiny  # type: ignore

        # overlap mark
        boxLens = right - left

        Tag = (torch.min(boxLens, dim=-1).values > 0.0)
        tmp_tag = Tag.view(-1)
        Index_overlap = torch.where(tmp_tag == True)[0]
        Index_disjoint = torch.where(tmp_tag == False)[0]
        pnorm = 2
        measures = torch.zeros_like(tmp_tag, dtype=torch.float)

        boxLens = boxLens.view(boxLens.shape[0] * boxLens.shape[1], -1)
        zeros = torch.zeros_like(boxLens)
        if Index_overlap.shape[0] > 0:
            measures[Index_overlap] = self.gradnorm(
                torch.max(zeros[Index_overlap], boxLens[Index_overlap]).clamp_min(eps))
        if Index_disjoint.shape[0] > 0:
            measures[Index_disjoint] = torch.norm(torch.max(zeros[Index_disjoint], -boxLens[Index_disjoint]), p=pnorm,
                                                  dim=-1)

        return Index_overlap, Index_disjoint, measures

    def box_volumn(self, delta):
        flag = torch.sum(delta <= 0, -1)
        product = torch.prod(delta, -1)
        zeros = torch.zeros_like(product)
        ones = torch.ones_like(product)
        mask = torch.where(flag == 0, ones, zeros)
        volumn = torch.mul(product, mask)

        return volumn

    def condition_score(self, child_center, child_delta, parent_center, parent_delta):
        inter_left, inter_right = self.box_intersection(child_center, child_delta, parent_center, parent_delta)
        inter_delta = (inter_right - inter_left) / 2
        flag = (inter_delta <= 0)
        zeros = torch.zeros_like(flag)
        ones = torch.ones_like(flag)
        mask = torch.where(flag == False, ones, zeros)
        masked_inter_delta = torch.mul(inter_delta, mask)

        score_pre = torch.div(masked_inter_delta, child_delta + 1e-6)
        score = torch.prod(score_pre, -1)

        # parent_volumn = self.box_volumn(parent_delta)

        return score.squeeze(), None

    def parent_child_contain_loss(self, parent_center, parent_delta, child_center, child_delta, rate):
        parent_left = (parent_center - parent_delta)
        parent_right = (parent_center + parent_delta)

        child_left = (child_center - child_delta)
        child_right = (child_center + child_delta)

        diff_left = child_left - parent_left
        zeros = torch.zeros_like(diff_left)
        ones = torch.ones_like(diff_left)
        margins = torch.ones_like(diff_left) * self.margin
        left_mask = torch.where(diff_left < self.margin, ones, zeros)
        left_loss = self.par_chd_left_loss(torch.mul(diff_left, left_mask), torch.mul(margins, left_mask))
        # left_loss = left_loss.mean(dim=-1) * rate

        diff_right = parent_right - child_right
        zeros = torch.zeros_like(diff_right)
        ones = torch.ones_like(diff_right)
        margins = torch.ones_like(diff_right) * self.margin
        right_mask = torch.where(diff_right < self.margin, ones, zeros)
        right_loss = self.par_chd_right_loss(torch.mul(diff_right, right_mask), torch.mul(margins, right_mask))
        # right_loss = right_loss.mean(dim=-1) * rate

        return (left_loss + right_loss) / 2

    def parent_child_contain_loss_prob(self, parent_center, parent_delta, child_center, child_delta, rate):
        score, _ = self.condition_score(child_center, child_delta, parent_center, parent_delta)
        ones = torch.ones_like(score)
        loss = self.positive_prob_loss(score, ones)
        # loss = loss * rate
        return loss

    def box_intersection(self, center1, delta1, center2, delta2):
        left1 = (center1 - delta1)
        right1 = (center1 + delta1)
        left2 = (center2 - delta2)
        right2 = (center2 + delta2)
        inter_left = torch.max(left1, left2)
        inter_right = torch.min(right1, right2)

        return inter_left, inter_right

    def negative_contain_loss(self, child_center, child_delta, neg_parent_center, neg_parent_delta, rate):
        inter_left, inter_right = self.box_intersection(child_center, child_delta, neg_parent_center, neg_parent_delta)

        inter_delta = (inter_right - inter_left) / 2
        zeros = torch.zeros_like(inter_delta)
        ones = torch.ones_like(inter_delta)
        epsilon = torch.ones_like(inter_delta) * self.eps
        inter_mask = torch.where(inter_delta > self.eps, ones, zeros)
        inter_loss = self.par_chd_negative_loss(torch.mul(inter_delta, inter_mask), torch.mul(epsilon, inter_mask))
        # inter_loss = inter_loss.mean(dim=-1) * rate

        return inter_loss

    def negative_contain_loss_prob(self, child_center, child_delta, neg_parent_center, neg_parent_delta, rate):
        score, _ = self.condition_score(child_center, child_delta, neg_parent_center, neg_parent_delta)
        zeros = torch.zeros_like(score)
        loss = self.negative_prob_loss(score, zeros)
        # loss = loss * rate
        return loss

    def IoU_intersection(self, center1, delta1, center2, delta2):
        score, _ = self.condition_score(center1, delta1, center2, delta2)
        dist_center = (center1 - center2).abs()
        # test 1
        left1 = center1 - delta1
        right1 = center1 + delta1
        left2 = center2 - delta2
        right2 = center2 + delta2
        inter_left = torch.min(left1, left2)
        inter_right = torch.max(right1, right2)
        dist_C = (inter_right - inter_left).abs()
        post = torch.norm(dist_center, p=2, dim=-1) / torch.norm(dist_C, p=2, dim=-1)
        post = post ** 2
        # test 2
        # delta = (dist_center - delta1 - delta2)/delta1
        # delta_temp = delta / 0.07
        # post = torch.sigmoid(delta_temp).mean(dim=-1)
        return score, post

    def forward(self, level_centers, level_offsets, level_info, level_inboxRate, n_neg, gumbel_beta=0.01, tanh=0,
                alpha=1.0, extra=0.1,
                IoU_margin=2, mode='B4T', IoU_mode='margin', d_alpha=1.0):
        box1_center = []
        box1_offset = []
        box2_center = []
        box2_offset = []
        child = level_info['child']
        parent = level_info['parent']
        box1_depth = []
        box2_depth = []
        together_depth = []
        together_rate = []

        level = len(level_centers)
        chunk_num = 0
        for cur_level in reversed(range(1, level)):
            for k, v in parent[cur_level].items():
                rest_neg = n_neg
                chunk_num += 1
                box1_center.append(level_centers[cur_level][k])
                box1_offset.append(level_offsets[cur_level][k])
                box2_center.append(level_centers[cur_level - 1][v])
                box2_offset.append(level_offsets[cur_level - 1][v])

                box1_depth.append(cur_level + 2)
                box2_depth.append(cur_level + 1)
                together_depth.append(cur_level + 1)

                together_rate.append((level_inboxRate[cur_level][k] + level_inboxRate[cur_level - 1][v]) / 2)

                bro_num = len(child[cur_level - 1][v]) - 1
                tmp_idxes = np.where(child[cur_level - 1][v] != k)[0]
                bro = child[cur_level - 1][v][tmp_idxes]
                if cur_level >= 2:
                    p_parent = parent[cur_level - 1][v]
                    uncle_num = len(child[cur_level - 2][p_parent]) - 1
                    tmp = child[cur_level - 2][p_parent]
                    tmp_idxes = np.where(tmp != v)[0]
                    uncle = tmp[tmp_idxes]
                    cousin = []
                    for i in uncle:
                        cousin.extend(child[cur_level - 1][i])
                    cousin_num = len(cousin)
                else:
                    uncle = []
                    cousin = []
                    uncle_num = 0
                    cousin_num = 0

                num = [bro_num, uncle_num, cousin_num]
                indexes = [bro, uncle, cousin]
                add_level = [cur_level, cur_level - 1, cur_level]

                for i, i_num, in enumerate(num):
                    if i_num >= rest_neg:
                        select_num = rest_neg
                        rest_neg = 0
                    elif i_num > 0:
                        select_num = i_num
                        rest_neg -= i_num
                    else:
                        break

                    if select_num > 0:
                        select_idxes = np.random.choice(indexes[i], size=select_num, replace=False)
                        box2_center.extend(level_centers[add_level[i]][select_idxes])
                        box2_offset.extend(level_offsets[add_level[i]][select_idxes])

                        box2_depth.extend([add_level[i] + 2 for _ in select_idxes])
                        if i == 0:
                            together_depth.extend([cur_level + 1 for _ in select_idxes])
                        else:
                            together_depth.extend([cur_level for _ in select_idxes])
                        together_rate.extend(
                            (level_inboxRate[cur_level][k] + level_inboxRate[add_level[i]][select_idxes]) / 2)
                    if rest_neg == 0:
                        break

                if rest_neg > 0:
                    tmp_level = cur_level - 1
                    tmp_p = v
                    while tmp_level > 0:
                        tmp_p = parent[tmp_level][tmp_p]
                        tmp_level -= 1
                        if tmp_level == 0:
                            break
                    start = []
                    for kk, vv in child[tmp_level].items():
                        if kk != tmp_p:
                            start.append(kk)
                    tmp_num = len(start)
                    while True:
                        if tmp_num >= rest_neg:
                            select_num = rest_neg
                            rest_neg = 0
                        elif tmp_num > 0:
                            select_num = tmp_num
                            rest_neg -= tmp_num
                        else:
                            break
                        if select_num > 0:
                            select_idxes = np.random.choice(start, size=select_num, replace=False)
                            box2_center.extend(level_centers[tmp_level][select_idxes])
                            box2_offset.extend(level_offsets[tmp_level][select_idxes])

                            box2_depth.extend([tmp_level + 2 for _ in select_idxes])
                            together_depth.extend([1 for _ in select_idxes])

                            together_rate.extend(
                                (level_inboxRate[cur_level][k] + level_inboxRate[tmp_level][select_idxes]) / 2)

                        if rest_neg == 0:
                            break
                        if tmp_level == level - 1:
                            break
                        tmp_start = []
                        for pos in start:
                            tmp_start.extend(child[tmp_level][pos])
                        start = tmp_start
                        tmp_level += 1
                        tmp_num = len(start)

                    if rest_neg > 0:
                        select_idxes = np.random.choice(list(range(n_neg - rest_neg)),
                                                        size=rest_neg, replace=True)
                        select_idxes = - select_idxes - 1
                        for ad, idx in enumerate(select_idxes):
                            box2_center.append(box2_center[idx - ad])
                            box2_offset.append(box2_center[idx - ad])

                            box2_depth.append(box2_depth[idx - ad])
                            together_depth.append(together_depth[idx - ad])

                            together_rate.append(together_rate[idx - ad])

        box1_center = torch.stack(box1_center)
        box1_offset = torch.stack(box1_offset)
        box2_center = torch.stack(box2_center)
        box2_offset = torch.stack(box2_offset)

        box1_depth = torch.tensor(box1_depth, dtype=torch.int32).to(level_centers[0].device)
        box2_depth = torch.tensor(box2_depth, dtype=torch.int32).to(level_centers[0].device)
        together_depth = torch.tensor(together_depth, dtype=torch.int32).to(level_centers[0].device)

        together_rate = torch.stack(together_rate)

        # box1_centers_exp = box1_centers.unsqueeze(1)
        # box1_offsets_exp = box1_offsets.unsqueeze(1)
        box2_center = box2_center.view(chunk_num, (1 + n_neg), -1)
        box2_offset = box2_offset.view(chunk_num, (1 + n_neg), -1)

        box2_depth = box2_depth.view(chunk_num, -1)
        together_depth = together_depth.view(chunk_num, -1)

        together_rate = together_rate.view(chunk_num, -1)

        if mode == 'B4T':
            box_inter_vol = cal_logit_box(box1_center.unsqueeze(1), box1_offset.unsqueeze(1), box2_center,
                                          box2_offset,
                                          gumbel_beta, tanh)
            if tanh == 0:
                x_max = box1_center + box1_offset
                x_min = box1_center - box1_offset
            else:
                x_max = torch.tanh(box1_center + box1_offset)
                x_min = torch.tanh(box1_center - box1_offset)
            euler_gamma = 0.57721566490153286060
            x_vol = log_soft_volume(x_max, x_min, euler_gamma, gumbel_beta)
            probs = box_inter_vol - x_vol.unsqueeze(1)
            targets = torch.zeros_like(probs)
            targets[:, 0] = 1
            return probs, targets
        elif mode == 'taxo':
            parent_center = box2_center[:, 0]
            parent_delta = box2_offset[:, 0]
            neg_parent_center = box2_center[:, 1:]
            neg_parent_delta = box2_offset[:, 1:]
            pos_rate = together_rate[:, 0]
            neg_rate = together_rate[:, 1:]
            parent_child_contain_loss = self.parent_child_contain_loss(parent_center, parent_delta, box1_center,
                                                                       box1_offset, pos_rate)
            parent_child_contain_loss_prob = self.parent_child_contain_loss_prob(parent_center, parent_delta,
                                                                                 box1_center,
                                                                                 box1_offset, pos_rate)

            # child_parent_negative_loss = self.negative_contain_loss(box1_center.unsqueeze(1),
            #                                                         box1_offset.unsqueeze(1),
            #                                                         neg_parent_center,
            #                                                         neg_parent_delta, neg_rate)
            # child_parent_negative_loss_prob = self.negative_contain_loss_prob(box1_center.unsqueeze(1),
            #                                                                   box1_offset.unsqueeze(1),
            #                                                                   neg_parent_center, neg_parent_delta,
            #                                                                   neg_rate)

            loss = alpha * (parent_child_contain_loss) + extra * (
                parent_child_contain_loss_prob)
            return loss, parent_child_contain_loss, parent_child_contain_loss_prob, 0, 0
        elif mode == 'box+':
            Index_overlap, Index_disjoint, pos_predictions = self.get_cond_probs(box1_center.unsqueeze(1),
                                                                                 box1_offset.unsqueeze(1),
                                                                                 box2_center, box2_offset)
            input_condprob_step = torch.zeros_like(pos_predictions)
            input_condprob_step = input_condprob_step.view(chunk_num, -1)
            input_condprob_step[:, 0] = 1
            input_condprob_step = input_condprob_step.view(-1)

            neg_prediction = torch.ones_like(pos_predictions) - pos_predictions
            prediction = torch.stack([neg_prediction, pos_predictions], dim=-1)

            if Index_overlap.shape[0] > 0:
                loss_overlap = self.overlap_loss(prediction[Index_overlap], input_condprob_step[Index_overlap])
            else:
                loss_overlap = torch.zeros(1).to(level_centers[0].device)
            if Index_disjoint.shape[0] > 0:
                loss_disjoint = (input_condprob_step[Index_disjoint] * prediction[Index_disjoint][:, 1]).mean()
            else:
                loss_disjoint = torch.zeros(1).to(level_centers[0].device)

            loss = loss_overlap + loss_disjoint
            return loss, loss_overlap, loss_disjoint
        elif mode == 'IoU':
            score, post = self.IoU_intersection(box1_center.unsqueeze(1), box1_offset.unsqueeze(1), box2_center,
                                                box2_offset)
            A = box1_depth.unsqueeze(1) + box2_depth
            B = 2 * together_depth
            d_margin = d_alpha * (B / A)
            d_margin = 1 - d_margin

            pos_score = score[:, 0]
            neg_score = score[:, 1:]
            pos_post = post[:, 0]
            neg_post = post[:, 1:]
            pos_margin = d_margin[:, 0]
            neg_margin = d_margin[:, 1:]
            pos_rate = together_rate[:, 0]
            neg_rate = together_rate[:, 1:]
            if IoU_mode == 'margin':
                # test 2
                # pos_score = pos_score - pos_post
                # neg_score = neg_score - neg_post
                # neg_score = neg_score.mean(dim=-1)
                # loss = torch.relu(IoU_margin + neg_score - pos_score)
                # loss = loss.mean()
                # test 1
                pos_loss = 1 - pos_score + pos_post
                neg_loss = neg_score + torch.relu(IoU_margin - neg_post)
                # test 3
                # pos_loss = pos_margin * pos_loss
                # neg_loss = neg_margin * neg_loss
                # test 4
                # pos_loss = pos_rate * pos_loss
                # neg_loss = neg_rate * neg_loss
                loss = pos_loss.mean() + neg_loss.mean()
            elif IoU_mode == 'BPR':
                pos_score = pos_score.unsqueeze(1)
                step1 = pos_score - neg_score
                step2 = torch.nn.functional.logsigmoid(step1)
                loss = -1 * torch.mean(step2)
            return loss


class Type_box_intersection(nn.Module):
    def __init__(self, margin=0.01, eps=-0.01):
        super(Type_box_intersection, self).__init__()
        self.margin = margin
        self.eps = eps

        self.par_chd_left_loss = nn.MSELoss()
        self.par_chd_right_loss = nn.MSELoss()
        self.par_chd_negative_loss = nn.MSELoss()
        self.positive_prob_loss = nn.MSELoss()
        self.negative_prob_loss = nn.MSELoss()

        self.gradnorm = GradNorm.apply
        self.overlap_loss = torch.nn.CrossEntropyLoss()

        self.std_vol = -2
        self.con_loss = torch.nn.CrossEntropyLoss()

    def get_cond_probs(self, boxes1_center, boxes1_offset, boxes2_center, boxes2_offset):
        left, right = self.box_intersection(boxes1_center, boxes1_offset, boxes2_center, boxes2_offset)
        Index_overlap, Index_disjoint, log_inter_measure = self.volumes(left, right)
        left = boxes1_center - boxes1_offset
        right = boxes1_center + boxes1_offset
        _, _, log_box1 = self.volumes(left, right)
        log_box1 = torch.repeat_interleave(log_box1, repeats=int(len(log_inter_measure) / len(log_box1)), dim=0)
        measures = torch.zeros_like(log_inter_measure)
        if Index_overlap.shape[0] > 0:
            measures[Index_overlap] = torch.exp(log_inter_measure[Index_overlap] - log_box1[Index_overlap])
        if Index_disjoint.shape[0] > 0:
            measures[Index_disjoint] = log_inter_measure[Index_disjoint]

        return Index_overlap, Index_disjoint, measures

    def volumes(self, left, right):
        eps = torch.finfo(left.dtype).tiny  # type: ignore

        # overlap mark
        boxLens = right - left

        Tag = (torch.min(boxLens, dim=-1).values > 0.0)
        tmp_tag = Tag.view(-1)
        Index_overlap = torch.where(tmp_tag == True)[0]
        Index_disjoint = torch.where(tmp_tag == False)[0]
        pnorm = 2
        measures = torch.zeros_like(tmp_tag, dtype=torch.float)

        boxLens = boxLens.view(boxLens.shape[0] * boxLens.shape[1], -1)
        zeros = torch.zeros_like(boxLens)
        if Index_overlap.shape[0] > 0:
            measures[Index_overlap] = self.gradnorm(
                torch.max(zeros[Index_overlap], boxLens[Index_overlap]).clamp_min(eps))
        if Index_disjoint.shape[0] > 0:
            measures[Index_disjoint] = torch.norm(torch.max(zeros[Index_disjoint], -boxLens[Index_disjoint]), p=pnorm,
                                                  dim=-1)

        return Index_overlap, Index_disjoint, measures

    def box_volumn(self, delta):
        flag = torch.sum(delta <= 0, -1)
        product = torch.prod(delta, -1)
        zeros = torch.zeros_like(product)
        ones = torch.ones_like(product)
        mask = torch.where(flag == 0, ones, zeros)
        volumn = torch.mul(product, mask)

        return volumn

    def condition_score(self, child_center, child_delta, parent_center, parent_delta):
        inter_left, inter_right = self.box_intersection(child_center, child_delta, parent_center, parent_delta)
        inter_delta = (inter_right - inter_left) / 2
        flag = (inter_delta <= 0)
        zeros = torch.zeros_like(flag)
        ones = torch.ones_like(flag)
        mask = torch.where(flag == False, ones, zeros)
        masked_inter_delta = torch.mul(inter_delta, mask)

        child_mask = (child_delta > 1e-6)
        zeros = torch.zeros_like(child_delta)
        score_pre = torch.where(child_mask, torch.div(masked_inter_delta, child_delta + 1e-6), zeros)
        score = torch.prod(score_pre, -1)

        # parent_volumn = self.box_volumn(parent_delta)

        return score.squeeze(), None

    def parent_child_contain_loss(self, parent_center, parent_delta, child_center, child_delta):
        parent_left = parent_center - parent_delta
        parent_right = parent_center + parent_delta

        child_left = child_center - child_delta
        child_right = child_center + child_delta

        diff_left = child_left - parent_left
        zeros = torch.zeros_like(diff_left)
        ones = torch.ones_like(diff_left)
        margins = torch.ones_like(diff_left) * self.margin
        left_mask = torch.where(diff_left < self.margin, ones, zeros)
        left_loss = self.par_chd_left_loss(torch.mul(diff_left, left_mask), torch.mul(margins, left_mask))
        # left_loss = left_loss.mean(dim=-1) * rate

        diff_right = parent_right - child_right
        zeros = torch.zeros_like(diff_right)
        ones = torch.ones_like(diff_right)
        margins = torch.ones_like(diff_right) * self.margin
        right_mask = torch.where(diff_right < self.margin, ones, zeros)
        right_loss = self.par_chd_right_loss(torch.mul(diff_right, right_mask), torch.mul(margins, right_mask))
        # right_loss = right_loss.mean(dim=-1) * rate

        return (left_loss + right_loss) / 2

    def parent_child_contain_loss_prob(self, parent_center, parent_delta, child_center, child_delta):
        score, _ = self.condition_score(child_center, child_delta, parent_center, parent_delta)
        ones = torch.ones_like(score)
        loss = self.positive_prob_loss(score, ones)
        # loss = loss * rate
        return loss

    def box_intersection(self, center1, delta1, center2, delta2):
        left1 = center1 - delta1
        right1 = center1 + delta1
        left2 = center2 - delta2
        right2 = center2 + delta2
        inter_left = torch.max(left1, left2)
        inter_right = torch.min(right1, right2)

        return inter_left, inter_right

    def negative_contain_loss(self, child_center, child_delta, neg_parent_center, neg_parent_delta):
        inter_left, inter_right = self.box_intersection(child_center, child_delta, neg_parent_center, neg_parent_delta)

        inter_delta = (inter_right - inter_left) / 2
        zeros = torch.zeros_like(inter_delta)
        ones = torch.ones_like(inter_delta)
        epsilon = torch.ones_like(inter_delta) * self.eps
        inter_mask = torch.where(inter_delta > self.eps, ones, zeros)
        inter_loss = self.par_chd_negative_loss(torch.mul(inter_delta, inter_mask), torch.mul(epsilon, inter_mask))
        # inter_loss = inter_loss.mean(dim=-1) * rate

        return inter_loss

    def negative_contain_loss_prob(self, child_center, child_delta, neg_parent_center, neg_parent_delta):
        score, _ = self.condition_score(child_center, child_delta, neg_parent_center, neg_parent_delta)
        zeros = torch.zeros_like(score)
        loss = self.negative_prob_loss(score, zeros)
        # loss = loss * rate
        return loss

    def IoU_intersection(self, center1, delta1, center2, delta2):
        score, _ = self.condition_score(center1, delta1, center2, delta2)
        dist_center = (center1 - center2).abs()
        # test 1
        left1 = center1 - delta1
        right1 = center1 + delta1
        left2 = center2 - delta2
        right2 = center2 + delta2
        inter_left = torch.min(left1, left2)
        inter_right = torch.max(right1, right2)
        dist_C = (inter_right - inter_left).abs()
        post = torch.norm(dist_center, p=2, dim=-1) / torch.norm(dist_C, p=2, dim=-1)
        post = post ** 2
        # test 2
        # delta = (dist_center - delta1 - delta2)/delta1
        # delta_temp = delta / 0.07
        # post = torch.sigmoid(delta_temp).mean(dim=-1)
        return score, post

    def log_inter_volumes(self, inter_min, inter_max, scale=1.):
        eps = 1e-16
        if isinstance(scale, float):
            s = torch.tensor(scale)
        else:
            s = scale
        log_vol = torch.sum(
            torch.log(
                F.softplus(inter_max - inter_min, beta=0.7).clamp_min(eps)
            ),
            dim=-1
        ) + torch.log(s)
        return log_vol

    def diou_loss(self, box1_embs, box1_off, box2_embs, box2_off, box_type='normal'):
        box1_min = (box1_embs - (box1_off))
        box1_max = (box1_embs + (box1_off))
        box2_min = (box2_embs - (box2_off))
        box2_max = (box2_embs + (box2_off))
        if box_type == 'normal':
            inter_min = torch.max(box1_min, box2_min)
            inter_max = torch.min(box1_max, box2_max)
            cen_dis = torch.norm(box1_embs - box2_embs, p=2, dim=-1)
            outer_min = torch.min(box1_min, box2_min)
            outer_max = torch.max(box1_max, box2_max)
            inter_vol = self.log_inter_volumes(inter_min, inter_max)
        else:
            raise ValueError(f'{box_type} is not support')
        c2 = torch.norm(outer_max - outer_min, p=2, dim=-1)
        d_loss = torch.sqrt(cen_dis) / torch.sqrt(c2)
        logit = torch.sigmoid(inter_vol + self.std_vol) - d_loss
        return logit

    def forward(self, centers, offsets, n_neg, neg_list, gumbel_beta=0.01, tanh=0, alpha=1.0, extra=0.1, mode='B4T',
                IoU_mode='margin', IoU_margin=2, d_alpha=1.0):
        box1_centers = []
        box1_offsets = []
        box2_centers = []
        box2_offsets = []

        box1_depth = []
        box2_depth = []
        together_depth = []
        for k, v in neg_list.items():
            level = v['level']
            box1_centers.append(centers[k])
            box1_offsets.append(offsets[k])
            box2_centers.append(centers[v['parent']])
            box2_offsets.append(offsets[v['parent']])

            box1_depth.append(level + 1)
            box2_depth.append(level)
            together_depth.append(level)

            order = ['bro', 'uncle', 'cousin']
            rest_neg = n_neg
            for i in order:
                num = len(v[i])
                if num == 0:
                    continue
                if num >= rest_neg:
                    select_num = rest_neg
                else:
                    select_num = num
                rest_neg -= select_num
                select_idxes = np.random.choice(v[i], size=select_num, replace=False)
                box2_centers.extend(centers[select_idxes])
                box2_offsets.extend(offsets[select_idxes])

                if i == 1:
                    box2_depth.extend([level for _ in select_idxes])
                else:
                    box2_depth.extend([level + 1 for _ in select_idxes])
                if i == 0:
                    together_depth.extend([level for _ in select_idxes])
                else:
                    together_depth.extend([level - 1 for _ in select_idxes])
                if rest_neg == 0:
                    break

            if rest_neg > 0:
                for i in range(len(v['rest'])):
                    num = len(v['rest'][i])
                    if num == 0:
                        continue
                    if num >= rest_neg:
                        select_num = rest_neg
                    else:
                        select_num = num
                    rest_neg -= select_num
                    select_idxes = np.random.choice(v['rest'][i], size=select_num, replace=False)
                    box2_centers.extend(centers[select_idxes])
                    box2_offsets.extend(offsets[select_idxes])

                    box2_depth.extend([i + 2 for _ in select_idxes])
                    together_depth.extend([1 for _ in select_idxes])
                    if rest_neg == 0:
                        break
                if rest_neg > 0:
                    select_idxes = np.random.choice(list(range(n_neg - rest_neg)),
                                                    size=rest_neg, replace=True)
                    select_idxes = - select_idxes - 1
                    for ad, idx in enumerate(select_idxes):
                        box2_centers.append(box2_centers[idx - ad])
                        box2_offsets.append(box2_offsets[idx - ad])
                        box2_depth.append(box2_depth[idx - ad])
                        together_depth.append(together_depth[idx - ad])

        box1_centers = torch.stack(box1_centers)
        box1_offsets = torch.stack(box1_offsets)
        box2_centers = torch.stack(box2_centers)
        box2_offsets = torch.stack(box2_offsets)

        box1_depth = torch.tensor(box1_depth, dtype=torch.int32).to(centers.device)
        box2_depth = torch.tensor(box2_depth, dtype=torch.int32).to(centers.device)
        together_depth = torch.tensor(together_depth, dtype=torch.int32).to(centers.device)
        assert len(box2_centers) % (1 + n_neg) == 0, 'box2 cant div by (1+n_neg)'
        # box1_centers_exp = box1_centers.unsqueeze(1)
        # box1_offsets_exp = box1_offsets.unsqueeze(1)
        box2_centers = box2_centers.view(len(neg_list), (1 + n_neg), -1)
        box2_offsets = box2_offsets.view(len(neg_list), (1 + n_neg), -1)
        box2_depth.view(len(neg_list), -1)
        together_depth.view(len(neg_list), -1)

        if mode == 'B4T':
            box_inter_vol = cal_logit_box(box1_centers.unsqueeze(1), box1_offsets.unsqueeze(1), box2_centers,
                                          box2_offsets,
                                          gumbel_beta, tanh)
            if tanh == 0:
                x_max = box1_centers + box1_offsets
                x_min = box1_centers - box1_offsets
            else:
                x_max = torch.tanh(box1_centers + box1_offsets)
                x_min = torch.tanh(box1_centers - box1_offsets)
            euler_gamma = 0.57721566490153286060
            x_vol = log_soft_volume(x_max, x_min, euler_gamma, gumbel_beta)
            probs = box_inter_vol - x_vol.unsqueeze(1)
            targets = torch.zeros_like(probs)
            targets[:, 0] = 1
            return probs, targets
        elif mode == 'taxo':
            parent_center = box2_centers[:, 0]
            parent_delta = box2_offsets[:, 0]
            neg_parent_center = box2_centers[:, 1:]
            neg_parent_delta = box2_offsets[:, 1:]
            parent_child_contain_loss = self.parent_child_contain_loss(parent_center, parent_delta, box1_centers,
                                                                       box1_offsets)
            parent_child_contain_loss_prob = self.parent_child_contain_loss_prob(parent_center, parent_delta,
                                                                                 box1_centers,
                                                                                 box1_offsets)

            child_parent_negative_loss = self.negative_contain_loss(box1_centers.unsqueeze(1),
                                                                    box1_offsets.unsqueeze(1),
                                                                    neg_parent_center,
                                                                    neg_parent_delta)
            child_parent_negative_loss_prob = self.negative_contain_loss_prob(box1_centers.unsqueeze(1),
                                                                              box1_offsets.unsqueeze(1),
                                                                              neg_parent_center, neg_parent_delta)

            loss = alpha * (parent_child_contain_loss + child_parent_negative_loss) + extra * (
                    parent_child_contain_loss_prob + child_parent_negative_loss_prob)
            return loss, parent_child_contain_loss, parent_child_contain_loss_prob, child_parent_negative_loss, child_parent_negative_loss_prob
        elif mode == 'box+':
            Index_overlap, Index_disjoint, pos_predictions = self.get_cond_probs(box1_centers.unsqueeze(1),
                                                                                 box1_offsets.unsqueeze(1),
                                                                                 box2_centers, box2_offsets)
            input_condprob_step = torch.zeros_like(pos_predictions)
            input_condprob_step = input_condprob_step.view(len(neg_list), -1)
            input_condprob_step[:, 0] = 1
            input_condprob_step = input_condprob_step.view(-1)

            neg_prediction = torch.ones_like(pos_predictions) - pos_predictions
            prediction = torch.stack([neg_prediction, pos_predictions], dim=-1)

            if Index_overlap.shape[0] > 0:
                loss_overlap = self.overlap_loss(prediction[Index_overlap], input_condprob_step[Index_overlap])
            else:
                loss_overlap = torch.zeros(1).to(centers.device)
            if Index_disjoint.shape[0] > 0:
                loss_disjoint = (input_condprob_step[Index_disjoint] * prediction[Index_disjoint][:, 1]).mean()
            else:
                loss_disjoint = torch.zeros(1).to(centers.device)

            loss = loss_overlap + loss_disjoint
            return loss, loss_overlap, loss_disjoint
        elif mode == 'IoU':
            score, post = self.IoU_intersection(box1_centers.unsqueeze(1), box1_offsets.unsqueeze(1), box2_centers,
                                                box2_offsets)
            A = box1_depth.unsqueeze(1) + box2_depth
            B = 2 * together_depth
            d_margin = d_alpha * (B / A)
            d_margin = 1 - d_margin

            pos_score = score[:, 0]
            neg_score = score[:, 1:]
            pos_post = post[:, 0]
            neg_post = post[:, 1:]
            pos_margin = d_margin[:, 0]
            neg_margin = d_margin[:, 1:]
            # pos_rate = together_rate[:, 0]
            # neg_rate = together_rate[:, 1:]
            if IoU_mode == 'margin':
                # test 2
                # pos_score = pos_score - pos_post
                # neg_score = neg_score - neg_post
                # neg_score = neg_score.mean(dim=-1)
                # loss = torch.relu(IoU_margin + neg_score - pos_score)
                # loss = loss.mean()
                # test 1
                pos_loss = 1 - pos_score + pos_post
                neg_loss = neg_score + torch.relu(IoU_margin - neg_post)
                # test 3
                # pos_loss = pos_margin * pos_loss
                # neg_loss = neg_margin * neg_loss
                # test 4
                # pos_loss = pos_rate * pos_loss
                # neg_loss = neg_rate * neg_loss
                loss = pos_loss.mean() + neg_loss.mean()
            elif IoU_mode == 'BPR':
                pos_score = pos_score.unsqueeze(1)
                step1 = pos_score - neg_score
                step2 = torch.nn.functional.logsigmoid(step1)
                loss = -1 * torch.mean(step2)
            return loss
        elif mode == 'cbox':
            logits = self.diou_loss(box1_centers.unsqueeze(1), box1_offsets.unsqueeze(1), box2_centers, box2_offsets)
            label = torch.zeros(len(logits)).to(centers.device).long()
            loss = self.con_loss(logits, label)
            return loss


class Cross_box_intersection(nn.Module):
    def __init__(self, margin=0.01, eps=-0.01, match_weight=0):
        super(Cross_box_intersection, self).__init__()
        self.margin = margin
        self.eps = eps

        if match_weight == 0:
            self.par_chd_left_loss = nn.MSELoss()
            self.par_chd_right_loss = nn.MSELoss()
            self.par_chd_negative_loss = nn.MSELoss()
            self.positive_prob_loss = nn.MSELoss()
            self.negative_prob_loss = nn.MSELoss()
        else:
            self.par_chd_left_loss = nn.MSELoss(reduction='none')
            self.par_chd_right_loss = nn.MSELoss(reduction='none')
            self.par_chd_negative_loss = nn.MSELoss(reduction='none')
            self.positive_prob_loss = nn.MSELoss(reduction='none')
            self.negative_prob_loss = nn.MSELoss(reduction='none')

        self.gradnorm = GradNorm.apply
        self.overlap_loss = torch.nn.CrossEntropyLoss()

        self.std_vol = -2
        self.con_loss = torch.nn.CrossEntropyLoss()

    def get_cond_probs(self, boxes1_center, boxes1_offset, boxes2_center, boxes2_offset):
        left, right = self.box_intersection(boxes1_center, boxes1_offset, boxes2_center, boxes2_offset)
        Index_overlap, Index_disjoint, log_inter_measure = self.volumes(left, right)
        left = boxes1_center - boxes1_offset
        right = boxes1_center + boxes1_offset
        _, _, log_box1 = self.volumes(left, right)
        # log_box1 = torch.repeat_interleave(log_box1, repeats=int(len(log_inter_measure) / len(log_box1)), dim=0)
        measures = torch.zeros_like(log_inter_measure)
        if Index_overlap.shape[0] > 0:
            measures[Index_overlap] = torch.exp(log_inter_measure[Index_overlap] - log_box1[Index_overlap])
        if Index_disjoint.shape[0] > 0:
            measures[Index_disjoint] = log_inter_measure[Index_disjoint]

        return Index_overlap, Index_disjoint, measures

    def volumes(self, left, right):
        eps = torch.finfo(left.dtype).tiny  # type: ignore

        # overlap mark
        boxLens = right - left

        Tag = (torch.min(boxLens, dim=-1).values > 0.0)
        tmp_tag = Tag.view(-1)
        Index_overlap = torch.where(tmp_tag == True)[0]
        Index_disjoint = torch.where(tmp_tag == False)[0]
        pnorm = 1
        measures = torch.zeros_like(tmp_tag, dtype=torch.float)

        # boxLens = boxLens.view(boxLens.shape[0] * boxLens.shape[1], -1)
        zeros = torch.zeros_like(boxLens)
        if Index_overlap.shape[0] > 0:
            measures[Index_overlap] = self.gradnorm(
                torch.max(zeros[Index_overlap], boxLens[Index_overlap]).clamp_min(eps))
        if Index_disjoint.shape[0] > 0:
            measures[Index_disjoint] = torch.norm(torch.max(zeros[Index_disjoint], -boxLens[Index_disjoint]), p=pnorm,
                                                  dim=-1)

        return Index_overlap, Index_disjoint, measures

    def box_volumn(self, delta):
        flag = torch.sum(delta <= 0, -1)
        product = torch.prod(delta, -1)
        zeros = torch.zeros_like(product)
        ones = torch.ones_like(product)
        mask = torch.where(flag == 0, ones, zeros)
        volumn = torch.mul(product, mask)

        return volumn

    def condition_score(self, child_center, child_delta, parent_center, parent_delta):
        inter_left, inter_right = self.box_intersection(child_center, child_delta, parent_center, parent_delta)
        inter_delta = (inter_right - inter_left) / 2
        flag = (inter_delta <= 0)
        zeros = torch.zeros_like(flag)
        ones = torch.ones_like(flag)
        mask = torch.where(flag == False, ones, zeros)
        masked_inter_delta = torch.mul(inter_delta, mask)

        score_pre = torch.div(masked_inter_delta, child_delta + 1e-6)
        score = torch.prod(score_pre, -1)

        # parent_volumn = self.box_volumn(parent_delta)

        return score.squeeze(), None

    def parent_child_contain_loss(self, parent_center, parent_delta, child_center, child_delta, weight):
        parent_left = (parent_center - parent_delta)
        parent_right = (parent_center + parent_delta)

        child_left = (child_center - child_delta)
        child_right = (child_center + child_delta)

        diff_left = child_left - parent_left
        zeros = torch.zeros_like(diff_left)
        ones = torch.ones_like(diff_left)
        margins = torch.ones_like(diff_left) * self.margin
        left_mask = torch.where(diff_left < self.margin, ones, zeros)
        left_loss = self.par_chd_left_loss(torch.mul(diff_left, left_mask), torch.mul(margins, left_mask))
        if weight is not None:
            left_loss = left_loss.mean(dim=-1) * weight

        diff_right = parent_right - child_right
        zeros = torch.zeros_like(diff_right)
        ones = torch.ones_like(diff_right)
        margins = torch.ones_like(diff_right) * self.margin
        right_mask = torch.where(diff_right < self.margin, ones, zeros)
        right_loss = self.par_chd_right_loss(torch.mul(diff_right, right_mask), torch.mul(margins, right_mask))
        if weight is not None:
            right_loss = right_loss.mean(dim=-1) * weight

        if weight is not None:
            return (left_loss.mean() + right_loss.mean()) / 2
        else:
            return (left_loss + right_loss) / 2

    def parent_child_contain_loss_prob(self, parent_center, parent_delta, child_center, child_delta, weight):
        score, _ = self.condition_score(child_center, child_delta, parent_center, parent_delta)
        ones = torch.ones_like(score)
        loss = self.positive_prob_loss(score, ones)
        if weight is not None:
            loss = loss * weight
            return loss.mean()

        return loss

    def box_intersection(self, center1, delta1, center2, delta2):
        left1 = (center1 - delta1)
        right1 = (center1 + delta1)
        left2 = (center2 - delta2)
        right2 = (center2 + delta2)
        inter_left = torch.max(left1, left2)
        inter_right = torch.min(right1, right2)

        return inter_left, inter_right

    def negative_contain_loss(self, child_center, child_delta, neg_parent_center, neg_parent_delta, weight):
        inter_left, inter_right = self.box_intersection(child_center, child_delta, neg_parent_center, neg_parent_delta)

        inter_delta = (inter_right - inter_left) / 2
        zeros = torch.zeros_like(inter_delta)
        ones = torch.ones_like(inter_delta)
        epsilon = torch.ones_like(inter_delta) * self.eps
        inter_mask = torch.where(inter_delta > self.eps, ones, zeros)
        inter_loss = self.par_chd_negative_loss(torch.mul(inter_delta, inter_mask), torch.mul(epsilon, inter_mask))
        if weight is not None:
            inter_loss = inter_loss.mean(dim=-1) * weight.unsqueeze(1)
            return inter_loss.mean()

        return inter_loss

    def negative_contain_loss_prob(self, child_center, child_delta, neg_parent_center, neg_parent_delta, weight):
        score, _ = self.condition_score(child_center, child_delta, neg_parent_center, neg_parent_delta)
        zeros = torch.zeros_like(score)
        loss = self.negative_prob_loss(score, zeros)
        if weight is not None:
            loss = loss * weight.unsqueeze(1)
            return loss.mean()
        return loss

    # def IoU_intersection(self, center1, delta1, center2, delta2):
    #     inter_left, inter_right = self.box_intersection(center1, delta1, center2, delta2)
    #     inter_delta = (inter_right - inter_left) / 2
    #     vol_A = self.box_volumn(delta1)
    #     vol_B = self.box_volumn(delta2)
    #     vol_inter = self.box_volumn(inter_delta)
    #     # mask = ((vol_A + vol_B - vol_inter) > 1e-6)
    #     bmask = (inter_delta > 1e-6)
    #     mask = bmask.all(dim=-1)
    #     zero = torch.zeros_like(vol_inter)
    #     score = torch.where(mask, vol_inter / (vol_A + vol_B - vol_inter), zero)
    #     # score = torch.log(score)
    #
    #     # score, _ = self.condition_score(center1, delta1, center2, delta2)
    #     dist_center = (center1 - center2).abs()
    #     # # test 1
    #     left1 = center1 - delta1
    #     right1 = center1 + delta1
    #     left2 = center2 - delta2
    #     right2 = center2 + delta2
    #     inter_left = torch.min(left1, left2)
    #     inter_right = torch.max(right1, right2)
    #     dist_C = (inter_right - inter_left).abs()
    #     post = torch.norm(dist_center, p=2, dim=-1) / torch.norm(dist_C, p=2, dim=-1)
    #     post = post ** 2
    #     # test 2
    #     # delta = (dist_center - delta1 - delta2)/delta1
    #     # delta_temp = delta / 0.07
    #     # post = torch.sigmoid(delta_temp).mean(dim=-1)
    #     return score, post

    def IoU_intersection(self, center1, delta1, center2, delta2):
        score, _ = self.condition_score(center1, delta1, center2, delta2)
        dist_center = (center1 - center2).abs()
        # test 1
        left1 = center1 - delta1
        right1 = center1 + delta1
        left2 = center2 - delta2
        right2 = center2 + delta2
        inter_left = torch.min(left1, left2)
        inter_right = torch.max(right1, right2)
        dist_C = (inter_right - inter_left).abs()
        post = torch.norm(dist_center, p=2, dim=-1) / torch.norm(dist_C, p=2, dim=-1)
        post = post ** 2
        # test 2
        # delta = (dist_center - delta1 - delta2)/delta1
        # delta_temp = delta / 0.07
        # post = torch.sigmoid(delta_temp).mean(dim=-1)
        return score, post

    def dist_based(self, center1, delta1, center2, delta2):
        d_cen = center1 - center2
        d_off = delta1 - delta2
        dist_BB = torch.norm(d_cen, p=2, dim=-1) + torch.norm(d_off, p=2, dim=-1)
        return dist_BB

    def log_inter_volumes(self, inter_min, inter_max, scale=1.):
        eps = 1e-16
        if isinstance(scale, float):
            s = torch.tensor(scale)
        else:
            s = scale
        log_vol = torch.sum(
            torch.log(
                F.softplus(inter_max - inter_min, beta=0.7).clamp_min(eps)
            ),
            dim=-1
        ) + torch.log(s)
        return log_vol

    def diou_loss(self, box1_embs, box1_off, box2_embs, box2_off, box_type='normal'):
        box1_min = (box1_embs - F.relu(box1_off))
        box1_max = (box1_embs + F.relu(box1_off))
        box2_min = (box2_embs - F.relu(box2_off))
        box2_max = (box2_embs + F.relu(box2_off))
        if box_type == 'normal':
            inter_min = torch.max(box1_min, box2_min)
            inter_max = torch.min(box1_max, box2_max)
            cen_dis = torch.norm(box1_embs - box2_embs, p=2, dim=-1)
            outer_min = torch.min(box1_min, box2_min)
            outer_max = torch.max(box1_max, box2_max)
            inter_vol = self.log_inter_volumes(inter_min, inter_max)
        else:
            raise ValueError(f'{box_type} is not support')
        c2 = torch.norm(outer_max - outer_min, p=2, dim=-1)
        d_loss = torch.sqrt(cen_dis) / torch.sqrt(c2)
        logit = torch.sigmoid(inter_vol + self.std_vol) - d_loss
        return logit

    def matchCF(self, level_centers, level_offsets, match_result, level_matrix, concept_type_path, ins_type_path):
        ins_box = {}
        idx_set = set()
        # conflict_dict = dict()
        c_i_set = dict()
        c_i_dict = dict()
        count = 0
        ins_centers, ins_offsets = [], []
        match_score = []
        for pair in match_result:
            c_path = concept_type_path[pair[0]]
            i_path = ins_type_path[pair[1]]
            score = level_matrix[pair[0]][pair[1]]
            level = len(c_path)
            for i in reversed(range(len(c_path))):
                # conflict_dict.setdefault(c_path[i],False)
                c_i_dict.setdefault(c_path[i], [])
                c_i_set.setdefault(c_path[i], set())
                if i_path[i] not in c_i_set[c_path[i]]:
                    c_i_set[c_path[i]].add(i_path[i])
                    c_i_dict[c_path[i]].append(count)
                    count += 1
                    cur_level = level - i - 1
                    ins_centers.append(level_centers[cur_level][i_path[i]])
                    ins_offsets.append(level_offsets[cur_level][i_path[i]])
                    match_score.append(score[i])
                # if c_path[i] not in idx_set:
                #     idx_set.add(c_path[i])
                #     cur_level = level - i - 1
                #     ins_box[c_path[i]] = (level_centers[cur_level][i_path[i]], level_offsets[cur_level][i_path[i]])
                # else:
                #     conflict_dict[c_path[i]] = True

        # for i in range(len(ins_box)):
        #     ins_centers.append(ins_box[i][0])
        #     ins_offsets.append(ins_box[i][1])

        ins_centers = torch.stack(ins_centers)
        ins_offsets = torch.stack(ins_offsets)
        return ins_centers, ins_offsets, c_i_dict, match_score

    def forward(self, centers, offsets, n_neg, neg_list, level_centers, level_offsets,
                match_result, level_matrix, concept_type_path, ins_type_path, gumbel_beta, tanh=0, alpha=1.0, extra=0.1,
                mode='B4T',
                IoU_mode='margin', IoU_margin=2, d_alpha=1.0, match_weight=0, onlyBottom=0, self_adv=0, adv_temp=0,
                dist_margin=0):
        ins_centers, ins_offsets, c_i_dict, match_score = self.matchCF(level_centers, level_offsets, match_result,
                                                                       level_matrix,
                                                                       concept_type_path,
                                                                       ins_type_path)
        box1_centers = []
        box1_offsets = []
        box2_centers = []
        box2_offsets = []
        box1_centers_add = []
        box1_offsets_add = []
        box2_centers_add = []
        box2_offsets_add = []

        # box1_depth = []
        # box2_depth = []
        # together_depth = []
        neg_chunk_num = 0
        box_score = []
        box_score_add = []
        for k, v in neg_list.items():
            level = v['level']
            if onlyBottom == 1 and level != len(level_centers):
                continue
            for c_i_idx in c_i_dict[k]:
                # box1_depth.append(level + 1)
                # box2_depth.append(level + 1)
                # together_depth.append(level)

                box1_centers.append(centers[k])
                box1_offsets.append(offsets[k])
                box2_centers.append(ins_centers[c_i_idx])
                box2_offsets.append(ins_offsets[c_i_idx])
                box_score.append(match_score[c_i_idx])
                if mode not in ['dist', 'cbox']:
                    box2_centers.append(centers[k])
                    box2_offsets.append(offsets[k])
                    box1_centers.append(ins_centers[c_i_idx])
                    box1_offsets.append(ins_offsets[c_i_idx])
                    box_score.append(match_score[c_i_idx])

            # for c_i_idx in c_i_dict[k]:
            #     box1_centers_add.append(ins_centers[c_i_idx])
            #     box1_offsets_add.append(ins_offsets[c_i_idx])
            # box2_centers_add.append(centers[k])
            # box2_offsets_add.append(offsets[k])
            for c_i_idx in c_i_dict[k]:
                neg_chunk_num += 1
                box1_centers_add.append(ins_centers[c_i_idx])
                box1_offsets_add.append(ins_centers[c_i_idx])
                box_score_add.append(match_score[c_i_idx])
            order = ['bro', 'uncle', 'cousin']
            rest_neg = n_neg
            for i in order:
                num = len(v[i])
                if num == 0:
                    continue
                if num >= rest_neg:
                    select_num = rest_neg
                else:
                    select_num = num
                rest_neg -= select_num
                select_idxes = np.random.choice(v[i], size=select_num, replace=False)
                # box2_centers.extend(ins_centers[select_idxes])
                # box2_offsets.extend(ins_offsets[select_idxes])
                box2_centers_add.extend(centers[select_idxes])
                box2_offsets_add.extend(offsets[select_idxes])
                if rest_neg == 0:
                    break

            if rest_neg > 0:
                for i in range(len(v['rest'])):
                    num = len(v['rest'][i])
                    if num == 0:
                        continue
                    if num >= rest_neg:
                        select_num = rest_neg
                    else:
                        select_num = num
                    rest_neg -= select_num
                    select_idxes = np.random.choice(v['rest'][i], size=select_num, replace=False)
                    # box2_centers.extend(ins_centers[select_idxes])
                    # box2_offsets.extend(ins_offsets[select_idxes])
                    box2_centers_add.extend(centers[select_idxes])
                    box2_offsets_add.extend(offsets[select_idxes])
                    if rest_neg == 0:
                        break
                if rest_neg > 0:
                    select_idxes = np.random.choice(list(range(n_neg - rest_neg)),
                                                    size=rest_neg, replace=True)
                    select_idxes = - select_idxes - 1
                    for ad, idx in enumerate(select_idxes):
                        # box2_centers.append(box2_centers[idx - ad])
                        # box2_offsets.append(box2_offsets[idx - ad])
                        box2_centers_add.append(box2_centers_add[idx - ad])
                        box2_offsets_add.append(box2_offsets_add[idx - ad])
            for i in range(len(c_i_dict[k]) - 1):
                box2_centers_add.extend(box2_centers_add[-n_neg:])
                box2_offsets_add.extend(box2_centers_add[-n_neg:])

        # box1_centers.extend(box1_centers_add)
        # box1_offsets.extend(box1_offsets_add)
        # box2_centers.extend(box2_centers_add)
        # box2_offsets.extend(box2_offsets_add)

        box1_centers = torch.stack(box1_centers)
        box1_offsets = torch.stack(box1_offsets)
        box2_centers = torch.stack(box2_centers)
        box2_offsets = torch.stack(box2_offsets)

        box1_centers_add = torch.stack(box1_centers_add)
        box1_offsets_add = torch.stack(box1_offsets_add)
        box2_centers_add = torch.stack(box2_centers_add)
        box2_offsets_add = torch.stack(box2_offsets_add)

        box_score = torch.FloatTensor(box_score).to(centers.device)
        box_score_add = torch.FloatTensor(box_score_add).to(centers.device)
        assert len(box2_centers_add) % (n_neg) == 0, 'box2 cant div by (n_neg)'
        # box1_centers_exp = box1_centers.unsqueeze(1)
        # box1_offsets_exp = box1_offsets.unsqueeze(1)
        box2_centers_add = box2_centers_add.view(neg_chunk_num, n_neg, -1)
        box2_offsets_add = box2_offsets_add.view(neg_chunk_num, n_neg, -1)

        if mode == 'B4T':
            box_inter_vol = cal_logit_box(box1_centers.unsqueeze(1), box1_offsets.unsqueeze(1), box2_centers,
                                          box2_offsets,
                                          gumbel_beta, tanh)
            if tanh == 0:
                x_max = box1_centers + box1_offsets
                x_min = box1_centers - box1_offsets
            else:
                x_max = torch.tanh(box1_centers + box1_offsets)
                x_min = torch.tanh(box1_centers - box1_offsets)
            euler_gamma = 0.57721566490153286060
            x_vol = log_soft_volume(x_max, x_min, euler_gamma, gumbel_beta)
            probs = box_inter_vol - x_vol.unsqueeze(1)
            targets = torch.zeros_like(probs)
            targets[:, 0] = 1
            return probs, targets
        elif mode == 'taxo':

            # parent_center = box2_centers[:, 0]
            # parent_delta = box2_offsets[:, 0]
            neg_parent_center = box2_centers_add
            neg_parent_delta = box2_offsets_add
            # print(box1_offsets)
            if match_weight == 1:
                pos_weight = box_score
                neg_weight = box_score_add
            else:
                pos_weight = None
                neg_weight = None
            parent_child_contain_loss = self.parent_child_contain_loss(box2_centers, box2_offsets, box1_centers,
                                                                       box1_offsets, pos_weight)
            parent_child_contain_loss_prob = self.parent_child_contain_loss_prob(box2_centers, box2_offsets,
                                                                                 box1_centers,
                                                                                 box1_offsets, pos_weight)

            child_parent_negative_loss = self.negative_contain_loss(box1_centers_add.unsqueeze(1),
                                                                    box1_offsets_add.unsqueeze(1),
                                                                    neg_parent_center,
                                                                    neg_parent_delta, neg_weight)
            child_parent_negative_loss_prob = self.negative_contain_loss_prob(box1_centers_add.unsqueeze(1),
                                                                              box1_offsets_add.unsqueeze(1),
                                                                              neg_parent_center, neg_parent_delta,
                                                                              neg_weight)

            loss = alpha * (parent_child_contain_loss + child_parent_negative_loss) + extra * (
                    parent_child_contain_loss_prob + child_parent_negative_loss_prob)
            return loss, parent_child_contain_loss, parent_child_contain_loss_prob, child_parent_negative_loss, child_parent_negative_loss_prob
        elif mode == 'box+':
            Index_overlap, Index_disjoint, pos_predictions = self.get_cond_probs(box1_centers, box1_offsets,
                                                                                 box2_centers, box2_offsets)
            input_condprob_step = torch.ones_like(pos_predictions)
            # input_condprob_step = input_condprob_step.view(len(neg_list), -1)
            # input_condprob_step[:, 0] = 1
            # input_condprob_step = input_condprob_step.view(-1)

            neg_prediction = torch.ones_like(pos_predictions) - pos_predictions
            prediction = torch.stack([neg_prediction, pos_predictions], dim=-1)

            if Index_overlap.shape[0] > 0:
                loss_overlap = self.overlap_loss(prediction[Index_overlap], input_condprob_step[Index_overlap])
            else:
                loss_overlap = torch.zeros(1).to(centers.device)
            if Index_disjoint.shape[0] > 0:
                loss_disjoint = (input_condprob_step[Index_disjoint] * prediction[Index_disjoint][:, 1]).mean()
            else:
                loss_disjoint = torch.zeros(1).to(centers.device)

            loss = loss_overlap + loss_disjoint
            return loss, loss_overlap, loss_disjoint
        elif mode == 'IoU':
            score, post = self.IoU_intersection(box1_centers, box1_offsets, box2_centers, box2_offsets)
            neg_score, neg_post = self.IoU_intersection(box1_centers_add.unsqueeze(1), box1_offsets_add.unsqueeze(1),
                                                        box2_centers_add, box2_offsets_add)
            # A = box1_depth + box2_depth
            # B = 2 * together_depth
            # d_margin = d_alpha * (B / A)
            # d_margin = 1 - d_margin

            # pos_score = score[:, 0]
            # neg_score = score[:, 1:]
            # pos_post = post[:, 0]
            # neg_post = post[:, 1:]
            # pos_margin = d_margin[:, 0]
            # neg_margin = d_margin[:, 1:]
            if IoU_mode == 'margin':
                # test 2
                # pos_score = pos_score - pos_post
                # neg_score = neg_score - neg_post
                # neg_score = neg_score.mean(dim=-1)
                # loss = torch.relu(IoU_margin + neg_score - pos_score)
                # loss = loss.mean()
                # test 1
                pos_loss = 1 - score + post
                neg_loss = neg_score + torch.relu(IoU_margin - neg_post)
                if match_weight == 1:
                    pos_loss = pos_loss * box_score
                    neg_loss = neg_loss * box_score_add.unsqueeze(1)
                # test 3
                # pos_loss = pos_margin * pos_loss
                # neg_loss = neg_margin * neg_loss
                # test 4
                # pos_loss = pos_rate * pos_loss
                # neg_loss = neg_rate * neg_loss
                loss = pos_loss.mean() + neg_loss.mean()
            elif IoU_mode == 'BPR':
                # 没改过，不要用
                pos_score = score
                step1 = pos_score
                step2 = torch.nn.functional.logsigmoid(step1)
                loss = -1 * torch.mean(step2)
            return loss
        elif mode == 'dist':
            pos_dist = self.dist_based(box1_centers, box1_offsets, box2_centers, box2_offsets)  # (B,,)
            neg_dist = self.dist_based(box1_centers_add.unsqueeze(1), box1_offsets_add.unsqueeze(1),
                                       box2_centers_add, box2_offsets_add)  # (B,K)
            pos_loss = pos_dist
            if self_adv == 1:
                pos_loss = pos_loss.unsqueeze(1)
                score = - neg_dist * adv_temp
                att = torch.softmax(score, dim=-1).detach()
                neg_loss = att * neg_dist
            else:
                neg_loss = neg_dist.mean(dim=-1)
            tot_loss = torch.relu(dist_margin + pos_loss - neg_loss)
            tot_loss = tot_loss.mean()
            return tot_loss
        elif mode == 'cbox':
            pos_logits = self.diou_loss(box1_centers, box1_offsets, box2_centers, box2_offsets)  # (B,,)
            neg_logits = self.diou_loss(box1_centers_add.unsqueeze(1), box1_offsets_add.unsqueeze(1),
                                        box2_centers_add, box2_offsets_add)  # (B,K)
            logits = torch.cat((pos_logits.unsqueeze(1), neg_logits), dim=-1)
            label = torch.zeros(len(logits)).to(centers.device).long()
            loss = self.con_loss(logits, label)
            return loss


def log_soft_volume(Z, z, euler_gamma, gumbel_beta):
    return torch.sum(
        torch.log(
            F.softplus(Z - z - 2 * euler_gamma * gumbel_beta, beta=1 / gumbel_beta) + 1e-23
        ),
        dim=-1,
    )


def cal_logit_box(box1_center, box1_offset, box2_center, box2_offset, gumbel_beta, tanh):
    # gumbel_beta = self.beta
    if tanh == 0:
        t1z, t1Z = box1_center - box1_offset, box1_center + box1_offset
        t2z, t2Z = box2_center - box2_offset, box2_center + box2_offset
    else:
        t1z, t1Z = torch.tanh(box1_center - box1_offset), torch.tanh(box1_center + box1_offset)
        t2z, t2Z = torch.tanh(box2_center - box2_offset), torch.tanh(box2_center + box2_offset)
    z = gumbel_beta * torch.logaddexp(
        t1z / gumbel_beta, t2z / gumbel_beta
    )
    z = torch.max(z, torch.max(t1z, t2z))
    # z =  torch.max(t1z, t2z)
    Z = -gumbel_beta * torch.logaddexp(
        -t1Z / gumbel_beta, -t2Z / gumbel_beta
    )
    Z = torch.min(Z, torch.min(t1Z, t2Z))
    # Z = torch.min(t1Z, t2Z)
    euler_gamma = 0.57721566490153286060
    return log_soft_volume(Z, z, euler_gamma, gumbel_beta)


def box_volume(centers, offsets, gumbel_beta=0.2):
    z, Z = centers - offsets, centers + offsets
    euler_gamma = 0.57721566490153286060
    return torch.sum(
        torch.log(
            F.softplus(Z - z - 2 * euler_gamma * gumbel_beta, beta=1 / gumbel_beta) + 1e-23
        ),
        dim=-1,
    )


def volume_reg_loss(centers, offsets, reg_margin=0.5):
    cluster_volume = torch.exp(box_volume(centers, offsets))
    return cluster_volume, torch.abs(reg_margin - cluster_volume).mean()
