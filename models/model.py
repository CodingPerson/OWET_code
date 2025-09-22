import math
from typing import Optional

import numpy as np
import torch.random
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

import constant


class Cluster2Box(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, mode):
        super(Cluster2Box, self).__init__()
        self.mode = mode
        self.dim = embedding_dim
        self.hidden_dim = hidden_dim
        if self.mode == 'R2B':  # Recourse2Box
            self.W_key = nn.Linear(embedding_dim, hidden_dim, bias=False)
            self.query = nn.Parameter(torch.randn(hidden_dim, 1))
            self.W_key2 = nn.Linear(embedding_dim, hidden_dim, bias=False)
            self.query2 = nn.Parameter(torch.randn((hidden_dim, 1)))
            self.W_center = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.W_offset = nn.Linear(hidden_dim, hidden_dim, bias=True)
            # nn.init.constant_(self.W_key.weight,1.0)
            # nn.init.constant_(self.W_key2.weight, 1.0)
            # nn.init.constant_(self.W_center.weight, 1.0)
            # nn.init.constant_(self.W_offset.weight, 1.0)
            # nn.init.constant_(self.W_offset.bias, 0.0)
        elif self.mode == 'CR_g':  # CubeRec_geometric
            self.wc = torch.nn.Linear(embedding_dim, hidden_dim, bias=False)
            self.wo = torch.nn.Linear(embedding_dim, hidden_dim, bias=False)
            # nn.init.constant_(self.wc.weight, 1.0)
            # nn.init.constant_(self.wo.weight, 1.0)
        elif self.mode == 'CR_a':  # CubeRec_attentive
            self.wc = torch.nn.Linear(embedding_dim, hidden_dim, bias=False)
            self.wo = torch.nn.Linear(embedding_dim, hidden_dim, bias=True)
            self.act_relu = torch.nn.ReLU(inplace=True)
            self.query = torch.nn.Linear(hidden_dim, 1, bias=False)
            self.key = torch.nn.Linear(embedding_dim, hidden_dim, bias=False)
            self.value = torch.nn.Linear(embedding_dim, hidden_dim, bias=False)
            # nn.init.constant_(self.wc.weight, 1.0)
            # nn.init.constant_(self.wo.weight, 1.0)
            # nn.init.constant_(self.wo.bias, 0.0)
            # nn.init.constant_(self.query.weight, 1.0)
            # nn.init.constant_(self.key.weight, 1.0)
            # nn.init.constant_(self.value.weight, 1.0)
        elif self.mode == 'R2B2':
            self.W_key = nn.Linear(embedding_dim, hidden_dim, bias=False)
            self.query = nn.Parameter(torch.randn(hidden_dim, 1))
            self.W_offset = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.act_relu = nn.ReLU(inplace=True)
            # self.act_relu = nn.ELU()
            # nn.init.normal_(self.W_key.weight,std=0.1)
            # nn.init.kaiming_normal_(self.W_offset.weight, mode='fan_in', nonlinearity='relu')
            # # nn.init.constant_(self.W_key.weight, 1.0)
            # # nn.init.constant_(self.W_offset.weight, 1.0)
            # nn.init.ones_(self.W_offset.bias)
        elif self.mode == 'R2B3':
            self.W_key = nn.Linear(embedding_dim, hidden_dim, bias=False)
            self.query = nn.Parameter(torch.randn(hidden_dim, 1))
            self.W_offset = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.act_relu = nn.ReLU(inplace=True)
            # self.act_relu = nn.ELU()
            # nn.init.normal_(self.W_key.weight,std=0.1)
            nn.init.kaiming_normal_(self.W_offset.weight, mode='fan_in', nonlinearity='relu')
            # # nn.init.constant_(self.W_key.weight, 1.0)
            # # nn.init.constant_(self.W_offset.weight, 1.0)
            # nn.init.ones_(self.W_offset.bias)

    def forward(self, x):
        if self.mode == 'R2B':
            key = self.W_key(x)
            query = self.query
            attention = torch.softmax(torch.mm(key, query) / (self.dim ** 0.5), dim=0)
            center = torch.sum(attention * x, dim=0)
            center = self.W_center(center)
            key2 = self.W_key2(x)
            query2 = self.query2
            attention2 = torch.softmax(torch.mm(key2, query2) / (self.dim ** 0.5), dim=0)
            offset = torch.sum(attention2 * x, dim=0)
            offset = torch.relu(self.W_offset(offset)) + 1e-6

            return center, offset
        elif self.mode == 'CR_g':
            u_max = torch.max(x, dim=0).values
            u_min = torch.min(x, dim=0).values
            # origin
            center = ((u_max + u_min) / 2)
            offset = (((u_max - u_min) / 2))
            # test 1
            # center = (u_max + u_min) / 2
            # offset = torch.relu((u_max - u_min) / 2) + 1e-6
            # test 2
            # center = x.mean(axis=0)
            # offset = torch.relu((u_max - u_min) / 2) + 1e-6
            # test 3
            # center = x.mean(axis=0)
            # v_max = torch.max((x - center), dim=0).values
            # v_min = torch.max((center - x), dim=0).values
            # offset = torch.where(v_max < v_min, v_max, v_min)
            # offset = torch.relu(offset) + 1e-6
            return center, offset
        elif self.mode == 'CR_a':
            y = self.key(x)
            y = self.query(y)
            key_user_query = F.softmax(y / (self.dim ** 0.5), dim=0)
            value_user = self.value(x)
            attn = torch.squeeze(torch.matmul(value_user.T, key_user_query))
            center = self.wc(attn)
            offset = self.act_relu(self.wo(attn)) + 1e-6
            return center, offset
        elif self.mode == 'R2B2':
            center = x.mean(axis=0)
            key = self.W_key(x)
            query = self.query
            # query = center.unsqueeze(1)
            attention = torch.softmax(torch.mm(key, query) / (self.dim ** 0.5), dim=0)
            offset = torch.sum(attention * x, dim=0)
            # offset = self.act_relu(self.W_offset(offset))
            offset = self.act_relu(self.W_offset(offset))
            return center, offset
        elif self.mode == 'R2B3':
            center = x.mean(axis=0)
            key = self.W_key(x)
            # query = self.query
            query = center.unsqueeze(1)
            attention = torch.softmax(torch.mm(key, query) / (self.dim ** 0.5), dim=0)
            offset = torch.sum(attention * x, dim=0)
            # offset = self.act_relu(self.W_offset(offset))
            offset = torch.abs(self.W_offset(offset))
            return center, offset


def cluster2box(feats, preds, model):
    assert len(feats) == len(preds), 'all feats len not equal preds len'
    k = max(preds)
    assert min(preds) == 1, 'cluster not begin at 1'
    centers = []
    offsets = []
    for i in range(1, k + 1):
        indexes = np.where(preds == i)[0]
        X = feats[indexes]
        center, offset = model(X)
        centers.append(center)
        offsets.append(offset)
    return torch.stack(centers), torch.stack(offsets)


class HighwayNetwork(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 n_layers: int = 2,
                 activation: Optional[nn.Module] = None):
        super(HighwayNetwork, self).__init__()
        self.n_layers = n_layers
        self.nonlinear = nn.ModuleList(
            [nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
        self.gate = nn.ModuleList(
            [nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
        for layer in self.gate:
            layer.bias = torch.nn.Parameter(0. * torch.ones_like(layer.bias))
        self.final_linear_layer = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU() if activation is None else activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer_idx in range(self.n_layers):
            gate_values = self.sigmoid(self.gate[layer_idx](inputs))
            nonlinear = self.activation(self.nonlinear[layer_idx](inputs))
            inputs = gate_values * nonlinear + (1. - gate_values) * inputs
        return self.final_linear_layer(inputs)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super(MLP, self).__init__()

        # self.fc1 = nn.Linear(input_dim, hidden, bias=True)
        # self.fc2 = nn.Linear(hidden, hidden, bias=True)
        self.fc3 = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        #x = self.fc1(x)
        # x = F.relu(x)
        x = self.fc3(x)

        return x


class Proj(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=768, n_layer=2, mode='B4T'):
        super(Proj, self).__init__()
        self.mode = mode
        if mode == 'B4T':
            self.proj = HighwayNetwork(input_dim, output_dim, n_layer)
        elif mode == 'mlp':
            self.projection_center = MLP(input_dim=768, hidden=hidden, output_dim=output_dim)
            self.projection_delta = MLP(input_dim=768, hidden=hidden, output_dim=output_dim)

    def forward(self, emb):
        if self.mode == 'B4T':
            proj_emb = self.proj(emb)
            centers, offsets = torch.chunk(proj_emb, chunks=2, dim=-1)
            offsets = torch.nn.functional.relu(offsets) + 1e-6
        elif self.mode == 'mlp':
            centers = (self.projection_center(emb))
            offsets = torch.relu(self.projection_delta(emb))+ 1e-6
        return centers, offsets


class Ins2TypeProj(nn.Module):
    def __init__(self, input_dim, output_dim, mode='B4T'):
        super(Ins2TypeProj, self).__init__()
        self.mode = mode
        self.wc = torch.nn.Linear(input_dim, output_dim, bias=True)
        self.wo = torch.nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, cen, off):
        center = (self.wc(cen))
        offset = torch.relu(self.wo(off))
        return center, offset


class TypeBox(nn.Module):
    def __init__(self, types_num, dim, init_center_span=0.01, init_offset_span=0.1):
        super(TypeBox, self).__init__()
        self.types_num = types_num
        self.dim = dim
        self.centers = nn.Parameter(torch.zeros(self.types_num, self.dim))
        self.offsets = nn.Parameter(torch.zeros(self.types_num, self.dim))
        self.box_embeddings = nn.Embedding(num_embeddings=types_num, embedding_dim=dim * 2)
        self.init_center_span = init_center_span
        self.init_offset_span = init_offset_span
        self.embedding_range = 0.5 / math.sqrt(self.dim)
        torch.nn.init.uniform_(self.centers, -self.embedding_range, self.embedding_range)
        self.shapes = nn.Parameter(torch.zeros(self.types_num, self.dim))
        nn.init.uniform_(tensor=self.shapes, a=-self.embedding_range, b=self.embedding_range)

        self.multiples = nn.Parameter(self.instantiate_box_embeddings(self.types_num))

        self.norm_shapes = nn.Parameter(self.product_normalise(self.shapes))
        self.offsets = nn.Parameter(torch.mul(self.multiples, self.norm_shapes))

    def forward(self, mode='B4T'):
        if mode == 'B4T':
            inputs = torch.arange(0,
                                  self.box_embeddings.num_embeddings,
                                  dtype=torch.int64,
                                  device=self.box_embeddings.weight.device)
            emb = self.box_embeddings(inputs)
            centers, offsets = torch.chunk(emb, 2, dim=-1)
            offsets = torch.nn.functional.relu(offsets) + 1e-6
            # centers = torch.nn.functional.sigmoid(centers)
            # offsets = torch.nn.functional.sigmoid(offsets)
            return centers, offsets
        elif mode == 'BoxE':
            return self.centers, torch.nn.functional.relu(self.offsets) + 1e-6

    def init_weights(self):
        torch.nn.init.uniform_(
            self.box_embeddings.weight[..., :self.dim],
            -self.init_center_span, self.init_center_span)
        torch.nn.init.uniform_(
            self.box_embeddings.weight[..., self.dim:],
            self.init_offset_span, self.init_offset_span)

    @staticmethod
    def instantiate_box_embeddings(ntype):
        scale_multiples = torch.empty([ntype, 1])
        nn.init.uniform_(scale_multiples, -1, 1)
        scale_multiples = torch.nn.functional.elu(scale_multiples) + 1.0
        return scale_multiples

    @staticmethod
    def product_normalise(input_tensor):
        NORM_LOG_BOUND = 1

        step1_tensor = torch.abs(input_tensor)
        step2_tensor = step1_tensor + 1e-8
        log_norm_tensor = torch.log(step2_tensor)
        minsize_tensor = torch.minimum(torch.min(log_norm_tensor, dim=1, keepdims=True)[0],
                                       torch.tensor(-NORM_LOG_BOUND))
        maxsize_tensor = torch.maximum(torch.max(log_norm_tensor, dim=1, keepdims=True)[0],
                                       torch.tensor(NORM_LOG_BOUND))
        minsize_ratio = -NORM_LOG_BOUND / minsize_tensor
        maxsize_ratio = NORM_LOG_BOUND / maxsize_tensor
        size_norm_ratio = torch.minimum(minsize_ratio, maxsize_ratio)
        normed_tensor = log_norm_tensor * size_norm_ratio

        return torch.exp(normed_tensor)


class OWETModel(nn.Module):
    def __init__(self, args):
        super(OWETModel, self).__init__()

        base_model = constant.base_model[args.model]
        tokenizer_ = base_model['tokenizer']
        modelPath = constant.model_type_path[args.model_type]
        self.model = BertForModel(modelPath, args)
        self.tokenizer = tokenizer_.from_pretrained(modelPath)
        self.boxEnable = False
        if args.single_box_model == 0:
            self.boxEnable = True
            self.Cluster2Box = Cluster2Box(args.type_box_dim, args.type_box_dim,
                                           mode=args.cluster2box)
        self.type_model = BertForModel(modelPath, args)
        self.gen_box_method = args.gen_box_method
        self.type_box_dim = args.type_box_dim
        if self.gen_box_method == 'B4T':
            self.proj = Proj(self.type_model.config.hidden_size, self.type_box_dim * 2,
                             n_layer=args.n_layer, mode=self.gen_box_method)
        elif self.gen_box_method == 'mlp':
            self.proj = Proj(self.type_model.config.hidden_size, self.type_box_dim,
                             hidden=(self.type_model.config.hidden_size + self.type_box_dim) // 2,
                             mode=self.gen_box_method)

        self.ins2type = Ins2TypeProj(self.type_box_dim, self.type_box_dim)
        self.type2ins = Ins2TypeProj(self.type_box_dim, self.type_box_dim)

        self.use_type_emb = args.use_type_emb
        self.type_embeding = TypeBox(args.num_known_class, args.type_box_dim, init_center_span=args.init_cen_span, init_offset_span=args.init_off_span)
        if args.init_type==1 and args.type_emb_method == 'B4T':
            self.type_embeding.init_weights()
    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None, mode=None, generator=None):
        if mode == 'type':
            if self.use_type_emb == 0:
                emb = self.type_model(input_ids, token_type_ids, attention_mask, labels, mode, generator)
                return self.proj(emb)
            else:
                return self.type_embeding()
        else:
            return self.model(input_ids, token_type_ids, attention_mask, labels, mode, generator)

    def ins2type_proj(self, level_cen, level_off):
        tmp_centers = []
        tmp_offsets = []
        for i in range(len(level_cen)):
            cen, off = level_cen[i], level_off[i]
            cen, off = self.ins2type(cen, off)
            tmp_centers.append(cen)
            tmp_offsets.append(off)
        return tmp_centers, tmp_offsets

    def type2ins_proj(self, cen, off):
        return self.type2ins(cen,off)

    def cluster2box(self, feats, preds):
        if not self.boxEnable:
            raise ValueError('box enable is False')
        assert len(feats) == len(preds), 'all feats len not equal preds len'
        k = max(preds)
        assert min(preds) == 1, 'cluster not begin at 1'
        centers = []
        offsets = []
        for i in range(1, k + 1):
            indexes = np.where(preds == i)[0]
            X = feats[indexes]
            center, offset = self.Cluster2Box(X)
            centers.append(center)
            offsets.append(offset)
        return torch.stack(centers), torch.stack(offsets)

    def cluster2box_level(self, feats, level_preds):
        if not self.boxEnable:
            raise ValueError('box enable is False')
        assert len(feats) == len(level_preds[0]), 'all feats len not equal preds len'
        level = len(level_preds)
        level_centers = [[] for _ in range(len(level_preds))]
        level_offsets = [[] for _ in range(len(level_preds))]
        level_child = {}
        level_parent = {}
        for i in range(level - 1):
            level_child[i], level_parent[i + 1] = self.findChild(level_preds, i)

        for i in range(level):
            k = max(level_preds[i])
            for j in range(1, k + 1):
                indexes = np.where(level_preds[i] == j)[0]
                X = feats[indexes]
                center, offset = self.Cluster2Box(X)
                level_centers[i].append(center)
                level_offsets[i].append(offset)
            level_centers[i] = torch.stack(level_centers[i])
            level_offsets[i] = torch.stack(level_offsets[i])
        return level_centers, level_offsets, {'child': level_child, 'parent': level_parent}

    def findChild(self, level_preds, i):
        assert min(level_preds[i]) == 1, 'cluster not begin at 1'
        k = max(level_preds[i])
        child = {}
        parent = {}
        for idx in range(1, k + 1):
            indexes = np.where(level_preds[i] == idx)[0]
            values = level_preds[i + 1][indexes]
            unique_values = np.unique(values)
            unique_values = unique_values - 1
            child[idx - 1] = unique_values
            for p in unique_values:
                parent.setdefault(p, -1)
                if parent[p] != -1:
                    raise Exception('level_pred not only one parent')
                parent[p] = idx - 1
        return child, parent


class BertForModel(nn.Module):
    def __init__(self, modelPath, args):
        super(BertForModel, self).__init__()
        self.num_labels = args.num_known_class
        self.model = AutoModelForMaskedLM.from_pretrained(modelPath)
        self.config = self.model.config
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        # self.dropout = nn.Dropout(0.7)
        self.mlp = nn.Linear(self.config.hidden_size, args.type_box_dim)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
        self.generator = None

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None, mode=None, generator=None):
        # self.generator = generator
        with torch.random.fork_rng(devices=[input_ids.device] if input_ids.is_cuda else []):
            # if generator is not None:
            #     if input_ids.is_cuda:
            #         torch.cuda.manual_seed_all(generator.seed())
            #     else:
            #         torch.manual_seed(generator.seed())
            outputs = self.model(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
            encoded_cls = outputs.hidden_states[-1][:, 0]
            output = self.dense(encoded_cls)
            output = self.activation(output)
            output = self.dropout(output)
            logits = self.classifier(output)
            output = self.mlp(output)

            if mode == 'feature_extract':
                return output
            elif mode == 'train':
                return output, logits
            elif mode == 'mlm':
                outputs = self.model(input_ids, attention_mask, token_type_ids, labels=labels)
                return outputs.loss
            elif mode == 'eval':
                return output
            elif mode == 'type':
                return encoded_cls
