from typing import Optional

import torch
from transformers import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import AdamW


def mask_tokens(inputs, tokenizer, g, special_tokens_mask=None, mlm_probability=0.15):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    # if inputs.is_cuda and g.device.type != 'cuda':
    #     inputs.to('cuda:0')
    # elif not inputs.is_cuda and g.device.type == 'cuda':
    #     inputs.to('cpu')
    with torch.random.fork_rng(devices=[inputs.device] if inputs.is_cuda else []):
        # if g.device.type == 'cuda':
        #     torch.cuda.manual_seed_all(g.seed())
        # else:
        #     torch.manual_seed(g.seed())

        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix[torch.where(inputs == 0)] = 0.0
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def get_optimizer(model, args, mode):
    """
    获取优化器和学习率调度器。

    参数:
        args: 包含超参数的对象（如学习率、预热比例等）。

    返回:
        optimizer: 优化器。
        scheduler: 学习率调度器。
    """
    # 获取模型的所有参数
    # param_optimizer = list(model.named_parameters())
    #
    # # 定义不需要权重衰减的参数（如 bias 和 LayerNorm）
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    #
    # # 分组参数
    # optimizer_grouped_parameters = [
    #     {
    #         'params': [p for n, p in param_optimizer if not any(nd in n for nd in (no_decay + box_name))],
    #         'weight_decay': args.weight_decay,  # 权重衰减
    #     },
    #     {
    #         'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    #         'weight_decay': 0.0,  # 不应用权重衰减
    #     }
    # ]
    base_param = list(model.model.named_parameters())
    box_param = list(model.Cluster2Box.named_parameters())
    type_param = list(model.type_model.named_parameters())
    proj_param = list(model.proj.named_parameters())
    ins2type_param = list(model.ins2type.named_parameters())
    ins2type_param.extend(list(model.type2ins.named_parameters()))
    type_emb_param = list(model.type_embeding.named_parameters())

    # 定义不需要权重衰减的参数（如 bias 和 LayerNorm）
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    # 分组参数
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in base_param if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,  # 权重衰减
            'lr': args.lr
        },
        {
            'params': [p for n, p in base_param if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,  # 不应用权重衰减
            'lr': args.lr
        },
        {
            'params': [p for n, p in box_param if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,  # 权重衰减
            'lr': args.lr_box
        },
        {
            'params': [p for n, p in box_param if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,  # 不应用权重衰减
            'lr': args.lr_box
        },
        {
            'params': [p for n, p in type_param if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,  # 权重衰减
            'lr': args.lr
        },
        {
            'params': [p for n, p in type_param if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,  # 不应用权重衰减
            'lr': args.lr
        },
        {
            'params': [p for n, p in proj_param if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,  # 权重衰减
            'lr': args.lr_box
        },
        {
            'params': [p for n, p in proj_param if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,  # 不应用权重衰减
            'lr': args.lr_box
        },
        {
            'params': [p for n, p in ins2type_param if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,  # 权重衰减
            'lr': args.lr_box
        },
        {
            'params': [p for n, p in ins2type_param if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,  # 不应用权重衰减
            'lr': args.lr_box
        },
        {
            'params': [p for n, p in type_emb_param if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,  # 权重衰减
            'lr': args.lr_box
        },
        {
            'params': [p for n, p in type_emb_param if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,  # 不应用权重衰减
            'lr': args.lr_box
        }
    ]

    # 初始化 AdamW 优化器
    if mode == 'bert':
        optimizer = AdamW(
            optimizer_grouped_parameters,
            # lr=args.lr,  # 学习率
            eps=args.adam_epsilon  # Adam 的 epsilon 参数
        )
    elif mode == 'box':
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.lr_box,  # 学习率
            eps=args.adam_epsilon  # Adam 的 epsilon 参数
        )

    # 初始化学习率调度器
    scheduler = get_lr_scheduler(args, optimizer)

    return optimizer, scheduler


def get_lr_scheduler(args, optimizer):
    """Returns lr scheduler."""
    if args.scheduler_type == "linear":
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_epoch * args.epoch_steps,
                                               num_training_steps=args.num_epoch * args.epoch_steps)
