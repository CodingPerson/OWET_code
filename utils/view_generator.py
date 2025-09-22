import copy

import numpy as np
import torch

from utils.forTrain import mask_tokens


class view_generator:
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def random_token_replace(self, ids):
        mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        ids, _ = mask_tokens(ids, self.tokenizer, self.args.cpu_g, mlm_probability=self.args.mask_prob)
        random_words = torch.randint(len(self.tokenizer), ids.shape, dtype=torch.long)
        indices_replaced = torch.where(ids == mask_id)
        ids[indices_replaced] = random_words[indices_replaced]
        return ids