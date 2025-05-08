import os.path as op
import pandas as pd
import numpy as np
import random
import torch

def get_data(data_path, split, tlen=0):
    data = pd.read_pickle(data_path)
    new_data = {'seq':[],'next':[], 'len_seq':[]}
    for i in range(len(data['seq'])):
        sample=data.iloc[i]
        if sample['len_seq']>3:
            if tlen>0:
                if (sample[f"{split}_idx"][0] is not None) and (sample[f"{split}_idx"][1] is not None):
                    for idx in range(max(3,sample[f"{split}_idx"][0]),sample[f"{split}_idx"][1]+1):
                        tseq = sample['seq'][max(0,idx-tlen):idx]
                        new_data['seq'].append(tseq)
                        new_data['next'].append(sample['seq'][idx])
                        new_data['len_seq'].append(len(tseq))
            else:
                if (sample[f"{split}_idx"][0] is not None) and (sample[f"{split}_idx"][1] is not None):
                    for idx in range(max(3,sample[f"{split}_idx"][0]),sample[f"{split}_idx"][1]+1):
                        tseq = sample['seq'][:idx]
                        new_data['seq'].append(tseq)
                        new_data['next'].append(sample['seq'][idx])
                        new_data['len_seq'].append(len(tseq))
    new_data = pd.DataFrame(new_data)
    return new_data

def get_prompt(prompt_path):
    if op.isfile(prompt_path):
        with open(prompt_path, 'r') as f:
            prompts = f.readlines()
            prompts = [prompt.strip() for prompt in prompts]
    return prompts

def get_id2title(data_dir):
    id2title = dict()
    item_path=op.join(op.join(data_dir), 'id2title.txt')
    with open(item_path, 'r') as f:
        for l in f.readlines():
            ll = l.strip('\n').split('||')
            id2title[int(ll[0])] = ll[1].strip()
    return id2title

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_parameters_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params}")
    print(f"Total trainable parameters: {total_trainable_params}")

def find_sub_list(lst, sub_lst):
    length = len(sub_lst)
    ret = -1
    for index in range(len(lst)):
        if lst[index:index+length] == sub_lst:
            ret = index+length-1
    return ret

def create_prefix_allowed_tokens_fn(candidate_token_ids, prefix_token_ids, tokenizer):
    max_height = max([len(one) for one in candidate_token_ids])
    trie = {}
    for token_ids in candidate_token_ids:
        level = trie
        token_ids = [prefix_token_ids[-1]] + token_ids
        for tidx, token_id in enumerate(token_ids):
            if token_id not in level:
                level[token_id] = {}
            level = level[token_id]
    def prefix_allowed_tokens_fn(batch_id, input_ids):
        input_ids = input_ids.tolist()
        start_idx = find_sub_list(input_ids, prefix_token_ids)
        if start_idx<0:
            return []
        else:
            start = trie
            for current_token in input_ids[start_idx:]:
                if current_token in start:
                    start = start[current_token]
                else:
                    break
            next_tokens = list(start.keys())
            if len(next_tokens)>0:
                return next_tokens
            else:
                return [tokenizer.eos_token_id]

    return prefix_allowed_tokens_fn