"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

This code is modified by Linjie Li from Jin-Hwa Kim's repository.
https://github.com/jnhwkim/ban-vqa
MIT License
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

import utils
from train import compute_score_with_logits
from dataset import Dictionary, VQAEvalDataset
# from dataset_cp_v2 import VQA_cp_Dataset, Image_Feature_Loader

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

''' Example use-case on VQA dataset with logits from 3 relation types
: --data_folder should be kept fixed
: --output_folder can be configured
: --output_name can be configured
: --save_answers to produce updated json

python3 eval_weighted.py \
  --logits "/hpcwork/lect0099/pretrained_models/regat_implicit/ban_1_implicit_vqa_196/eval/logits_vqa_val.npy" \
"/hpcwork/lect0099/pretrained_models/regat_semantic/ban_1_semantic_vqa_7971/eval/logits_vqa_val.npy" \
"/hpcwork/lect0099/pretrained_models/regat_spatial/ban_1_spatial_vqa_1687/eval/logits_vqa_val.npy" \
  --alphas 0.3 0.4 \
  --data_folder "/hpcwork/lect0099/data" \
  --output_folder "/hpcwork/lect0099/test_weighted_eval" \
  --output_name "test" \
  --save_answers
'''


def evaluate(dataloader, batch_size, args, device):
    label2ans = dataloader.dataset.label2ans
    num_answers = len(label2ans)
    N = len(dataloader.dataset)
    results, relations = [], []
    score = 0
    pbar = tqdm(total=len(dataloader))
    
    # args.logits is a list of paths to logit files
    assert args.logits and args.alphas and len(args.logits) > 1
        
    # Extract all logits from specified paths
    logits = []
    for path in args.logits:
        logits.append(np.load(path))
        # Assuming paths are well-defined
        if 'implicit' in path: 
            relations.append('implicit')
        elif 'semantic' in path: 
            relations.append('semantic')
        elif 'spatial' in path: 
            relations.append('spatial')
        else:
            pass
    logits, alphas = np.array(logits), np.array(args.alphas) 
    # logits, alphas = torch.FloatTensor(logits), torch.FloatTensor(args.alphas)
    
    # Preliminary checks for consistency
    assert len(logits) == len(alphas) + 1, "len(logits) != len(alphas) + 1"
    assert np.all((alphas >= 0) & (alphas <= 1)) and np.sum(alphas) <= 1, "invalid weights"
    # assert torch.all((alphas >= 0) & (alphas <= 1)) and torch.sum(alphas) <= 1, "invalid weights"
    
    # Calculate weighted prediction logits
    logits = np.transpose(logits, (1, 2, 0))
    alphas = np.append(alphas, 1-np.sum(alphas))
    # logits = torch.permute(logits, (1, 2, 0))
    # alphas = torch.cat((alphas, torch.FloatTensor([1 - torch.sum(alphas)])))

    result = np.dot(logits, alphas)
    # result = torch.matmul(logits, alphas)
    
    # Obtain the batch size from the dataloader, prepare predictions
    # pred       :size [num_batches (N-1) x batch_size x num_answers]
    # pref_final :size [final_batch_size x num_answers]
    
    num_batches = result.shape[0] // batch_size 
    # num_batches = result.size(0) // batch_size 
    pred_size = num_batches * batch_size 
    res, res_final = result[:pred_size], result[pred_size:]
    res = np.reshape(res, (-1, batch_size, num_answers))
    # res = torch.reshape(res, (-1, batch_size, num_answers))
    
    # Tensorise and attach to cuda device 
    res = torch.from_numpy(res).to(device)
    res_final = torch.from_numpy(res_final).to(device)
    
    for i, (_, target, qid, _) in enumerate(dataloader):
        if target.size(-1) == num_answers:
            target = Variable(target).to(device)
            pred = res[i] if i < len(dataloader)-1 else res_final
            batch_score = compute_score_with_logits(pred, target, device).sum()
            score += batch_score
            
        if args.save_answers:
            qid = qid.cpu()
            pred = pred.cpu()
            current_results = make_json(pred, qid, dataloader)
            results.extend(current_results)
    
        pbar.update(1)
    

    score = score / N
    results_folder = f"{args.output_folder}"
    results_suffix = '_'.join(relations)
    
    # As before, maybe useful to enable save_answers 
    if args.save_answers:
        utils.create_dir(results_folder)
        save_to = f"{results_folder}/{args.output_name}.json"
        # save_to = f"{results_folder}/{args.dataset}_" +\
        #     f"{args.split}_" + f"{results_suffix}.json"
        json.dump(results, open(save_to, "w"))
    return score
            
def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]


def make_json(logits, qIds, dataloader):
    utils.assert_eq(logits.size(0), len(qIds))
    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = qIds[i].item()
        result['answer'] = get_answer(logits[i], dataloader)
        results.append(result)
    return results


def parse_args():
    parser = argparse.ArgumentParser()

    '''
    For eval logistics
    '''
    parser.add_argument('--save_logits', action='store_true',
                        help='save logits')
    parser.add_argument('--save_answers', action='store_true',
                        help='save predicted answers')
    parser.add_argument('--output_folder', type=str, default="",
                        help="folder to store logits/answers")
    parser.add_argument('--output_name', type=str, default="",
                        help="filename for stored json")

    '''
    For dataset
    '''
    parser.add_argument('--data_folder', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='vqa',
                        choices=["vqa", "vqa_cp"])
    parser.add_argument('--split', type=str, default="val",
                        choices=["val", "test", "test2015"],
                        help="test for vqa_cp, test2015 for vqa")
    
    '''
    For weighted sum
    '''
    parser.add_argument('--logits', nargs="*", type=str, default=None)
    parser.add_argument('--alphas', nargs="*", type=float, default=None)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")

    n_device = torch.cuda.device_count()
    print(f"Found {n_device} GPU cards for eval")
    device = torch.device("cuda")
    
    # Keep dictionary 
    dictionary = Dictionary.load_from_file(
                    os.path.join(args.data_folder, 'glove/dictionary.pkl'))
    
    # Hardcode batch_size to 256 for weighted sum evaluation
    batch_size = 256 * n_device

    print(f"Evaluating on {args.dataset} dataset")
    # For now, only solve the "vqa" case
        
    eval_dset = VQAEvalDataset(args.split, dictionary, dataroot=args.data_folder, adaptive=True)
            
    eval_loader = DataLoader(
        eval_dset, batch_size, shuffle=False,
        num_workers=4, collate_fn=utils.trim_collate)

    eval_score = evaluate(eval_loader, batch_size, args, device)
    print('\teval score: %.2f' % (100 * eval_score))
