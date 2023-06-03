"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

This code is modified by Linjie Li from Jin-Hwa Kim's repository.
https://github.com/jnhwkim/ban-vqa
MIT License
"""
import re 
import os
import time
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler

import wandb 
import utils
from tqdm import tqdm
from model.position_emb import prepare_graph_variables
                                 

def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2 # [batch_size, num_answers]
    loss = F.binary_cross_entropy_with_logits(logits, 
                                              labels, 
                                              reduction=reduction)
    if reduction == "mean":
        loss = loss * labels.size(1)
    return loss


def compute_score_with_logits(logits, labels, device):
    # argmax
    logits = torch.max(logits, 1)[1].data
    logits = logits.view(-1, 1)
    one_hots = torch.zeros(*labels.size()).to(device)
    one_hots.scatter_(1, logits, 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_loader, eval_loader, args, device=torch.device("cuda")):    

    wandb_run_name = os.path.basename(args.output)
    wandb_run_name = re.sub(r'^[0-9]{8}_', '', wandb_run_name)
    step = 0

    if args.wandb:
        run = wandb.init(project="vqa_regat",
                         entity='lect0099', 
                         name=wandb_run_name, 
                         group=args.wandb_group)
        
        if args.wandb == "run":
            run.config.learning_rate = args.base_lr
            run.config.epochs = args.epochs 
            run.config.optimizer = "torch.optim" + args.optimizer
            run.watch(model)
            
        # If using wandb.agent to run a wandb.sweep, the config is initialized 
        # by the sweep, i.e. the parameters can be re-written from the run.config       
        elif args.wandb == "sweep":
            args.base_lr = run.config.base_lr
            args.batch_size = run.config.batch_size
            args.epochs = run.config.epochs
            args.optimizer = run.config.optimizer
        else:
            raise ValueError("args.wandb accepts values from [None, 'run', 'sweep']")
        
        wandb_logger = utils.WandbLogger(run=run)

    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    utils.print_model(model, logger)
    
    N = len(train_loader.dataset)
    num_epochs = args.epochs
    base_lr = args.base_lr 
    
    momentum = args.momentum 
    weight_decay = args.weight_decay
    
        
    logger.write("------------ SETTINGS ------------")
    
    ### Optimizers ###
    if args.optimizer == 'SGD':
        ## Needs a lot of finetuning, probably investigate later...
        optim = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=base_lr, momentum=momentum, weight_decay=weight_decay)
        logger.write(f"Optimizer: {args.optimizer} with momentum={momentum}")
    else:
        ## NOTE: Adam + weight_decay is not the same as AdamW, refer to: 
        ## https://stackoverflow.com/questions/64621585/adamw-and-adam-with-weight-decay
        ## See the issues cited; PT docs are misleading, so as a rule of thumb:
        ## --> Adam with weight_decay: applies L2 regularization (adding to loss)
        ## --> AdamW: applies weight decay (subtracting from weights)
        
        # Adam can be used without weight_decay
        if args.optimizer == 'Adam':
            optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=base_lr, weight_decay=weight_decay) 
             
        # AdamW is supposed to be used with weight_decay (!), i.e. default = 1e-2
        elif args.optimizer == 'AdamW':
            optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=base_lr, weight_decay=weight_decay)
            
        # Default Adamax from ReGAT paper
        elif args.optimizer == "Adamax":
            optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=base_lr, weight_decay=weight_decay)
            
        else:
            raise ValueError("Chosen optimizer is not available!")  
        
        logger.write(f"Optimizer: {args.optimizer}")  
        
        
    ### LR Schedulers ###
    
    if args.lr_scheduler == "default":
        ## NOTE: Only the weights are needed, see the docs for LambdaLR: 
        ## https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html
        default_learning_rates = [0.5, 1, 1.5, 2] + [2] * 11 + [0.5] * 2 + [0.125] * 2 + [0.03125]
        default_lr_schedule = {k:v for k,v in enumerate(default_learning_rates)}   
        scheduler = lr_scheduler.LambdaLR(optim, lr_lambda=lambda epoch: default_lr_schedule[epoch])
        
        lr_epoch_update = True # Differentiate between epoch-based and step-based LR updates
        logger.write(f"Default LR scheduler with base_lr={base_lr:.6f}")
    else:
        init_lr, peak_lr, final_lr = args.init_lr, args.peak_lr, args.final_lr
        
        if args.lr_scheduler == "custom":
            ## NOTE: Implemented in utils.py, class WarmupConstantExpDecayLR
            ## Linear warmup, constant phase and exponential decay 
            begin_constant, begin_decay = args.begin_constant, args.begin_decay
            lr_epoch_update = True
            scheduler = utils.WarmupConstantExpDecayLR(optim, 
                                                       init_lr=init_lr,
                                                       peak_lr=peak_lr, 
                                                       final_lr=final_lr,
                                                       begin_constant=begin_constant,
                                                       begin_decay=begin_decay,
                                                       epochs=num_epochs)
            
            logger.write(f"Custom LR scheduler with init_lr={init_lr:.6f}, peak_lr={peak_lr:.6f}, final_lr={final_lr:.6f}")
            logger.write(f"Warmup until epoch {begin_constant}; constant until epoch {begin_decay}")

        elif args.lr_scheduler == "OCLR":
            ## NOTE: Step-based updates
            div_factor, final_div_factor = peak_lr / init_lr, peak_lr / final_lr
            strategy=args.strategy
            lr_epoch_update = False 
        
            scheduler = lr_scheduler.OneCycleLR(optim, 
                                                max_lr=peak_lr, 
                                                epochs=num_epochs,
                                                steps_per_epoch=N,
                                                pct_start=args.increase_frac, 
                                                anneal_strategy=strategy,        
                                                div_factor=div_factor,
                                                final_div_factor=final_div_factor)  
        
            logger.write(f"OCLR scheduler with init_lr={init_lr:.6f}, peak_lr={peak_lr:.6f}, final_lr={final_lr:.6f}")
            logger.write(f"Increase for approx. {int(args.increase_frac * num_epochs)} epochs " + \
                     f"(approx. {int(args.increase_frac * N * num_epochs)} steps)")
        
        else:
            raise ValueError("LR scheduler not available!")
        
    eval_score = 0
    relation_type = train_loader.dataset.relation_type

    for epoch in range(0, args.epochs):
        pbar = tqdm(total=len(train_loader))
        total_norm, count_norm = 0, 0
        total_loss, train_score = 0, 0
        count, average_loss, att_entropy = 0, 0, 0
        t = time.time()
        
        # Print out the learning rate for epoch, index by param_group
        logger.write(f"learning rate: {scheduler.get_last_lr()[-1]:.6f}")
        
        mini_batch_count = 0
        batch_multiplier = args.grad_accu_steps
        for i, (v, norm_bb, q, target, _, _, bb, spa_adj_matrix, sem_adj_matrix) in enumerate(train_loader):
            batch_size = v.size(0)
            num_objects = v.size(1)

            v = Variable(v).to(device)
            norm_bb = Variable(norm_bb).to(device)
            q = Variable(q).to(device)
            target = Variable(target).to(device)
            pos_emb, sem_adj_matrix, spa_adj_matrix = prepare_graph_variables(
                relation_type, bb, sem_adj_matrix, spa_adj_matrix, num_objects,
                args.nongt_dim, args.imp_pos_emb_dim, args.spa_label_num,
                args.sem_label_num, device)
            pred, att = model(v, norm_bb, q, pos_emb, sem_adj_matrix, spa_adj_matrix, target)
            loss = instance_bce_with_logits(pred, target)

            loss /= batch_multiplier
            loss.backward()
            mini_batch_count += 1

            if mini_batch_count == batch_multiplier:
                # init wandb logging (per mini-batch, maybe change step)
                if args.wandb and step > 10:
                    wandb_logger.log({"train_loss": loss, "epoch": epoch + ((i+1)/len(train_loader))}, step=step) 
                
                total_norm += nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                count_norm += 1
                optim.step()
                optim.zero_grad()
                mini_batch_count = 0
                step += 1
                
                # OCLR is step-based LR scheduler, updates at each training step
                if not lr_epoch_update:
                    scheduler.step()

            batch_score = compute_score_with_logits(pred, target, device).sum()
            total_loss += loss.data.item() * batch_multiplier * v.size(0)
            train_score += batch_score
            pbar.update(1)

            if args.log_interval > 0:
                average_loss += loss.data.item() * batch_multiplier
                if model.module.fusion == "ban":
                    current_att_entropy = torch.sum(calc_entropy(att.data))
                    att_entropy += current_att_entropy / batch_size / att.size(1)
                count += 1
                if i % args.log_interval == 0:
                    att_entropy /= count
                    average_loss /= count
                    
                    if args.wandb and step > 10:
                        wandb_logger.log({"att_entropy": att_entropy, "average_loss": average_loss}, step=step)
                    
                    print("step {} / {} (epoch {}), ave_loss {:.3f},".format(
                            i, len(train_loader), epoch,
                            average_loss),
                        "att_entropy {:.3f}".format(att_entropy))
                    average_loss = 0
                    count = 0
                    att_entropy = 0

        total_loss /= N
        train_score = 100 * train_score / N
        if eval_loader is not None:
            eval_score, bound, entropy = evaluate(
                model, eval_loader, device, args)

        if args.wandb and step > 10:
            wandb_logger.log({"epoch": epoch, "train_loss": total_loss, "train_score": train_score}, step=step) 
        
        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, norm: %.4f, score: %.2f'
                    % (total_loss, total_norm / count_norm, train_score))
        
        if eval_loader is not None:
            
            if args.wandb and step > 10:
                wandb_logger.log({"eval_score": 100 * eval_score}, step=step)
            
            logger.write('\teval score: %.2f (%.2f)'
                        % (100 * eval_score, 100 * bound))
            
            if entropy is not None:
                info = ''
                for i in range(entropy.size(0)):
                    info = info + ' %.2f' % entropy[i]
                logger.write('\tentropy: ' + info)
        if (eval_loader is not None)\
        or (eval_loader is None and epoch >= args.saving_epoch):
            logger.write("saving current model weights to folder")
            model_path = os.path.join(args.output, 'model_%d.pth' % epoch)
            opt = optim if args.save_optim else None
            utils.save_model(model_path, model, epoch, opt)
        
        # Default LR schedule is epoch-based, updates after each epoch 
        if lr_epoch_update:
            scheduler.step()


@torch.no_grad()
def evaluate(model, dataloader, device, args):
    model.eval()
    relation_type = dataloader.dataset.relation_type
    score = 0
    upper_bound = 0
    num_data = 0
    N = len(dataloader.dataset)
    entropy = None
    if model.module.fusion == "ban":
        entropy = torch.Tensor(model.module.glimpse).zero_().to(device)
    pbar = tqdm(total=len(dataloader))

    for i, (v, norm_bb, q, target, _, _, bb, spa_adj_matrix,
            sem_adj_matrix) in enumerate(dataloader):
        batch_size = v.size(0)
        num_objects = v.size(1)
        v = Variable(v).to(device)
        norm_bb = Variable(norm_bb).to(device)
        q = Variable(q).to(device)
        target = Variable(target).to(device)

        pos_emb, sem_adj_matrix, spa_adj_matrix = prepare_graph_variables(
            relation_type, bb, sem_adj_matrix, spa_adj_matrix, num_objects,
            args.nongt_dim, args.imp_pos_emb_dim, args.spa_label_num,
            args.sem_label_num, device)
        pred, att = model(v, norm_bb, q, pos_emb, sem_adj_matrix,
                          spa_adj_matrix, target)
        batch_score = compute_score_with_logits(
                        pred, target, device).sum()
        score += batch_score
        upper_bound += (target.max(1)[0]).sum()
        num_data += pred.size(0)
        if att is not None and 0 < model.module.glimpse\
                and entropy is not None:
            entropy += calc_entropy(att.data)[:model.module.glimpse]
        pbar.update(1)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    if entropy is not None:
        entropy = entropy / len(dataloader.dataset)
    model.train()
    return score, upper_bound, entropy


def calc_entropy(att):
    # size(att) = [b x g x v x q]
    sizes = att.size()
    eps = 1e-8
    p = att.view(-1, sizes[1], sizes[2] * sizes[3])
    return (-p * (p + eps).log()).sum(2).sum(0)  # g
