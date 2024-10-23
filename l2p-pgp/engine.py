# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from timm.utils import accuracy
from timm.optim import create_optimizer

import utils
import memory


def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, feature_mat, key_feature_mat, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None):

    model.train(set_training_mode)
    original_model.eval()
    is_base = True if task_id == 0 else False
    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if model.composition:
        metric_logger.add_meter('Comp_Lr', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
        
        output = model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode)
        logits = output['logits']
        feat_map = output['final_map']
        if model.composition:
            if not args.distributed:
                map_metric_logits = model.map_metric_logits(feat=feat_map)
                if args.primitive_recon_cls_weight != 0:
                    recon_map_logits = model.map_metric_recon_logits(feat=feat_map, is_base=is_base, task_id=task_id, device=device)
            else:
                map_metric_logits = model.modules.map_metric_logits(feat=feat_map)
                if args.primitive_recon_cls_weight != 0:
                    recon_map_logits = model.modules.map_metric_recon_logits(feat=feat_map, is_base=is_base, task_id=task_id, device=device)
        
        prompt_idx = output['prompt_idx'][0]

        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id] 
            # print("Class mask", mask)
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
            # print("Logits shape", logits.shape)
            loss = criterion(logits, target) # base criterion (CrossEntropyLoss)
            # print("Base loss: ", loss)
            if model.composition:
                map_metric_logits = map_metric_logits.index_fill(dim=1, index=not_mask, value=float('-inf')) 
                # print("Map logits shape: ", map_metric_logits.shape)
                # print("Map logits: ", map_metric_logits)               
                map_metric_loss = criterion(map_metric_logits, target)
                loss = args.backbone_feat_cls_weight*loss + args.map_metric_cls_weight*map_metric_loss
                # print("Base+Compare loss: ", loss)
                if args.primitive_recon_cls_weight != 0:
                    # print(map_metric_logits[:, mask])
                    map_metric_logits[:, mask] = recon_map_logits
                    # print("Reconstruct logits:",map_metric_logits)
                    # print("recon shape", recon_map_logits.shape)
                    recon_loss = criterion(map_metric_logits, target)
                    loss += args.primitive_recon_cls_weight * recon_loss   
                    # print("Total loss: ", loss)   
                
        if args.pull_constraint and 'reduce_sim' in output:
            loss = loss - args.pull_constraint_coeff * output['reduce_sim']

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        if model.composition:
            ind_acc1, ind_acc5 = accuracy(map_metric_logits, target, topk=(1,5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Gradient Projection
        if task_id != 0 and not args.no_pgp:
            for k, (m, params) in enumerate(model.named_parameters()):
                if m == "prompt.prompt":
                    params.grad.data[prompt_idx] = params.grad.data[prompt_idx] - torch.matmul(
                        params.grad.data[prompt_idx], feature_mat)
                if m == "prompt.prompt_key":
                    params.grad.data[prompt_idx] = params.grad.data[prompt_idx] - torch.mm(
                        params.grad.data[prompt_idx], key_feature_mat)
               
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        if model.composition:
            metric_logger.update(Comp_Lr=optimizer.param_groups[1]["lr"])
            metric_logger.meters['Ind_Acc@1'].update(ind_acc1.item(), n=input.shape[0])
            metric_logger.meters['Ind_Acc@5'].update(ind_acc5.item(), n=input.shape[0])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, args=None,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            output = model(input, task_id=task_id, cls_features=cls_features)
            logits = output['logits']
            feat_map = output['final_map']
            if model.composition:
                map_metric_logits = model.map_metric_logits(feat=feat_map)

            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask
                if model.composition:
                    map_metric_logits = map_metric_logits + logits_mask
            # print("target in evaluate: ", target)
            # print("logits in evaluate: ", logits)        
            loss = criterion(logits, target)
            
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
            if model.composition:
                map_metric_loss = criterion(map_metric_logits, target)
                combine_logits = (map_metric_logits + logits)/2
                ind_acc1, ind_acc5 = accuracy(map_metric_logits, target, topk=(1, 5))
                combine_acc1, combine_acc5 = accuracy(combine_logits, target, topk=(1,5))
                
                metric_logger.meters['Map_metric_loss'].update(map_metric_loss.item())
                metric_logger.meters['Ind_acc@1'].update(ind_acc1.item(), n=input.shape[0])
                metric_logger.meters['Ind_acc@5'].update(ind_acc5.item(), n=input.shape[0])
                metric_logger.meters['Combine_acc@1'].update(combine_acc1.item(), n=input.shape[0])
                metric_logger.meters['Combine_acc@5'].update(combine_acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))
    if model.composition:
        print('* Ind_Acc@1 {top1.global_avg:.3f} Ind_Acc@5 {top5.global_avg:.3f} map_loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.meters['Ind_acc@1'], top5=metric_logger.meters['Ind_acc@5'], losses=metric_logger.meters['Map_metric_loss']))
        print('* Combine_Acc@1 {top1.global_avg:.3f} Combine_Acc@5 {top5.global_avg:.3f} '
            .format(top1=metric_logger.meters['Combine_acc@1'], top5=metric_logger.meters['Combine_acc@5']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                      device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id+1):
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                              device=device, task_id=i, class_mask=class_mask, args=args)
        if model.composition:
            stat_matrix[0, i] = test_stats['Combine_acc@1']
            stat_matrix[1, i] = test_stats['Combine_acc@5']
            stat_matrix[2, i] = test_stats['Loss'] + test_stats['Map_metric_loss']

            acc_matrix[i, task_id] = test_stats['Combine_acc@1']
        else:
            stat_matrix[0, i] = test_stats['Acc@1']
            stat_matrix[1, i] = test_stats['Acc@5']
            stat_matrix[2, i] = test_stats['Loss']

            acc_matrix[i, task_id] = test_stats['Acc@1']
        
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats


def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, args=None,):

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    feature, feature_mat = None, None
    key_feature, key_feature_mat = None, None

    for task_id in range(args.num_tasks):
        if not args.no_pgp and task_id != 0:
            print("prompt feature shape", feature.shape)
            Uf = torch.Tensor(np.dot(feature, feature.transpose())).to(device)
            print('Prompt Projection Matrix Shape: {}'.format(Uf.shape))
            feature_mat = Uf

            print("key feature shape", key_feature.shape)
            Uf = torch.Tensor(np.dot(key_feature, key_feature.transpose())).to(device)
            print('Key Projection Matrix Shape: {}'.format(Uf.shape))
            key_feature_mat = Uf

        # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (slice(cur_start, cur_end))
                    prev_idx = (slice(prev_start, prev_end))

                    with torch.no_grad():
                        if args.distributed:
                            model.module.prompt.prompt.grad.zero_()
                            model.module.prompt.prompt[cur_idx] = model.module.prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            model.prompt.prompt.grad.zero_()
                            model.prompt.prompt[cur_idx] = model.prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.parameters()
                    
        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                with torch.no_grad():
                    if args.distributed:
                        model.module.prompt.prompt_key.grad.zero_()
                        model.module.prompt.prompt_key[cur_idx] = model.module.prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.module.parameters()
                    else:
                        model.prompt.prompt_key.grad.zero_()
                        model.prompt.prompt_key[cur_idx] = model.prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.parameters()
    
                    
        print("----------Trainable Parameters----------")
        for name, params in model.named_parameters():
            if params.requires_grad:
                print(name)    
        print("----------------------------------------")       
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Number of trainable params:', n_parameters)
        print("----------------------------------------")       
        n_parameters = sum(p.numel() for p in model.parameters())
        print('Total number of params:', n_parameters)                 
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            # Double compositional learning rate after each task
            args.lr = args.lr * 0.5 if args.lr < 0.1 else args.lr
            
            # optimizer = create_optimizer(args, model)
            proto = [p for name, p in model.named_parameters() if 'proto' in name]
            others = [p for name, p in model.named_parameters() if 'proto' not in name]

            parameters = [{'params': others},
                     {'params': proto, 'lr': args.comp_lr},
                    ]
            # optimizer = create_optimizer(args, model_without_ddp)
            optimizer = create_optimizer(args, parameters)
            
        print("----------------Training----------------")            
        
        # Training
        # if task_id != 0:        
        #     args.epochs = 15
            
        for epoch in range(args.epochs):
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion,
                                        data_loader=data_loader[task_id]['train'], optimizer=optimizer, device=device,
                                        epoch=epoch, feature_mat=feature_mat, key_feature_mat=key_feature_mat, max_norm=args.clip_grad,
                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args)

            if lr_scheduler:
                lr_scheduler.step(epoch)
                
        # Evaluating
        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                       task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
        
        #Update feature matrix, key_feature matrix
        if not args.no_pgp:
            model.eval()
            original_model.eval()
            mem_example = memory.get_representation_matrix(data_loader[task_id]['mem'], device)
            # rep, rep_key = memory.get_rep(model, original_model, mem_example, task_id)
            _, rep_key = memory.get_rep(model, original_model, mem_example, task_id)
            rep = model.proto[task_id].permute(0,2,1).reshape(-1, 768).detach().cpu().numpy()
            
            # rep = torch.cat(rep)
            # rep = rep.detach().cpu().numpy()
            
            pca = PCA(n_components=9)
            pca = pca.fit(rep)
            rep = pca.transform(rep)
            
            # print(rep.shape)
            # if task_id != 0:
            for k, (m, params) in enumerate(model.named_parameters()):
                if m == "prompt.prompt":
                    p_ = params.data
                    p_ = p_.view(-1, 768).detach().cpu().numpy()#.transpose(1, 0)

            pca = PCA(n_components=9)
            pca = pca.fit(p_)
            p = pca.transform(p_)
            # rep = rep + p
            rep = np.concatenate((rep, p), axis=0) #Replace element-wise summation with concatenation
               
            rep_key = torch.cat(rep_key)
            rep_key = rep_key.detach().cpu().numpy()
            pca = PCA(n_components=5)
            pca = pca.fit(rep_key)
            rep_key = pca.transform(rep_key)

            feature = memory.update_memory(rep, 0.6, feature)
            key_feature = memory.update_memory(rep_key, 0.97, key_feature)

        
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            state_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            
            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

