import json
import collections
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import data
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Dataset, download_url
from torch_geometric.loader import DataLoader
from lung_function.modules.datasets import clean_data, pat_from_json
from torch_geometric.nn import global_mean_pool, TopKPooling, ASAPooling

import networkx as nx 
# -*- coding: utf-8 -*-
# @Time    : 4/5/22 12:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
# log_dict is used to record super parameters and metrics

import sys
sys.path.append("/home/jjia/data/lung_function/lung_function/modules/networks/models_pcd")
sys.path.append("/home/jjia/data/lung_function/lung_function/modules")

import random
import statistics
import threading
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from medutils import medutils
from medutils.medutils import count_parameters
from mlflow import log_metric, log_metrics, log_param, log_params
from mlflow.tracking import MlflowClient
from monai.utils import set_determinism
from typing import List, Sequence
from argparse import Namespace
import functools
import thop
import os
import copy
import pandas as pd
from glob import glob
from torch import nn
import torch
import torch.nn.functional as F
from lung_function.modules.path import PFTPath
import os

from monai.data import DataLoader
from lung_function.modules.networks.get_net import FCNet, GCN
from lung_function.modules import provider
from lung_function.modules.compute_metrics import icc, metrics
from lung_function.modules.dataset_gcn import all_loaders
from lung_function.modules.loss import get_loss
from lung_function.modules.path import PFTPath
from lung_function.modules.set_args import get_args
from lung_function.modules.tool import (record_1st, dec_record_cgpu, retrive_run, try_func, int2str, mae, me, mre, 
                                        ensemble_4folds_validation, ensemble_4folds_testing, 
                                        log_metrics_all_folds_average, average_all_folds)
import optuna


args = get_args()
global_lock = threading.Lock()


log_metric = try_func(log_metric)
log_metrics = try_func(log_metrics)

    
    
class Run:
    """A class which has its dataloader and step_iteration. It is like Lighting. 
    """

    def __init__(self, args: Namespace, dataloader_flag=True):
        self.args = args
        self.device = torch.device("cuda")  # 'cuda'
        self.target = [i.lstrip() for i in args.target.split('-')]
        
        print('conv name', args.gconv_name)
        
        self.net = GCN(in_chn=4, out_chn=len(self.target), args=args)  # receive ct and pcd as input
        self.net.to(self.device)
        self.fold = args.fold
        self.flops_done = False
        self.mypath = PFTPath(args.id, check_id_dir=False, space=args.ct_sp)
        
        print('net:', self.net)

        net_parameters = count_parameters(self.net)
        net_parameters = str(round(net_parameters / 1e6, 2))
        log_param('net_parameters_M', net_parameters)

        self.loss_fun = get_loss(args.loss, mat_diff_loss_scale=args.mat_diff_loss_scale)
        self.opt = torch.optim.Adam( self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.net = self.net.to(self.device)
        self.dataloader = all_loaders(args, nb=10000)

        validMAEEpoch_AllBest = 1000


        self.BestMetricDt = {'trainLossEpochBest': 1000,
                             # 'trainnoaugLossEpochBest': 1000,
                             'validLossEpochBest': 1000,
                             'testLossEpochBest': 1000,

                             'trainMAEEpoch_AllBest': 1000,
                             # 'trainnoaugMAEEpoch_AllBest': 1000,
                             'validMAEEpoch_AllBest': validMAEEpoch_AllBest,
                             'testMAEEpoch_AllBest': 1000,
                             }

    def step(self, mode, epoch_idx, save_pred=False, suffix=None):

        loss_fun_mae = nn.L1Loss()

        scaler = torch.cuda.amp.GradScaler()
        print(mode + "ing ......")
        if mode == 'train':
            self.net.train()
        else:
            self.net.eval()

        t0 = time.time()
        data_idx = 0
        loss_accu = 0
        mae_accu_ls = [0 for _ in self.target]
        mae_accu_all = 0
        len_data = len(self.dataloader[mode])
        for data_batch in self.dataloader[mode]:
            data_batch.x = data_batch.x.to(self.device)
            data_batch.edge_index = data_batch.edge_index.to(self.device)
            
            data_batch.batch = data_batch.batch.to(self.device)
            data_batch.y = data_batch.y.to(self.device)
            
            torch.cuda.empty_cache()  # avoid memory leak

            if not self.flops_done:  # only calculate macs and params once
                macs, params = thop.profile(self.net, inputs=(data_batch.x, data_batch.edge_index, data_batch.batch ))
                self.flops_done = True
                log_param('macs_G', str(round(macs/1e9, 2)))
                log_param('net_params_M', str(round(params/1e6, 2)))
                
            data_idx += 1

            t1 = time.time()
            with torch.cuda.amp.autocast():
                if mode != 'train' or save_pred:  # save pred for inference
                    with torch.no_grad():
                        pred = self.net(data_batch.x, data_batch.edge_index, data_batch.batch)
                else:
                    pred = self.net(data_batch.x, data_batch.edge_index, data_batch.batch)
                
                if save_pred:
                    head = ['pat_id']
                    head.extend(self.target)

                    batch_pat_id = np.array(data_batch.pat_id)  # shape (N,1)
                    batch_pat_id = int2str(batch_pat_id)  # shape (N,1)

                    batch_y_np = data_batch.y.reshape(pred.shape).cpu().detach().numpy()  # shape (N, out_nb)
                    pred_np = pred.cpu().detach().numpy()  # shape (N, out_nb)

                    saved_label = np.hstack((batch_pat_id, batch_y_np))
                    saved_pred = np.hstack((batch_pat_id, pred_np))
                    
                    if suffix not in (None, 0, '0'):
                        pred_fpath = self.mypath.save_pred_fpath(mode).replace('.csv', '_'+ suffix + '.csv')
                        label_fpath = self.mypath.save_label_fpath(mode).replace('.csv', '_'+ suffix + '.csv')
                    else:
                        pred_fpath = self.mypath.save_pred_fpath(mode)
                        label_fpath = self.mypath.save_label_fpath(mode)
                        
                    medutils.appendrows_to(label_fpath, saved_label, head=head)
                    medutils.appendrows_to(pred_fpath, saved_pred, head=head)

                loss = self.loss_fun(pred, data_batch.y.reshape(pred.shape))
                with torch.no_grad():
                    mae_ls = [loss_fun_mae(pred[:, i], data_batch.y.reshape(pred.shape)[:, i]).item() for i in range(len(self.target))]
                    mae_all = loss_fun_mae(pred, data_batch.y.reshape(pred.shape)).item()
                

                    

            if mode == 'train' and save_pred is not True:  # update gradients only when training
                self.opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.opt)
                scaler.update()
                
            loss_cpu = loss.item()
            print('loss:', loss_cpu)

            loss_accu += loss_cpu
            for i, mae in enumerate(mae_ls):
                mae_accu_ls[i] += mae
            mae_accu_all += mae_all

            # print('pred:', pred.clone().detach().cpu().numpy())
            # print('label:', batch_y.clone().detach().cpu().numpy())
            if epoch_idx < 3:
                t2 = time.time()
                log_metric('TUpdateWBatch', t2-t1, data_idx + epoch_idx*len_data)
                t0 = t2  # reset the t0

        log_metric(mode+'LossEpoch', loss_accu/len_data, epoch_idx)
        log_metric(mode+'MAEEpoch_All', mae_accu_all / len_data, epoch_idx)
        for t, i in zip(self.target, mae_accu_ls):
            log_metric(mode + 'MAEEpoch_' + t, i / len_data, epoch_idx)

        self.BestMetricDt[mode + 'LossEpochBest'] = min( self.BestMetricDt[mode+'LossEpochBest'], loss_accu/len_data)
        tmp = self.BestMetricDt[mode+'MAEEpoch_AllBest']
        self.BestMetricDt[mode + 'MAEEpoch_AllBest'] = min( self.BestMetricDt[mode+'MAEEpoch_AllBest'], mae_accu_all/len_data)

        log_metric(mode+'LossEpochBest', self.BestMetricDt[mode + 'LossEpochBest'], epoch_idx)
        log_metric(mode+'MAEEpoch_AllBest', self.BestMetricDt[mode + 'MAEEpoch_AllBest'], epoch_idx)

        if self.BestMetricDt[mode+'MAEEpoch_AllBest'] == mae_accu_all/len_data:
            for t, i in zip(self.target, mae_accu_ls):
                log_metric(mode + 'MAEEpoch_' + t + 'Best', i / len_data, epoch_idx)

            if mode == 'valid':
                print(
                    f"Current mae is {self.BestMetricDt[mode+'MAEEpoch_AllBest']}, better than the previous mae: {tmp}, save model.")
                ckpt = {'model': self.net.state_dict(),
                        'metric_name': mode+'MAEEpoch_AllBest',
                        'current_metric_value': self.BestMetricDt[mode+'MAEEpoch_AllBest']}
                torch.save(ckpt, self.mypath.model_fpath)
        


@dec_record_cgpu(args.outfile)
def run(args: Namespace):
    """
    Run the whole  experiment using this args.
    """
    myrun = Run(args)
    modes = ['valid', 'test', 'train'] if args.mode != 'infer' else ['valid', 'test']
    if args.mode == 'infer':
        for mode in ['valid', 'test', 'train']:
            if mode=='train':
                steps = 100
            else:
                steps = 1
            for i in range(steps):
                myrun.step(mode,  0,  save_pred=True)
    else:  # 'train' or 'continue_train'
        for i in range(args.epochs):  # 20000 epochs
            myrun.step('train', i)
            if i % args.valid_period == 0:  # run the validation
                myrun.step('valid',  i)
                myrun.step('test',  i)
            if i == args.epochs - 1:  # load best model and do inference
                print('start inference')
                if os.path.exists(myrun.mypath.model_fpath):
                    ckpt = torch.load(myrun.mypath.model_fpath,
                                      map_location=myrun.device)
                    if isinstance(ckpt, dict) and 'model' in ckpt:
                        model = ckpt['model']
                    else:
                        model = ckpt
                    # model_fpath need to exist
                    myrun.net.load_state_dict(model, strict=False)
                    print(f"load net from {myrun.mypath.model_fpath}")
                else:
                    print(
                        f"no model found at {myrun.mypath.model_fpath}, let me save the current model to this lace")
                    ckpt = {'model': myrun.net.state_dict()}
                    torch.save(ckpt, myrun.mypath.model_fpath)
                for mode in modes:
                    
                    myrun.step(mode, i, save_pred=True)

    mypath = PFTPath(args.id, check_id_dir=False, space=args.ct_sp)
    label_ls = [mypath.save_label_fpath(mode) for mode in modes]
    pred_ls = [mypath.save_pred_fpath(mode) for mode in modes]

    for pred_fpath, label_fpath in zip(pred_ls, label_ls):
        r_p_value = metrics(pred_fpath, label_fpath, ignore_1st_column=True)
        log_params(r_p_value)
        print('r_p_value:', r_p_value)

        icc_value = icc(label_fpath, pred_fpath, ignore_1st_column=True)
        log_params(icc_value)
        print('icc:', icc_value)
        
    print('Finish all things!')
    return myrun.BestMetricDt['validMAEEpoch_AllBest']


        

        
def main():
   
    SEED = 4
    set_determinism(SEED)  # set seed for this run

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.manual_seed(SEED)

    random.seed(SEED)
    np.random.seed(SEED)

    mlflow.set_tracking_uri("http://nodelogin01:5000")
    experiment = mlflow.set_experiment("lung_fun_gcn")
    
    RECORD_FPATH = f"{Path(__file__).absolute().parent}/results/record.log"
    # write super parameters from set_args.py to record file.
    id = record_1st(RECORD_FPATH)

    with mlflow.start_run(run_name=str(id), tags={"mlflow.note.content": args.remark}):
        args.id = id  # do not need to pass id seperately to the latter function
        args.gconv_name = 'GATConv' 
        args.gnorm ='InstanceNorm'
        args.heads = 1
        args.batch_size = 16
        args.hidden_channels = 128
        args.layers_nb = 2
        
        current_id = id
        tmp_args_dt = vars(args)
        tmp_args_dt['fold'] = 'all'
        log_params(tmp_args_dt)

        all_folds_id_ls = []
        for fold in [1, 2, 3, 4]:
            # write super parameters from set_args.py to record file.

            id = record_1st(RECORD_FPATH)
            all_folds_id_ls.append(id)
            with mlflow.start_run(run_name=str(id) + '_fold_' + str(fold), tags={"mlflow.note.content": f"fold: {fold}"}, nested=True):
                args.fold = fold
                args.id = id  # do not need to pass id seperately to the latter function
                # args.mode = 'infer'  # disable it for normal training
                tmp_args_dt = vars(args)
                log_params(tmp_args_dt)
                run(args)
        log_metrics_all_folds_average( all_folds_id_ls, current_id, experiment)
        
        fold_ex_dt = {0: current_id, 
                            1: all_folds_id_ls[0], 
                            2: all_folds_id_ls[1], 
                            3: all_folds_id_ls[2], 
                            4: all_folds_id_ls[3]}
        
        ensemble_4folds_testing(fold_ex_dt)  
        ensemble_4folds_validation(fold_ex_dt)

        for mode in ['valid', 'test']:
            
        
            parent_dir = '/home/jjia/data/lung_function/lung_function/scripts/results/experiments/'
            label_fpath = parent_dir + str(fold_ex_dt[0]) + f'/{mode}_label.csv'
            pred_fpath = parent_dir + str(fold_ex_dt[0]) + f'/{mode}_pred.csv'
            
            # add icc
            icc_value = icc(label_fpath, pred_fpath, ignore_1st_column=True)
            icc_value_ensemble = {'ensemble_' + k:v  for k, v in icc_value.items()}  # update keys
            print(icc_value_ensemble)
            log_params(icc_value_ensemble)
            
            # add r
            r_p_value = metrics(pred_fpath, label_fpath, ignore_1st_column=True)
            r_p_value_ensemble = {'ensemble_' + k:v  for k, v in r_p_value.items()}  # update keys
            log_params(r_p_value_ensemble)

            # add mae
            mae_dict = mae(pred_fpath, label_fpath, ignore_1st_column=True)
            mae_ensemble = {'ensemble_' + k:v for k, v in mae_dict.items()}
            print(mae_ensemble)
            log_params(mae_ensemble)  
    


# def main2():

#     SEED = 4
#     set_determinism(SEED)  # set seed for this run

#     torch.manual_seed(SEED)
#     torch.cuda.manual_seed_all(SEED)
#     torch.cuda.manual_seed(SEED)

#     random.seed(SEED)
#     np.random.seed(SEED)

#     mlflow.set_tracking_uri("http://nodelogin02:5000")
#     experiment = mlflow.set_experiment("lung_fun_gcn")
    
#     RECORD_FPATH = f"{Path(__file__).absolute().parent}/results/record.log"
#     # write super parameters from set_args.py to record file.
#     id = record_1st(RECORD_FPATH)

#     with mlflow.start_run(run_name=str(id), tags={"mlflow.note.content": args.remark}):
#         args.id = id  # do not need to pass id seperately to the latter function

#         current_id = id
#         tmp_args_dt = vars(args)
#         tmp_args_dt['fold'] = 'all'
#         log_params(tmp_args_dt)

#         all_folds_id_ls = []




#         def run2(trial):
#             args.trial = trial
#             args.epochs = 50
           
#             for fold in [1]:
#                 # write super parameters from set_args.py to record file.
#                 args.gconv_name = 'GCNConv' 
#                 args.gnorm ='BatchNorm'# args.trial.suggest_categorical('gnorm', ['BatchNorm', 'InstanceNorm', 'LayerNorm','GraphNorm',  'DiffGroupNorm'])
#                 args.batch_size = 32
#                 args.heads = 1 # args.trial.suggest_int('GATConv_head', 1, 5)

#                 id = record_1st(RECORD_FPATH)
#                 all_folds_id_ls.append(id)
#                 with mlflow.start_run(run_name=str(id) + '_fold_' + str(fold), tags={"mlflow.note.content": f"fold: {fold}"}, nested=True):
#                     args.fold = fold
#                     args.id = id  # do not need to pass id seperately to the latter function
#                     # args.mode = 'infer'  # disable it for normal training
#                     # args.model_name = args.trial.suggest_categorical('gconv_name', [ 'GIN', 'GCN',  'GAT'])  # ,  
#                     tmp_args_dt = vars(args)
#                     log_params(tmp_args_dt)
#                     loss = run(args)
                    
#             return loss

        
#         storage_name = "sqlite:///optuna.db"
#         study = optuna.create_study(storage=storage_name)
#         study.optimize(run2, n_trials=1)
#         print(study.best_params)  





if __name__ == "__main__":
    main()



