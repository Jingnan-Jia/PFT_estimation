# -*- coding: utf-8 -*-
# @Time    : 4/5/22 12:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
# log_dict is used to record super parameters and metrics

import sys
sys.path.append("/home/jjia/data/lung_function/lung_function/modules/networks/models_pcd")
sys.path.append("/home/jjia/data/lung_function/lung_function/modules")

import random
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

from lung_function.modules import provider
from lung_function.modules.compute_metrics import icc, metrics
from lung_function.modules.datasets import all_loaders
from lung_function.modules.loss import get_loss
from lung_function.modules.networks import get_net_3d
from lung_function.modules.path import PFTPath
from lung_function.modules.set_args import get_args
from lung_function.modules.tool import (record_1st, dec_record_cgpu, retrive_run, try_func, int2str, txtprocess, log_all_metrics, process_dict)
from lung_function.modules.trans import batch_bbox2_3D

args = get_args()
global_lock = threading.Lock()


log_metric = try_func(log_metric)
log_metrics = try_func(log_metrics)


def reinit_fc(net, nb_fc0, fc1_nodes, fc2_nodes, num_classes):
    net.ln1 = nn.Linear(nb_fc0, fc1_nodes)
    net.rl1 = nn.ReLU(inplace=True)
    net.dp1 = nn.Dropout()
    net.ln2 = nn.Linear(fc1_nodes, fc2_nodes)
    net.rl2 = nn.ReLU(inplace=True)
    net.dp2 = nn.Dropout()
    net.ln3 = nn.Linear(fc2_nodes, num_classes)
    return net


    
    
class Run:
    """A class which has its dataloader and step_iteration. It is like Lighting. 
    """

    def __init__(self, args: Namespace, dataloader_flag=True):
        self.args = args
        self.mypath = PFTPath(args.id, check_id_dir=False, space=args.ct_sp)
        self.device = torch.device("cuda")  # 'cuda'
        self.target = [i.lstrip() for i in args.target.split('-')]

        self.pointnet_fc_ls = [int(i) for i in args.pointnet_fc_ls.split('-')]

        self.net = get_net_3d(name=args.net, nb_cls=len(self.target), image_size=args.x_size,
                              pretrained=args.pretrained_imgnet, pointnet_fc_ls=self.pointnet_fc_ls, loss=args.loss,
                              dp_fc1_flag=args.dp_fc1_flag, args=args)  # output FVC and FEV1
        self.fold = args.fold
        self.flops_done = False

        print('net:', self.net)

        net_parameters = count_parameters(self.net)
        net_parameters = str(round(net_parameters / 1e6, 2))
        log_param('net_parameters_M', net_parameters)

        self.loss_fun = get_loss(
            args.loss, mat_diff_loss_scale=args.mat_diff_loss_scale)
        if args.adamw:
            self.opt = torch.optim.AdamW(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        else:
            self.opt = torch.optim.Adam( self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.cosine_decay:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=10, eta_min=0, last_epoch=-1, verbose=False)


        validMAEEpoch_AllBest = 1000
        args.pretrained_id = str(args.pretrained_id)
        if args.pretrained_id != '0':
            if 'SSc' in args.pretrained_id:  # pretrained by ssc_pos L-Net weights
                pretrained_id = args.pretrained_id.split(
                    '-')[self.fold]  # [852] [853] [854] [855]
                pretrained_model_path = f"/home/jjia/data/ssc_scoring/ssc_scoring/results/models_pos/{pretrained_id}/model.pt"
                print(f"pretrained_model_path: {pretrained_model_path}")
                ckpt = torch.load(pretrained_model_path,
                                  map_location=self.device)
                print(f"model is loaded arom {pretrained_model_path}")

                del ckpt['ln3.weight']
                del ckpt['ln3.bias']
                del self.net.ln3  # remove the last layer because they do not match

                # model_fpath need to exist
                self.net.load_state_dict(ckpt, strict=False)
                self.net = reinit_fc(self.net, nb_fc0=8 * 16 * 6 * 6 * 6, fc1_nodes=1024, fc2_nodes=1024,
                                     num_classes=len(self.target))
                # move the new initialized layers to GPU
                self.net = self.net.to(self.device)
                print(f"use the pretrained model from {pretrained_model_path}")

            else:
                if '-' in args.pretrained_id:
                    pretrained_ids = args.pretrained_id.split('-')
                    args.pretrained_id = pretrained_ids[self.fold-1]

                pretrained_path = PFTPath(args.pretrained_id, check_id_dir=False, space=args.ct_sp)
                ckpt = torch.load(pretrained_path.model_fpath, map_location=self.device)
                print(f"model is loaded arom {pretrained_path.model_fpath}")

                if type(ckpt) is dict and 'model' in ckpt:
                    model = ckpt['model']
                    # if 'metric_name' in ckpt:  # not applicable if the pre-trained model is from ModelNet40
                    #     if 'validMAEEpoch_AllBest' == ckpt['metric_name']:
                    #         validMAEEpoch_AllBest = ckpt['current_metric_value']
                    client = mlflow.MlflowClient()
                    experiment = mlflow.get_experiment_by_name("lung_fun_db15")
                    pre_run =  client.search_runs(experiment_ids=[experiment.experiment_id], filter_string=f"params.id = '{str(args.pretrained_id)}'")[0]
                    if ('dataset' in pre_run.data.params) and (pre_run.data.params['dataset'] in ['modelnet40']):   # pre-trained by an classification dataset
                        assert pre_run.data.params['net'] == self.args.net
                        if 'pointmlp_reg' == self.args.net:
                            model = {k:v for k,v in model.items() if 'classifier' != k.split('.')[0]}
                        elif 'pointnet2_reg' == self.args.net:
                            # model = {k:v for k,v in model.items() if 'fc1' in k }
                            excluded_keys = ['fc1', 'bn1', 'drop1', 'fc2', 'bn2', 'drop2', 'fc3']  # FC layers
                            model = {key: value for key, value in model.items() if all(excluded_key not in key for excluded_key in excluded_keys)}
                else:
                    model = ckpt
                # model_fpath need to exist
                # strict=false due to the calculation of FLOPs and params. In addition, the pre-trained model may be a 
                # classification model with different output nodes
                self.net.load_state_dict(model, strict=False)  
                # move the new initialized layers to GPU
                self.net = self.net.to(self.device)
        if dataloader_flag:
            self.data_dt = all_loaders(self.mypath.data_dir, self.mypath.label_fpath, args, nb=1000)
        self.net = self.net.to(self.device)

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
        dataloader = self.data_dt[mode]
        if self.args.loss == 'ce':
            loss_fun_mae = nn.CrossEntropyLoss()
        else:
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
        if self.args.loss!='ce':
            mae_accu_ls = [0 for _ in self.target]
        else:
            mae_accu_ls = [0]
        mae_accu_all = 0
        for data in dataloader:
            if args.mode == 'infer' and 8365740 not in data['pat_id']:
                continue
            torch.cuda.empty_cache()  # avoid memory leak
            data_idx += 1
            if epoch_idx < 3:  # only show first 3 epochs' data loading time
                t1 = time.time()
                log_metric('TLoad', t1 - t0, data_idx +
                           epoch_idx * len(dataloader))
            key = args.input_mode

            if args.input_mode in ['vessel_skeleton_pcd', 'lung_mask_pcd']:  # first 3 columns are xyz, last 1 is value
                points = data[key].data.numpy()
                # if points.shape[0] == 1:  # batch size=1, change it to batch size of 2. TODO: Why?!
                #     points = np.concatenate([points, points])
                #     data['label'] = np.concatenate(
                #         [data['label'], data['label']])
                #     data['label'] = torch.tensor(data['label'])

                points = provider.random_point_dropout(points)
                if args.scale_range not in ['0', 0, None, False, 'None']:
                    scale_low, scale_high = args.scale_range.split('-')
                    scale_low, scale_high = float(scale_low), float(scale_high)
                    
                    points[:, :, 0:3] = provider.random_scale_point_cloud(
                        points[:, :, 0:3], scale_low=scale_low, scale_high=scale_high)
                points[:, :, 0:3] = provider.shift_point_cloud(
                    points[:, :, 0:3], shift_range=args.shift_range)
                points = torch.Tensor(points)
                
                if 'pointnext' in args.net:  # data input for pointnext shoudl be split to two parts
                    # 'pos' shape: Batch, N, 3;  'x' shape: Batch, 3+1, N
                    data[key] = {'pos': points[:, :, :3], 'x': points.transpose(2, 1)}
                # else:   # switch dims
                #     data[key] = points.transpose(2, 1)
                
            if args.input_mode in ['vessel_skeleton_pcd', 'lung_mask_pcd']:
                batch_x = data[key]  
            elif args.input_mode == 'modelnet40_pcd':  # ModelNet, ShapeNet
                batch_x = data[0]
            else:
                batch_x = data[key]
            
            if args.input_mode == 'ct_masked_by_lung':
                a = copy.deepcopy(data['lung_mask'])
                a[a > 0] = 1
                batch_x += 1  # shift lowest value from -1 to 0
                batch_x = batch_x * a
                batch_x -= 1
            elif args.input_mode == 'lung_masks':
   
                batch_x = data['lung_mask']
            
            elif args.input_mode == 'ct_masked_by_left_lung':
                a = copy.deepcopy(data['lung_mask'])
                a[a !=2] = 0
                batch_x += 1  # shift lowest value from -1 to 0
                batch_x = batch_x * a
                batch_x -= 1
            elif args.input_mode == 'ct_masked_by_right_lung':
                a = copy.deepcopy(data['lung_mask'])
                a[a !=1] = 0
                batch_x += 1  # shift lowest value from -1 to 0
                batch_x = batch_x * a
                batch_x -= 1
            elif args.input_mode in ('ct_left', 'ct_right', 'ct_upper', 'ct_lower', 'ct_front', 'ct_back'):
                lung_mask = copy.deepcopy(data['lung_mask'])
                lung_mask[lung_mask > 0] = 1
                if 'in_lung' in args.input_mode:  # only keep values in lung
                    batch_x += 1  # shift lowest value from -1 to 0
                    batch_x = batch_x * lung_mask  # masked by lung
                    batch_x -= 1

                z_bottom, z_top, y_bottom, y_top, x_bottom, x_top = batch_bbox2_3D(lung_mask)
                z_mid, y_mid, x_mid = (z_bottom + z_top)//2, (y_bottom + y_top)//2, (x_bottom + x_top)//2
                for idx in range(batch_x.shape[0]):
                    if args.input_mode == 'ct_upper':
                        batch_x[idx, :, :z_mid[idx], :, :] = - 1  # remove bottom
                    elif args.input_mode == 'ct_lower':
                        batch_x[idx, :, z_mid[idx]:, :, :] = - 1  # remove upper
                    elif args.input_mode == 'ct_back':
                        batch_x[idx, :, :, y_mid[idx]:, :] = - 1  # remove front, keep back
                    elif args.input_mode == 'ct_front':
                        batch_x[idx, :, :, :y_mid[idx], :] = - 1  # remove back, keep front
                    elif args.input_mode == 'ct_left':
                        batch_x[idx, :, :, :, :x_mid[idx]] = - 1  # remove right
                    else:  # args.input_mode == 'ct_front':
                        batch_x[idx, :, :, :, x_mid[idx]:] = - 1  # remove left
            else:
                pass
            if 'pointnext' in args.net:  # data input for pointnext shoudl be split to two parts
                batch_x['pos'] = batch_x['pos'].to(self.device)
                batch_x['x'] = batch_x['x'].to(self.device)  # n, z, y, x
            else:
                batch_x = batch_x.to(self.device)  # n, z, y, x
                
            if args.input_mode not in ['modelnet40_pcd']:
                batch_y = data['label'].to(self.device)
            else:  # ModelNet, ShapeNet
                batch_y = data[1].to(self.device)
            

            if 'pcd' == args.input_mode[-3:]:  #TODO: 
                batch_x = batch_x.permute(0, 2, 1) # from b, n, d to b, d, n	
            if args.net == 'mlp_reg' and args.set_all_xyz_to_1 is True:
                batch_x = batch_x[:, -1, :]

            if not self.flops_done:  # only calculate macs and params once
                macs, params = thop.profile(self.net, inputs=(batch_x, ))
                self.flops_done = True
                log_param('macs_G', str(round(macs/1e9, 2)))
                log_param('net_params_M', str(round(params/1e6, 2)))
                
            with torch.cuda.amp.autocast():
                if mode != 'train' or save_pred:  # save pred for inference
                    with torch.no_grad():
                        if args.loss == 'mse_regular':
                            pred, trans_feat = self.net(batch_x)
                        else:
                            pred = self.net(batch_x)
                else:
                    if args.loss == 'mse_regular':
                        pred, trans_feat = self.net(batch_x)
                    else:
                        pred = self.net(batch_x)
                # print('pred',pred )
                if save_pred:
                    head = ['pat_id']
                    head.extend(self.target)

                    batch_pat_id = data['pat_id'].cpu(
                    ).detach().numpy()  # shape (N,1)
                    batch_pat_id = int2str(batch_pat_id)  # shape (N,1)

                    batch_y_np = batch_y.cpu().detach().numpy()  # shape (N, out_nb)
                    pred_np = pred.cpu().detach().numpy()  # shape (N, out_nb)
                    # batch_pat_id = np.expand_dims(batch_pat_id, axis=-1)  # change the shape from (N,) to (N, 1)

                    # shape (1,1)
                    # if args.input_mode == 'vessel_skeleton_pcd' and len(batch_pat_id) == 1:
                    #     batch_pat_id = np.array(
                    #         [[int(batch_pat_id[0])]])
                    #     batch_pat_id = torch.tensor(batch_pat_id)

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

                if args.loss == 'mse_regular':
                    loss = self.loss_fun(pred, batch_y, trans_feat)
                else:
                    if args.loss == 'ce':
                        loss = self.loss_fun(pred, batch_y.to(torch.int64))
                    else:
                        loss = self.loss_fun(pred, batch_y)
                
                with torch.no_grad():
                    if len(batch_y.shape) == 2 and self.args.loss!='ce':
                        mae_ls = [loss_fun_mae(pred[:, i], batch_y[:, i]).item() for i in range(len(self.target))]
                        mae_all = loss_fun_mae(pred, batch_y).item()
                    else:
                        mae_ls = [loss]
                        mae_all = loss.item()

                    

            if mode == 'train' and save_pred is not True:  # update gradients only when training
                self.opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.opt)
                scaler.update()
            loss_cpu = loss.item()
            print('loss:', loss_cpu)
            # log_metric(mode+'LossBatch', loss_cpu, data_idx+epoch_idx*len(dataloader))
            # log_metric(mode+'MAEBatch_All', mae_all, data_idx+epoch_idx*len(dataloader))
            # [log_metric(mode+'MAEBatch_'+t, m, data_idx+epoch_idx*len(dataloader)) for t, m in zip(self.target, mae_ls)]

            loss_accu += loss_cpu
            for i, mae in enumerate(mae_ls):
                mae_accu_ls[i] += mae
            mae_accu_all += mae_all

            # print('pred:', pred.clone().detach().cpu().numpy())
            # print('label:', batch_y.clone().detach().cpu().numpy())
            if epoch_idx < 3:
                t2 = time.time()
                log_metric('TUpdateWBatch', t2-t1, data_idx +
                           epoch_idx*len(dataloader))
                t0 = t2  # reset the t0
        if args.cosine_decay:
            self.scheduler.step() # update the scheduler learning rate

        log_metric(mode+'LossEpoch', loss_accu/len(dataloader), epoch_idx)
        log_metric(mode+'MAEEpoch_All', mae_accu_all / len(dataloader), epoch_idx)
        for t, i in zip(self.target, mae_accu_ls):
            log_metric(mode + 'MAEEpoch_' + t, i / len(dataloader), epoch_idx)

        self.BestMetricDt[mode + 'LossEpochBest'] = min( self.BestMetricDt[mode+'LossEpochBest'], loss_accu/len(dataloader))
        tmp = self.BestMetricDt[mode+'MAEEpoch_AllBest']
        self.BestMetricDt[mode + 'MAEEpoch_AllBest'] = min( self.BestMetricDt[mode+'MAEEpoch_AllBest'], mae_accu_all/len(dataloader))

        log_metric(mode+'LossEpochBest', self.BestMetricDt[mode + 'LossEpochBest'], epoch_idx)
        log_metric(mode+'MAEEpoch_AllBest', self.BestMetricDt[mode + 'MAEEpoch_AllBest'], epoch_idx)

        if self.BestMetricDt[mode+'MAEEpoch_AllBest'] == mae_accu_all/len(dataloader):
            for t, i in zip(self.target, mae_accu_ls):
                log_metric(mode + 'MAEEpoch_' + t + 'Best',
                           i / len(dataloader), epoch_idx)

            if mode == 'valid':
                print(
                    f"Current mae is {self.BestMetricDt[mode+'MAEEpoch_AllBest']}, better than the previous mae: {tmp}, save model.")
                ckpt = {'model': self.net.state_dict(),
                        'metric_name': mode+'MAEEpoch_AllBest',
                        'current_metric_value': self.BestMetricDt[mode+'MAEEpoch_AllBest']}
                torch.save(ckpt, self.mypath.model_fpath)

class RunCombined:
    """A class which has its dataloader and step_iteration. It is like Lighting. 
    """

    def __init__(self, args: Namespace, dataloader_flag=True):
        self.args = args
        
        self.device = torch.device("cuda")  # 'cuda'
        self.target = [i.lstrip() for i in args.target.split('-')]
        self.mypath = PFTPath(args.id, check_id_dir=False, space=args.ct_sp)
        if dataloader_flag:
            self.data_dt_all = all_loaders(self.mypath.data_dir, self.mypath.label_fpath, args, nb=10)
            
        self.pointnet_fc_ls = [int(i) for i in args.pointnet_fc_ls.split('-')]

        self.net = get_net_3d(name=args.net, nb_cls=len(self.target), args=args)  # receive ct and pcd as input
        if args.freeze_encoder:
            for name, para in self.net.named_parameters():
                if 'ct_net_extractor' in name or 'pcd_net_extractor' in name:
                    para.requires_grad = False
            
        self.fold = args.fold
        self.flops_done = False

        print('net:', self.net)

        net_parameters = count_parameters(self.net)
        net_parameters = str(round(net_parameters / 1e6, 2))
        log_param('net_parameters_M', net_parameters)

        self.loss_fun = get_loss(
            args.loss, mat_diff_loss_scale=args.mat_diff_loss_scale)
        self.loss_fun_cosine = nn.CosineSimilarity(dim=2)
        if args.adamw:
            self.opt = torch.optim.AdamW(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        else:
            self.opt = torch.optim.Adam( self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.cosine_decay:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=10, eta_min=0, last_epoch=-1, verbose=False)


        validMAEEpoch_AllBest = 1000
        args.pretrained_id = str(args.pretrained_id)


        self.BestMetricDt = {'trainLossEpochBest': 1000,
                             # 'trainnoaugLossEpochBest': 1000,
                             'validLossEpochBest': 1000,
                             'testLossEpochBest': 1000,

                             'trainMAEEpoch_AllBest': 1000,
                             # 'trainnoaugMAEEpoch_AllBest': 1000,
                             'validMAEEpoch_AllBest': validMAEEpoch_AllBest,
                             'testMAEEpoch_AllBest': 1000,
                             }
        self.net = self.net.to(self.device)

    def step(self, mode, epoch_idx, save_pred=False, suffix=None):
        dataloader_all = self.data_dt_all[mode]
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
        mae_accu_ls = [0]
        mae_accu_all = 0
        for batch_dt_ori in dataloader_all:

            # print(data_ct['pat_id'])
            # print('--------------------------------')
            # print(data_pcd['pat_id'])
            # assert all(data_ct['pat_id'] == data_pcd['pat_id'])
            
            
            # if args.mode == 'infer' and 8365740 not in data['pat_id']:
            #     continue
            torch.cuda.empty_cache()  # avoid memory leak
            data_idx += 1
            # if epoch_idx < 3:  # only show first 3 epochs' data loading time
            #     t1 = time.time()
            #     log_metric('TLoad', t1 - t0, data_idx +
            #                epoch_idx * len(dataloader_ct))
     
            key_ls = args.input_mode.split('-')


            # label
            batch_dt = {}
            batch_y = batch_dt_ori['label'].to(self.device)
            batch_x = []  # ordered by ct, pnn, graph
            for key in key_ls:
                if key not in ['vessel_skeleton_pcd', 'vessel_skeleton_graph']:
                    # data for ct
                    if key == 'ct':
                        batch_ct = batch_dt_ori[key]
                    if key == 'ct_masked_by_lung':
                        a = copy.deepcopy(batch_dt_ori['lung_mask'])
                        a[a > 0] = 1
                        batch_ct += 1  # shift lowest value from -1 to 0
                        batch_ct = batch_ct * a
                        batch_ct -= 1
                    elif key == 'lung_masks':
        
                        batch_ct = batch_dt_ori['lung_mask']
                    
                    elif key == 'ct_masked_by_left_lung':
                        a = copy.deepcopy(batch_dt_ori['lung_mask'])
                        a[a !=2] = 0
                        batch_ct += 1  # shift lowest value from -1 to 0
                        batch_ct = batch_ct * a
                        batch_ct -= 1
                    elif key == 'ct_masked_by_right_lung':
                        a = copy.deepcopy(batch_dt_ori['lung_mask'])
                        a[a !=1] = 0
                        batch_ct += 1  # shift lowest value from -1 to 0
                        batch_ct = batch_ct * a
                        batch_ct -= 1
                    batch_dt[key] = batch_ct
                    batch_x.append(batch_ct)
                elif 'vessel_skeleton_pcd' == key: # data for PCD            
                
                    points = batch_dt_ori[key].data.numpy()
                    points = provider.random_point_dropout(points)
                    if args.scale_range not in ['0', 0, None, False, 'None']:
                        scale_low, scale_high = args.scale_range.split('-')
                        scale_low, scale_high = float(scale_low), float(scale_high)                        
                        points[:, :, 0:3] = provider.random_scale_point_cloud( points[:, :, 0:3], scale_low=scale_low, scale_high=scale_high)
                    points[:, :, 0:3] = provider.shift_point_cloud( points[:, :, 0:3], shift_range=args.shift_range)
                    points = torch.Tensor(points)
                    
                    batch_pcd = batch_pcd.to(self.device)  # n, z, y, x                        
                    batch_pcd = batch_pcd.permute(0, 2, 1) # from b, n, d to b, d, n	

                    if args.net == 'mlp_reg' and args.set_all_xyz_to_1 is True:                        
                        batch_pcd = batch_pcd[:, -1, :]
                    batch_dt[key] = batch_pcd
                    batch_x.append(batch_pcd)

                elif 'vessel_skeleton_graph' == key:
                    batch_graph = batch_dt_ori[key]
                    batch_graph.x = batch_graph.x.to(self.device)
                    batch_graph.edge_index = batch_graph.edge_index.to(self.device)                    
                    batch_graph.batch = batch_graph.batch.to(self.device)
                    batch_graph.y = batch_graph.y.to(self.device)            
                    batch_dt[key] = batch_graph
                    batch_x.append(batch_graph)
                else:
                    raise Exception('wrong key', key)

            if not self.flops_done:  # only calculate macs and params once
                macs, params = thop.profile(self.net, inputs=(*batch_x, ))
                self.flops_done = True
                log_param('macs_G', str(round(macs/1e9, 2)))
                log_param('net_params_M', str(round(params/1e6, 2)))
            


            out_features = True
            tt0 = time.time()
            with torch.cuda.amp.autocast():
                if mode != 'train' or save_pred:  # save pred for inference
                    with torch.no_grad():
                        pred, ct_features, pcd_features = self.net(*batch_x, out_features=out_features)
                else:
                    pred, ct_features, pcd_features = self.net(*batch_x, out_features=out_features)
                tt1 = time.time()
                print(f'time forward: , {tt1-tt0: .2f}')
                # # save features to disk for the future analysis or re-training
                # ct_features_fpath = self.mypath.save_pred_fpath(mode).replace('.csv', '_ct_feature.csv')
                # pcd_features_fpath = self.mypath.save_pred_fpath(mode).replace('.csv', '_pcd_feature.csv')
                
                # batch_pat_id = data_ct['pat_id'].cpu().detach().numpy()
                # batch_labels = batch_y.cpu().detach().numpy()
                # ct_features = ct_features.clone().cpu().numpy()
                # pcd_features= pcd_features.clone().cpu().numpy()
                # ct_features_saved = np.array([batch_pat_id.flatten()[0],*batch_labels.flatten()[:], *ct_features])
                # pcd_features_saved = np.array([batch_pat_id.flatten()[0],*batch_labels.flatten()[:], *pcd_features])
                # head_ct = np.array(['pat_id', 'DLCOc', 'FEV1', 'FVC', 'TLC', *[i for i in range(len(ct_features))]])
                # head_pcd = np.array(['pat_id', 'DLCOc', 'FEV1', 'FVC', 'TLC', *[i for i in range(len(pcd_features))]])
                # medutils.appendrows_to(ct_features_fpath, ct_features_saved, head=head_ct)
                # medutils.appendrows_to(pcd_features_fpath, pcd_features_saved, head=head_pcd)
                
                if save_pred:
                    for k in batch_dt_ori:
                        if 'pat_id' in k:
                            key_pat_id = k    
                            break                        
                   
                    head = ['pat_id']
                    head.extend(self.target)

                    batch_pat_id = batch_dt_ori[key_pat_id].cpu( ).detach().numpy()  # shape (N,1)
                    batch_pat_id = int2str(batch_pat_id)  # shape (N,1)

                    batch_y_np = batch_y.cpu().detach().numpy()  # shape (N, out_nb)
                    pred_np = pred.cpu().detach().numpy()  # shape (N, out_nb)
                    # batch_pat_id = np.expand_dims(batch_pat_id, axis=-1)  # change the shape from (N,) to (N, 1)

                    # shape (1,1)
                    # if args.input_mode == 'vessel_skeleton_pcd' and len(batch_pat_id) == 1:
                    #     batch_pat_id = np.array(
                    #         [[int(batch_pat_id[0])]])
                    #     batch_pat_id = torch.tensor(batch_pat_id)

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

                
                loss = self.loss_fun(pred, batch_y)
                loss_cos = self.loss_fun_cosine(ct_features, pcd_features)[0][0]
                loss = loss + loss_cos
                
                with torch.no_grad():                
                    mae_ls = [loss]
                    mae_all = loss.item()

            if mode == 'train' and save_pred is not True:  # update gradients only when training
                self.opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.opt)
                scaler.update()
            tt2 = time.time()
            print(f'time backward: , {tt2-tt1: .2f}')
            loss_cpu = loss.item()
            print('loss:', loss_cpu)
         
            loss_accu += loss_cpu
            for i, mae in enumerate(mae_ls):
                mae_accu_ls[i] += mae
            mae_accu_all += mae_all

        if args.cosine_decay:
            self.scheduler.step() # update the scheduler learning rate

        log_metric(mode+'LossEpoch', loss_accu/len(dataloader_all), epoch_idx)
        log_metric(mode+'MAEEpoch_All', mae_accu_all / len(dataloader_all), epoch_idx)
        for t, i in zip(self.target, mae_accu_ls):
            log_metric(mode + 'MAEEpoch_' + t, i / len(dataloader_all), epoch_idx)

        self.BestMetricDt[mode + 'LossEpochBest'] = min( self.BestMetricDt[mode+'LossEpochBest'], loss_accu/len(dataloader_all))
        tmp = self.BestMetricDt[mode+'MAEEpoch_AllBest']
        self.BestMetricDt[mode + 'MAEEpoch_AllBest'] = min( self.BestMetricDt[mode+'MAEEpoch_AllBest'], mae_accu_all/len(dataloader_all))

        log_metric(mode+'LossEpochBest', self.BestMetricDt[mode + 'LossEpochBest'], epoch_idx)
        log_metric(mode+'MAEEpoch_AllBest', self.BestMetricDt[mode + 'MAEEpoch_AllBest'], epoch_idx)

        if self.BestMetricDt[mode+'MAEEpoch_AllBest'] == mae_accu_all/len(dataloader_all):
            for t, i in zip(self.target, mae_accu_ls):
                log_metric(mode + 'MAEEpoch_' + t + 'Best',
                           i / len(dataloader_all), epoch_idx)

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
    if '-' in args.net:
        myrun = RunCombined(args)
    else:
        myrun = Run(args)
    modes = ['valid', 'test', 'train'] if args.mode != 'infer' else ['valid', 'test']
    if args.mode == 'infer':
        for mode in ['test']:
            for i in range(1):
                myrun.step(mode,  0,  save_pred=True, suffix=str(i))
    else:  # 'train' or 'continue_train'
        for i in range(args.epochs):  # 20000 epochs
            myrun.step('train', i)
            if i % args.valid_period == 0:  # run the validation
                # mypath = PFTPath(args.id, check_id_dir=False, space=args.ct_sp)
                # if os.path.exists(mypath.save_label_fpath('valid')):
                #     os.remove(mypath.save_label_fpath('valid'))
                # if os.path.exists(mypath.save_label_fpath('test')):
                #     os.remove(mypath.save_label_fpath('test'))
                # if os.path.exists(mypath.save_pred_fpath('valid')):
                #     os.remove(mypath.save_pred_fpath('valid'))
                # if os.path.exists(mypath.save_pred_fpath('test')):
                #     os.remove(mypath.save_pred_fpath('test'))
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
                    myrun.net.load_state_dict(model)
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
            r_p_value = txtprocess(r_p_value)
            log_params(r_p_value)
            print('r_p_value:', r_p_value)

            icc_value = icc(label_fpath, pred_fpath, ignore_1st_column=True)
            log_params(icc_value)
            print('icc:', icc_value)
            if os.path.exists(os.path.dirname(pred_fpath) + '/valid_scatter.png'):
                os.rename(os.path.dirname(pred_fpath) + '/valid_scatter.png', os.path.dirname(pred_fpath) + f'/valid_scatter_{i}.png')
                
            if os.path.exists(os.path.dirname(pred_fpath) + '/test_scatter.png'):
                os.rename(os.path.dirname(pred_fpath) + '/test_scatter.png', os.path.dirname(pred_fpath) + f'/test_scatter_{i}.png')
                
    print('Finish all things!')

   
        
def main():
    SEED = 4
    set_determinism(SEED)  # set seed for this run

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.manual_seed(SEED)

    random.seed(SEED)
    np.random.seed(SEED)

    mlflow.set_tracking_uri("http://nodelogin01:5000")
    if args.net in ['pointnet_reg','pointnet2_reg', 'pointnext', 'pointmlp_reg', 'mlp_reg']:
        ex_name = 'pft_pnn'
    elif args.net in ['gnn']:
        ex_name = 'pft_gnn'
    elif '-' in args.net:
        ex_name = 'pft_combined'
    else:
        ex_name = 'lung_fun_db15'
    experiment = mlflow.set_experiment(ex_name)
    
    RECORD_FPATH = f"{Path(__file__).absolute().parent}/results/record.log"
    # write super parameters from set_args.py to record file.
    id = record_1st(RECORD_FPATH)

    # if merge 4 fold results, uncommit the following code.
    # From here ======================================================
    # current_id = 427
    # id_ls = [428, 431, 433, 435]
    # client = MlflowClient()
    # run_ls = client.search_runs(experiment_ids=[experiment.experiment_id],
    #                             filter_string=f"params.id LIKE '%{current_id}%'")
    # run_ = run_ls[0]
    # run_id = run_.info.run_id
    # with mlflow.start_run(run_id=run_id, tags={"mlflow.note.content": args.remark}):
    #     args.id = id  # do not need to pass id seperately to the latter function

    # to here =======================================================
    # if args.mode == 'infer':  # get the id of the run
    #     client = MlflowClient()
    #     run_ls = client.search_runs(experiment_ids=[experiment.experiment_id], filter_string=f"params.id = '{args.pretrained_id}'")
    #     run_ = run_ls[0]
    #     run_id = run_.info.run_id
    #     with mlflow.start_run(run_id=run_id, tags={"mlflow.note.content": args.remark}):
    #         args.id = args.pretrained_id  # log the metrics to the pretrained_id
    #         run(args)
    # else:
    with mlflow.start_run(run_name=str(id), tags={"mlflow.note.content": args.remark}):
        args.id = id  # do not need to pass id seperately to the latter function

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
                args.pretrained_id = get_args().pretrained_id
                args.id = id  # do not need to pass id seperately to the latter function
                # args.mode = 'infer'  # disable it for normal training
                tmp_args_dt = vars(args)
                log_params(tmp_args_dt)
                run(args)
                
    log_all_metrics(all_folds_id_ls, current_id, experiment)

if __name__ == "__main__":
    main()
