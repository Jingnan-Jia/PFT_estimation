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

from lung_function.modules import provider
from lung_function.modules.compute_metrics import icc, metrics
from lung_function.modules.datasets import all_loaders
from lung_function.modules.loss import get_loss
from lung_function.modules.networks import get_net_3d
from lung_function.modules.path import PFTPath
from lung_function.modules.set_args import get_args
from lung_function.modules.tool import (record_1st, dec_record_cgpu, retrive_run, try_func, int2str, mae, me, mre, txtprocess, ensemble_4folds_validation, ensemble_4folds_testing, 
                                        log_metrics_all_folds_average, average_all_folds)
from lung_function.modules.trans import batch_bbox2_3D

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

        self.net = self.net.to(self.device)

        validMAEEpoch_AllBest = 1000
        args.pretrained_id = str(args.pretrained_id)
        
        if dataloader_flag:
            ori_input_mode = args.input_mode
            args.input_mode = ori_input_mode.split('-')[0]
            args.ct_sp = '1.5'
            self.mypath = PFTPath(args.id, check_id_dir=False, space=args.ct_sp)
            self.data_dt_ct = all_loaders(self.mypath.data_dir, self.mypath.label_fpath, args, nb=2000)
            
            args.input_mode = ori_input_mode.split('-')[1]
            args.ct_sp = 'ori'
            self.mypath = PFTPath(args.id, check_id_dir=False, space=args.ct_sp)
            self.data_dt_pcd = all_loaders(self.mypath.data_dir, self.mypath.label_fpath, args, nb=2000)
            args.input_mode = ori_input_mode

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
        dataloader_pcd = self.data_dt_pcd[mode]
        dataloader_ct = self.data_dt_ct[mode]

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
        data_iter_ct = iter(dataloader_ct)
        data_iter_pcd = iter(dataloader_pcd)
        for i in range(len(dataloader_ct)):
            ttt = time.time()
            data_ct = next(data_iter_ct)
            ttt2 = time.time()
            print(f'get a ct in {ttt2-ttt}')
            data_pcd = next(data_iter_pcd)
            print(f'get a pcd in {time.time() - ttt2}')
        # for data_ct, data_pcd in zip(dataloader_ct, dataloader_pcd):
            assert all(data_ct['pat_id'] == data_pcd['pat_id'])
            
            
            # if args.mode == 'infer' and 8365740 not in data['pat_id']:
            #     continue
            torch.cuda.empty_cache()  # avoid memory leak
            data_idx += 1
            if epoch_idx < 3:  # only show first 3 epochs' data loading time
                t1 = time.time()
                log_metric('TLoad', t1 - t0, data_idx +
                           epoch_idx * len(dataloader_ct))
     
            key_ct, key_pcd = args.input_mode.split('-')


            # label
            batch_y = data_ct['label'].to(self.device)
            
            # data for PCD
            points = data_pcd[key_pcd].data.numpy()
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
                data_pcd[key_pcd] = {'pos': points[:, :, :3], 'x': points.transpose(2, 1)}
            # else:   # switch dims
            #     data[key] = points.transpose(2, 1)
            
            batch_pcd = data_pcd[key_pcd]  # n, c, z, y, x

            if 'pointnext' in args.net:  # data input for pointnext shoudl be split to two parts
                batch_pcd['pos'] = batch_pcd['pos'].to(self.device)
                batch_pcd['x'] = batch_pcd['x'].to(self.device)  # n, z, y, x
            else:
                batch_pcd = batch_pcd.to(self.device)  # n, z, y, x
                
            if 'pcd' == args.input_mode[-3:]:  #TODO: 
                batch_pcd = batch_pcd.permute(0, 2, 1) # from b, n, d to b, d, n	

            if args.net == 'mlp_reg' and args.set_all_xyz_to_1 is True:
                batch_pcd = batch_pcd[:, -1, :]

            # data for ct
            if key_ct == 'ct':
                batch_ct = data_ct[key_ct]
            if key_ct == 'ct_masked_by_lung':
                a = copy.deepcopy(data_ct['lung_mask'])
                a[a > 0] = 1
                batch_ct += 1  # shift lowest value from -1 to 0
                batch_ct = batch_ct * a
                batch_ct -= 1
            elif key_ct == 'lung_masks':
   
                batch_ct = data_ct['lung_mask']
            
            elif key_ct == 'ct_masked_by_left_lung':
                a = copy.deepcopy(data_ct['lung_mask'])
                a[a !=2] = 0
                batch_ct += 1  # shift lowest value from -1 to 0
                batch_ct = batch_ct * a
                batch_ct -= 1
            elif key_ct == 'ct_masked_by_right_lung':
                a = copy.deepcopy(data_ct['lung_mask'])
                a[a !=1] = 0
                batch_ct += 1  # shift lowest value from -1 to 0
                batch_ct = batch_ct * a
                batch_ct -= 1
            
            batch_ct = batch_ct.to(self.device)
            
            batch_x = (batch_ct, batch_pcd)

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
                    head = ['pat_id']
                    head.extend(self.target)

                    batch_pat_id = data_ct['pat_id'].cpu(
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

                if args.loss == 'ce':
                    loss = self.loss_fun(pred, batch_y.to(torch.int64))
                else:
                    loss = self.loss_fun(pred, batch_y)
                    loss_cos = self.loss_fun_cosine(ct_features, pcd_features)[0][0]
                    loss = loss + loss_cos
                
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
            tt2 = time.time()
            print(f'time backward: , {tt2-tt1: .2f}')
            loss_cpu = loss.item()
            print('loss:', loss_cpu)
            # log_metric(mode+'LossBatch', loss_cpu, data_idx+epoch_idx*len(dataloader_ct))
            # log_metric(mode+'MAEBatch_All', mae_all, data_idx+epoch_idx*len(dataloader_ct))
            # [log_metric(mode+'MAEBatch_'+t, m, data_idx+epoch_idx*len(dataloader_ct)) for t, m in zip(self.target, mae_ls)]

            loss_accu += loss_cpu
            for i, mae in enumerate(mae_ls):
                mae_accu_ls[i] += mae
            mae_accu_all += mae_all

            # print('pred:', pred.clone().detach().cpu().numpy())
            # print('label:', batch_y.clone().detach().cpu().numpy())
            if epoch_idx < 3:
                t2 = time.time()
                log_metric('TUpdateWBatch', t2-t1, data_idx +
                           epoch_idx*len(dataloader_ct))
                t0 = t2  # reset the t0
        if args.cosine_decay:
            self.scheduler.step() # update the scheduler learning rate

        log_metric(mode+'LossEpoch', loss_accu/len(dataloader_ct), epoch_idx)
        log_metric(mode+'MAEEpoch_All', mae_accu_all / len(dataloader_ct), epoch_idx)
        for t, i in zip(self.target, mae_accu_ls):
            log_metric(mode + 'MAEEpoch_' + t, i / len(dataloader_ct), epoch_idx)

        self.BestMetricDt[mode + 'LossEpochBest'] = min( self.BestMetricDt[mode+'LossEpochBest'], loss_accu/len(dataloader_ct))
        tmp = self.BestMetricDt[mode+'MAEEpoch_AllBest']
        self.BestMetricDt[mode + 'MAEEpoch_AllBest'] = min( self.BestMetricDt[mode+'MAEEpoch_AllBest'], mae_accu_all/len(dataloader_ct))

        log_metric(mode+'LossEpochBest', self.BestMetricDt[mode + 'LossEpochBest'], epoch_idx)
        log_metric(mode+'MAEEpoch_AllBest', self.BestMetricDt[mode + 'MAEEpoch_AllBest'], epoch_idx)

        if self.BestMetricDt[mode+'MAEEpoch_AllBest'] == mae_accu_all/len(dataloader_ct):
            for t, i in zip(self.target, mae_accu_ls):
                log_metric(mode + 'MAEEpoch_' + t + 'Best',
                           i / len(dataloader_ct), epoch_idx)

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
    modes = ['train', 'valid', 'test'] if args.mode != 'infer' else ['valid', 'test']
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

    print('Finish all things!')

        
def main():
    args.net = "x3d_m-pointnet2_reg"
    
    SEED = 4
    set_determinism(SEED)  # set seed for this run

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.manual_seed(SEED)

    random.seed(SEED)
    np.random.seed(SEED)

    mlflow.set_tracking_uri("http://nodelogin01:5000")
    experiment = mlflow.set_experiment("pft_combined")
    
    RECORD_FPATH = f"{Path(__file__).absolute().parent}/results/record.log"
    # write super parameters from set_args.py to record file.
    id = record_1st(RECORD_FPATH)

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
        log_metrics_all_folds_average(
            all_folds_id_ls, current_id, experiment)
        
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
            r_p_value_ensemble = txtprocess(r_p_value_ensemble)
            log_params(r_p_value_ensemble)

            # add mae
            mae_dict = mae(pred_fpath, label_fpath, ignore_1st_column=True)
            mae_ensemble = {'ensemble_' + k:v for k, v in mae_dict.items()}
            print(mae_ensemble)
            log_params(mae_ensemble)    




if __name__ == "__main__":
    main()
