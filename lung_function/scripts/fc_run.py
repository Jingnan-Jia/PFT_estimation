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
from sklearn.preprocessing import StandardScaler


from lung_function.modules import provider
from lung_function.modules.compute_metrics import icc, metrics
from lung_function.modules.datasets import all_loaders
from lung_function.modules.loss import get_loss
from lung_function.modules.networks import get_net_3d
from lung_function.modules.path import PFTPath
from lung_function.modules.set_args import get_args
from lung_function.modules.tool import record_1st, dec_record_cgpu, retrive_run
from lung_function.modules.trans import batch_bbox2_3D

args = get_args()
global_lock = threading.Lock()


def thread_safe(func):
    def thread_safe_fun(*args, **kwargs):
        with global_lock:
            print('get lock by main thread')
            func(*args, **kwargs)
            print('release lock by main thread')
    return thread_safe_fun


def try_func(func):
    def _try_fun(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as err:
            print(err, file=sys.stderr)
            pass
    return _try_fun


log_metric = try_func(log_metric)
log_metrics = try_func(log_metrics)







class FCNet(nn.Module):
    def __init__(self, in_chn, out_chn):
        super().__init__()

        self.nb_feature = in_chn

        self.fc1 = nn.Linear(self.nb_feature, 512)
        self.bn1 = nn.InstanceNorm1d(512) 
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.InstanceNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, out_chn)
        
        

    def forward(self, x):  # x shape: (B, n)

        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        out = self.fc3(x)
        return out
    
    
    
def int2str(batch_id: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        batch_id (np.ndarray): _description_

    Raises:
        Exception: _description_

    Returns:
        np.ndarray: _description_
    """
    tmp = batch_id.shape
    id_str_ls = []
    for id in batch_id:
        if isinstance(id, np.ndarray):
            id = id[0]
        id = str(id)
        while len(id) < 7:  # the pat id should be 7 digits
            id = '0' + id
        if len(tmp) == 2:
            id_str_ls.append([id])
        elif len(tmp) == 1:
            id_str_ls.append(id)
        else:
            raise Exception(
                f"the shape of batch_id is {tmp}, but it should be 1-dim or 2-dim")

    return np.array(id_str_ls)

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, fpath_ct, fpath_pcd, in_chn):
        self.device = torch.device("cuda")  # 'cuda'
        data_ct = pd.read_csv(fpath_ct)
        data_pcd = pd.read_csv(fpath_pcd)
        
        assert (data_ct['pat_id']==data_pcd['pat_id']).all().all()
        assert (data_ct.iloc[:, 1:5]==data_pcd.iloc[:, 1:5]).all().all()
        assert len(data_ct) == len(data_pcd)
        
        if in_chn == 192:  # ct
            self.data = data_ct.astype('float32')
        elif in_chn == 1024:  # pcd
            self.data = data_pcd.astype('float32')
        else:
            scaler = StandardScaler()
            data_ct_norm = pd.DataFrame(scaler.fit_transform(data_ct), columns=data_ct.columns)
            data_pcd_norm = pd.DataFrame(scaler.fit_transform(data_pcd), columns=data_pcd.columns)
            self.data = pd.concat([data_ct_norm, data_pcd_norm.iloc[:, 5:]], axis=1).astype('float32')

        
        
        self.features = torch.tensor(self.data.iloc[:, 5:].to_numpy()).to(self.device)
        self.labels = torch.tensor(self.data.iloc[:, 1:5].to_numpy()).to(self.device)
        self.pat_ids = torch.tensor(self.data.iloc[:, :1].to_numpy()).to(self.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.features[idx], self.labels[idx], self.pat_ids[idx]
    
    
    
class Run:
    """A class which has its dataloader and step_iteration. It is like Lighting. 
    """

    def __init__(self, args: Namespace, dataloader_flag=True):
        self.args = args
        self.device = torch.device("cuda")  # 'cuda'
        self.target = [i.lstrip() for i in args.target.split('-')]
        if args.net == 'fcnet_192':
            in_chn = 192
        elif args.net == 'fcnet_1024':
            in_chn = 1024
        elif args.net == 'fcnet_1216':
            in_chn = 1216
        else:
            raise Exception(f'undefined network {args.net}')
        self.net = FCNet(in_chn=in_chn, out_chn=len(self.target))  # receive ct and pcd as input
            
        self.fold = args.fold
        self.flops_done = False
        self.mypath = PFTPath(args.id, check_id_dir=False, space=args.ct_sp)

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

        self.net = self.net.to(self.device)

        validMAEEpoch_AllBest = 1000
        
        ex_dir = '/home/jjia/data/lung_function/lung_function/scripts/results/experiments'
        data_id = [3387, 3391, 3392, 3439]
        
        self.dataloader = {}
        for mode in ['train', 'valid', 'test']:
            shuffle = True if mode=='train' else False
            ct_path = f"{ex_dir}/{data_id[self.fold]}/{mode}_pred_ct_feature.csv" 
            pcd_path = f"{ex_dir}/{data_id[self.fold]}/{mode}_pred_pcd_feature.csv" 

            data = FeatureDataset(ct_path, pcd_path, in_chn)
            self.dataloader[mode] = DataLoader(data, batch_size=args.batch_size, shuffle=shuffle)
        

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
        for fea, lab, patid in self.dataloader[mode]:
            

            torch.cuda.empty_cache()  # avoid memory leak
            data_idx += 1
            if epoch_idx < 3:  # only show first 3 epochs' data loading time
                t1 = time.time()
                log_metric('TLoad', t1 - t0, data_idx + epoch_idx * len_data)
     

            batch_x = fea
            batch_y = lab

            if not self.flops_done:  # only calculate macs and params once
                macs, params = thop.profile(self.net, inputs=(batch_x, ))
                self.flops_done = True
                log_param('macs_G', str(round(macs/1e9, 2)))
                log_param('net_params_M', str(round(params/1e6, 2)))
            


            with torch.cuda.amp.autocast():
                if mode != 'train' or save_pred:  # save pred for inference
                    with torch.no_grad():
                        pred = self.net(batch_x)
                else:
                    pred = self.net(batch_x)
                
                if save_pred:
                    head = ['pat_id']
                    head.extend(self.target)

                    batch_pat_id = patid.cpu(
                    ).detach().numpy()  # shape (N,1)
                    batch_pat_id = int2str(batch_pat_id)  # shape (N,1)

                    batch_y_np = batch_y.cpu().detach().numpy()  # shape (N, out_nb)
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
                log_metric('TUpdateWBatch', t2-t1, data_idx + epoch_idx*len_data)
                t0 = t2  # reset the t0
        if args.cosine_decay:
            self.scheduler.step() # update the scheduler learning rate

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
        log_params(r_p_value)
        print('r_p_value:', r_p_value)

        icc_value = icc(label_fpath, pred_fpath, ignore_1st_column=True)
        log_params(icc_value)
        print('icc:', icc_value)

    print('Finish all things!')


def average_all_folds(id_ls: Sequence[int], current_id: int, experiment, key='params'):
    """
    Average the logs form mlflow for all folds.
    """
    current_run = retrive_run(experiment=experiment, reload_id=current_id)

    all_dt = {}
    for id in id_ls:
        mlflow_run = retrive_run(experiment=experiment, reload_id=id)
        if key == 'params':
            target_dt = mlflow_run.data.params
            current_dt = current_run.data.params
        elif key == 'metrics':
            target_dt = mlflow_run.data.metrics
            current_dt = current_run.data.metrics
        else:
            raise Exception(
                f"Expected key of 'params' or 'metrics', but got key: {key}")

        for k, v in target_dt.items():
            if k not in current_dt:  # re-writing parameters in mlflow is not allowed
                if k not in all_dt:
                    all_dt[k] = []
                # this is a value, not a list (see bellow)
                if not isinstance(all_dt[k], list):
                    continue
                try:
                    all_dt[k].append(float(v))
                except Exception:
                    # can not be converted to numbers which can not be averaged
                    all_dt[k] = v

    all_dt = {k: statistics.mean(v) if isinstance(
        v, list) else v for k, v in all_dt.items()}

    return all_dt


def log_metrics_all_folds_average(id_ls: list, id: int, experiment):
    """
    Get the 4 folds metrics and parameters
    Average them
    Log average values to the parent mlflow
    """
    # average parameters
    param_dt = average_all_folds(id_ls, id, experiment, key='params')
    if len(param_dt) < 100:
        log_params(param_dt)

    elif len(param_dt) >= 100 and len(param_dt) < 200:
        dt_1 = {k: param_dt[k] for i, k in enumerate(param_dt) if i < 100}
        dt_2 = {k: param_dt[k] for i, k in enumerate(param_dt) if i >= 100}
        log_params(dt_1)
        log_params(dt_2)
    else:
        raise Exception(
            f"Our logging request can contain at most 200 params. Got {len(param_dt)} params")

    # average metrics
    metric_dt = average_all_folds(id_ls, id, experiment, key='metrics')
    log_metrics(metric_dt, 0)
 


def ensemble_4folds_testing(fold_ex_dt):
    parent_dir = '/home/jjia/data/lung_function/lung_function/scripts/results/experiments/'

    dir0 = parent_dir + str(fold_ex_dt[0])
    ave_fpath = dir0  + '/test_pred.csv'
    label_fpath = dir0  + '/test_label.csv'

    output_file_path = Path(ave_fpath)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_ls = []
    for i in [1,2,3,4]:
        data_fpath_ls = glob(parent_dir + str(fold_ex_dt[i]) + '/test_pred*.csv')
        for data_fpath in data_fpath_ls:
            df = pd.read_csv(data_fpath,index_col=0)
            df_ls.append(df)
            
    df_ave = sum(df_ls)/len(df_ls)
    df_ave.to_csv(ave_fpath)
    print(ave_fpath)
    
    label_fpath_fold1 = parent_dir + str(fold_ex_dt[i]) + '/test_label.csv'
    df_label = pd.read_csv(label_fpath_fold1,index_col=0)
    df_label.to_csv(label_fpath)
    
        
def ensemble_4folds_validation(fold_ex_dt_ls):
    parent_dir = '/home/jjia/data/lung_function/lung_function/scripts/results/experiments/'
    if type(fold_ex_dt_ls) is not list:
        fold_ex_dt_ls = [fold_ex_dt_ls]
    for fold_ex_dt in fold_ex_dt_ls:
        dir0 = parent_dir + str(fold_ex_dt[0])
        pred_all_fpath = dir0  + '/valid_pred.csv'
        label_all_fpath = dir0  + '/valid_label.csv'
        output_file_path = Path(pred_all_fpath)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        df_pred_ls, df_label_ls = [], []
        for i in [1,2,3,4]:
            
            df_ls = []
            data_fpath_ls = glob(parent_dir + str(fold_ex_dt[i]) + '/valid_pred*.csv')
            for data_fpath in data_fpath_ls:
                df = pd.read_csv(data_fpath,index_col=0)
                df_ls.append(df)
            df_pred = sum(df_ls)/len(df_ls)
    
            label_fpath = parent_dir + str(fold_ex_dt[i]) + '/valid_label.csv'
            df_label = pd.read_csv(label_fpath,index_col=0)

            df_pred_ls.append(df_pred)
            df_label_ls.append(df_label)
        df_pred_valid = pd.concat(df_pred_ls)
        df_label_valid = pd.concat(df_label_ls)
        
        df_pred_valid.to_csv(pred_all_fpath)
        df_label_valid.to_csv(label_all_fpath)
        print(pred_all_fpath)

def mae(pred_fpath, label_fpath, ignore_1st_column=True):
    mae_dict = {}

    label = pd.read_csv(label_fpath)
    pred = pd.read_csv(pred_fpath)
    if ignore_1st_column:
        pred = pred.iloc[: , 1:]
        label = label.iloc[: , 1:]
    if 'ID' == label.columns[0]:
        del label["ID"]
    if 'ID' == pred.columns[0]:
        del pred["ID"]

    original_columns = label.columns

    # ori_columns = list(label.columns)

    for column in original_columns:
        abs_err = (pred[column] - label[column]).abs()
        mae_value = abs_err.mean().round(2)
        std_value = abs_err.std().round(2)
        
        prefix = label_fpath.split("/")[-1].split("_")[0]
        mae_dict['mae_' + prefix + '_' + column] = mae_value
        mae_dict['mae_std_' + prefix + '_' + column] = std_value

    return mae_dict

def me(pred_fpath, label_fpath, ignore_1st_column=True):
    mae_dict = {}

    label = pd.read_csv(label_fpath)
    pred = pd.read_csv(pred_fpath)
    if ignore_1st_column:
        pred = pred.iloc[: , 1:]
        label = label.iloc[: , 1:]
    if 'ID' == label.columns[0]:
        del label["ID"]
    if 'ID' == pred.columns[0]:
        del pred["ID"]

    original_columns = label.columns

    for column in original_columns:
        abs_err = (pred[column] - label[column])
        mae_value = abs_err.mean().round(2)
        std_value = abs_err.std().round(2)
        
        prefix = label_fpath.split("/")[-1].split("_")[0]
        mae_dict['me_' + prefix + '_' + column] = mae_value
        mae_dict['me_std_' + prefix + '_' + column] = std_value

    return mae_dict

def mre(pred_fpath, label_fpath, ignore_1st_column=True):
    label = pd.read_csv(label_fpath)
    pred = pd.read_csv(pred_fpath)
    
    if ignore_1st_column:
        pred = pred.iloc[: , 1:]
        label = label.iloc[: , 1:]

    rel_err_dict = {}
    for column in label.columns:
        mae_value = (pred[column] - label[column]).abs()
        rel_err = mae_value / label[column]
        # print(f'relative error for {column}:')
        # for i in rel_err:
        #     if i > 2:
        #         print(i)
        mean_rel_err = rel_err.mean().round(2)
        mean_rel_err_std = rel_err.std().round(2)
        prefix = label_fpath.split("/")[-1].split("_")[0]
        rel_err_dict['mre_' + prefix + '_' + column] = mean_rel_err
        rel_err_dict['mre_std_' + prefix + '_' + column] = mean_rel_err_std
       
    return rel_err_dict

        
def main():
   
    SEED = 4
    set_determinism(SEED)  # set seed for this run

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.manual_seed(SEED)

    random.seed(SEED)
    np.random.seed(SEED)

    mlflow.set_tracking_uri("http://nodelogin02:5000")
    experiment = mlflow.set_experiment("lung_fun_db15")
    
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




if __name__ == "__main__":
    main()
