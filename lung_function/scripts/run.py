# -*- coding: utf-8 -*-
# @Time    : 4/5/22 12:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
# log_dict is used to record super parameters and metrics

import sys
sys.path.append("../..")
from mlflow import log_metric, log_param, log_params
import mlflow
import threading
import time
from medutils.medutils import count_parameters
import torch
import torch.nn as nn
import sqlite3
from sqlite3 import Error



from lung_function.modules.datasets import all_loaders
from lung_function.modules.loss import get_loss
from lung_function.modules.networks import get_net_3d
from lung_function.modules.path import PFTPath
from lung_function.modules.set_args import get_args
from lung_function.modules.tool import record_1st, record_artifacts, record_cgpu_info

args = get_args()


def step(mode, net, dataloader, loss_fun, opt, epoch_idx):
    loss_fun_mae = nn.L1Loss()

    device = torch.device("cuda")
    scaler = torch.cuda.amp.GradScaler()
    print(mode + "ing ......")
    if mode == 'train' or mode == 'validaug':
        net.train()
    else:
        net.eval()

    t0 = time.time()
    data_idx = 0
    loss_accu, mae_accu = 0, 0
    for data in dataloader:
        data_idx += 1
        if epoch_idx < 3:  # only show first 3 epochs' data loading time
            t1 = time.time()
            log_metric('TLoad', t1-t0, data_idx+epoch_idx*len(dataloader))

        batch_x = data['image'].to(device)
        batch_y = data['label'].to(device)
        with torch.cuda.amp.autocast():
            if mode != 'train':
                with torch.no_grad():
                    pred = net(batch_x)
            else:
                pred = net(batch_x)

            loss = loss_fun(pred, batch_y)
            mae = loss_fun_mae(pred, batch_y)
        if mode == 'train':  # update gradients only when training
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        print('loss:', loss.item())
        log_metric(mode+'LossBatch', loss.item(), data_idx+epoch_idx*len(dataloader))
        log_metric(mode+'MAEBatch', mae.item(), data_idx+epoch_idx*len(dataloader))
        loss_accu += loss.item()
        mae_accu += mae.item()

        print('pred:', pred.clone().detach().cpu().numpy())
        print('label:', batch_y.clone().detach().cpu().numpy())
        if epoch_idx < 3:
            t2 = time.time()
            log_metric('TUpdateWBatch', t2-t1, data_idx+epoch_idx*len(dataloader))
            t0 = t2  # reset the t0
    log_metric(mode+'LossEpoch', loss_accu/len(dataloader), epoch_idx)
    log_metric(mode+'MAEEpoch', mae_accu/len(dataloader), epoch_idx)


def run(args):
    mypath = PFTPath(id, check_id_dir=False, space=args.ct_sp)
    device = torch.device("cuda")
    outs = 2  # output FVC and FEV1
    net = get_net_3d(name=args.net, nb_cls=outs)
    print('net:', net)

    net_parameters = count_parameters(net)
    net_parameters = str(round(net_parameters / 1000 / 1000, 2))
    log_param('net_parameters_M', net_parameters)

    data_dt = all_loaders(mypath.data_dir, mypath.label_fpath, args)

    net = net.to(device)
    if args.eval_id:
        net.load_state_dict(torch.load(mypath.model_fpath, map_location=device))  # model_fpath need to exist

    loss_fun = get_loss(args.loss)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.epochs = 0 if args.mode == 'infer' else args.epochs
    for i in range(args.epochs):  # 20000 epochs
        if args.mode in ['train', 'continue_train']:
            step('train', net, data_dt['train'], loss_fun, opt, i)
        if i % args.valid_period == 0:  # run the validation
            step('valid', net, data_dt['valid'], loss_fun, opt, i)
            step('test', net, data_dt['test'], loss_fun, opt, i)
    print('Finish all things!')

if __name__ == "__main__":
    database_rui = 'mlrunsdb.db'
    conn = sqlite3.connect(database_rui)
    mlflow.set_tracking_uri("sqlite:///mlrunsdb.db")
    mlflow.set_experiment("lung_fun_db")
    id = record_1st("results/record.log")  # write super parameters from set_args.py to record file.

    with mlflow.start_run(run_name=str(id), tags={"mlflow.note.content": args.remark}):
        p1 = threading.Thread(target=record_cgpu_info, args=(args.outfile,))
        p1.start()
        p2 = threading.Thread(target=record_artifacts, args=(args.outfile,))
        p2.start()

        args.id = id  # do not need to pass id seperately to the latter function
        log_params(vars(args))
        run(args)

        p1.do_run = False  # stop the thread
        p2.do_run = False  # stop the thread
        p1.join()
        p2.join()
    conn.close()  # close the database

