import mlflow
from lung_function.modules.compute_metrics import icc, metrics
from mlflow import log_params
import pandas as pd
from pathlib import Path
import numpy as np 
import pandas as pd
from mlflow import MlflowClient

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


parent_ids = [1303, 1308, 1458, 1460, 1503, 1513, 1516, 1526, 1538,1560,1577,1579,1585,1608,1610,1612,1614,1616,1633,1635,1637,1639,1641,1648,1650,1652,1654,1656,1676,1678,1680,1682,1684,1687,1710,1714,1725,1753,1755,1757,1786,1788,1790,1798,1800,1802,1812,1816,1840,1842,1844,1846,1848,1867,1865,1869,1882,1884,1896,1898,1908,1910,1918,1927,1930,1932,1934,2203,2228,2321,2323,2325,2327,2329,2335,2339,2331,2353,2362,2364,2366,2368,2370,2372,2408,2410,2432,2440,2450,2456,2460,2462,2464,2466,2468,2484,2486,2488]
def child_run_id(parent_id: str, experiment, client):
    
    parent_run = client.search_runs(experiment_ids=[experiment.experiment_id], filter_string=f"params.id = '{str(parent_id)}'")[0]
    runss = client.search_runs(experiment_ids=[experiment.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'")  # 4 days after after the parent ID created 
    
    if len(runss) == 4:  # all folds are there
        out = {int(ru.data.params['fold']): int(ru.data.params['id']) for ru in runss}
        out[0 ] = parent_id
        print(f"successfully found four folds runs for {parent_id}: {out}")
        return out
    else:
        tmp = {int(ru.data.params['fold']): int(ru.data.params['id']) for ru in runss}
        print(f"the searched runs for {parent_id} do not 4 folds, only has {len(runss)} folds: {tmp}")
        return None
    
     
    
mlflow.set_tracking_uri("http://nodelogin02:5000")
experiment = mlflow.set_experiment("lung_fun_db15")
client = MlflowClient()

fold_ex_dt_ls = [ ]
for parent_id in parent_ids[::-1]:
    dt = child_run_id(parent_id, experiment, client)
    if dt:
        fold_ex_dt_ls.append(dt)


# Create a run under the default experiment (whose id is '0').
parent_dir = '/home/jjia/data/lung_function/lung_function/scripts/results/experiments/'

for i in fold_ex_dt_ls:
    print(i)
    experiment_id = i[0]
    run_ = client.search_runs(experiment_ids=[experiment.experiment_id], filter_string=f"params.id = '{str(experiment_id)}'")[0]
    run_id = run_.info.run_id

    with mlflow.start_run(run_id=run_id):
        for mode in ['valid', 'test']:
            print(mode)
            label_fpath = f"{parent_dir}{i[0]}/{mode}_label.csv"
            pred_fpath = f"{parent_dir}{i[0]}/{mode}_pred.csv"

            # #add icc
            icc_value = icc(label_fpath, pred_fpath, ignore_1st_column=True)
            icc_value_ensemble = {'ensemble_' + k:v  for k, v in icc_value.items()}  # update keys
            print(icc_value_ensemble)
            try:
                log_params(icc_value_ensemble)
            except:
                pass

            # add r
            r_p_value = metrics(pred_fpath, label_fpath, ignore_1st_column=True)
            r_p_value_ensemble = {'ensemble_' + k:v  for k, v in r_p_value.items()}  # update keys
            print(r_p_value_ensemble)
            try:
                log_params(r_p_value_ensemble)
            except:
                pass
            
            # add mae
            mae_dict = mae(pred_fpath, label_fpath, ignore_1st_column=True)
            mae_ensemble = {'ensemble_' + k:v for k, v in mae_dict.items()}
            print(mae_ensemble)
            try:
                log_params(mae_ensemble)
            except:
                pass
            
            # add me
            me_dict = me(pred_fpath, label_fpath, ignore_1st_column=True)
            me_ensemble = {'ensemble_' + k:v for k, v in me_dict.items()}
            print(me_ensemble)
            try:
                log_params(me_ensemble)
            except:
                pass   
            
            # add mre
            mre_dict = mre(pred_fpath, label_fpath, ignore_1st_column=True)
            mre_ensemble = {'ensemble_' + k:v for k, v in mre_dict.items()}
            print(mre_ensemble)
            try:
                log_params(mre_ensemble)
            except:
                pass   