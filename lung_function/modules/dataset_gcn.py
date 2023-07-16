import json
import collections
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pickle
import os
from torch_geometric.data import data
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Dataset, download_url
from torch_geometric.loader import DataLoader
from lung_function.modules.datasets import clean_data, pat_from_json
from tqdm import tqdm
import networkx as nx 
from multiprocessing import Pool, Queue, Process, Value, Array, Manager, Lock


def build_graph(xyzr_dt):
    graph = nx.Graph()
    data_key = 'data'
    xyzr = xyzr_dt[data_key]
    xyzr_mm = xyzr_dt['data_mm']
    # 添加节点
    for i in range(xyzr_mm.shape[0]):
        node = tuple(xyzr_mm[i])  # 使用xyzr作为节点的唯一标识
        graph.add_node(node)

    
    # 添加边
    for i in tqdm(range(xyzr.shape[0])):
        current_node = xyzr[i]
        # for j in range(i + 1, xyzr.shape[0]):
        #     neighbor_node = xyzr[j, :]
        dis = np.abs(current_node[:3] - xyzr[:, :3])
        dis[i] = 100
        condition = np.all(dis <= 1, axis=1)
        indices = np.where(condition)[0]
        assert len(indices) < 27
        for j in indices:  # 3*3*3 neighbors
            graph.add_edge(i, j)
    # # 获取edge_index
    # edge_index = np.array(list(graph.edges())).T

    for k, v in xyzr_dt.items():
        if k not in ['data']:
            graph.graph[k] = v
    
    return graph


    

def getdata(dt,  target = 'DLCOc_SB-FEV1-FVC-TLC_He'):
    print('start ...')
        
    xyzr_dt = pd.read_pickle(dt['fpath'])
    # remove the interpolated points, kep the original neighbors
    data_key = 'data'
    # xyzr_dt[data_key] = xyzr_dt[data_key][:1000]  # TODO: REMOE it before running
    xyzr_dt[data_key] = xyzr_dt[data_key][:140000]

    tmp_ls = []
    for row in xyzr_dt[data_key]:  # check three coordinates are integers
        if not row[0]%1 and not row[1]%1 and not row[2]%1:  
            tmp_ls.append(row)
    xyzr_dt[data_key] = np.array(tmp_ls)
    
    data_key = 'data'
    
    # for node information
    xyz_mm = xyzr_dt[data_key][:,:3] * xyzr_dt['spacing']  # convert voxel location to physical mm
    xyz_mm -= xyz_mm.mean(axis=0) # center normalization
    xyzr_mm = np.concatenate((xyz_mm, xyzr_dt[data_key][:,3:]), axis=1)  # 4 features, xyz and r
    xyzr_dt['data_mm'] = xyzr_mm
    
    graph = build_graph(xyzr_dt)
    
    out = Data(x=torch.tensor(xyzr_mm, dtype=torch.float),
               pos=torch.tensor(xyzr_dt[data_key][:, :3], dtype=torch.long),
               y= torch.tensor([dt[i] for i in target.split('-')], dtype=torch.float),
               edge_index= torch.tensor(list(graph.edges())).T, 
               dtype=torch.float) 
    
    # global dataset_dt
    # dataset_dt[mode].append(out)
    pickle.dump(out, open(dt['pcd_graph_fpath'], 'wb'))
    print('save data to ', dt['pcd_graph_fpath'])
    
    
def call_back(res):
    pass
    # print('')
    # print(f'Hello,World! {res}')
def error_callback(res):
    print(f'error: {res}')
    
def all_loaders(args, nb=None):
    target = 'DLCOc_SB-FEV1-FVC-TLC_He'
    label_fpath = '/home/jjia/data/dataset/lung_function/SScBaseline_PFT_anonymized_with_percent.xlsx'
    data_dir = '/home/jjia/data/dataset/lung_function//ori_resolution'
    label_excel = pd.read_excel(label_fpath, engine='openpyxl')
    label_excel = label_excel.sort_values(by=['subjectID'])
    label_excel = clean_data(label_excel, data_dir, target, top_pats=None)

    data = np.array(label_excel.to_dict('records'))
    for d in data:
        d['fpath'] = data_dir + '/' + d['subjectID'] + '_skeleton_coordinates140000.pt'
        

    # random.shuffle(data)  # Four fold are not right !!!
    tr_data, vd_data, ts_data = pat_from_json(data, args.fold)
    if nb:
        tr_data, vd_data, ts_data = tr_data[:nb], vd_data[:nb], ts_data[:nb]
        
    dataset_dt = {'train': [], 'valid': [], 'test': []} 
    
    def load_data():
        LOADED = True
        for data, mode in tqdm(zip([tr_data, vd_data, ts_data], ['train', 'valid', 'test'])): 
            
            for dt in tqdm(data):
                dt['pcd_graph_fpath'] = dt['fpath'].replace('.pt', '_graph.pt')
                if os.path.exists(dt['pcd_graph_fpath']):
                    # print(f"exist path: {dt['pcd_graph_fpath']}, load it directly")
                    with open(dt['pcd_graph_fpath'], 'rb') as f:
                        one_graph = pickle.load(f)
                        # one_graph.pat_id = np.array([int(dt['pcd_graph_fpath'].split('SSc_patient_')[-1][:7])]) 
                        # pickle.dump(one_graph, open(dt['pcd_graph_fpath'], 'wb'))
                        # print('save data to ', dt['pcd_graph_fpath'])
    
                        dataset_dt[mode].append(one_graph)
                else:
                    if LOADED:
                        pool = Pool(processes=12)
                        LOADED = False
                    print('start to calculate ', dt['pcd_graph_fpath'])
                    pool.apply_async(getdata, 
                                    args=(dt, 'DLCOc_SB-FEV1-FVC-TLC_He', ),
                                    callback=call_back,
                                    error_callback =error_callback )
            if not LOADED:
                pool.close()
                pool.join()
        
        if not LOADED:
            load_data()
            
    load_data()
        
       
    all_loaders = {k: DataLoader(data_ls, batch_size=args.batch_size) for k, data_ls in dataset_dt.items()}

    return all_loaders


