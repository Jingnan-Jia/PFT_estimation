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
from lung_function.modules.tool import (record_1st, dec_record_cgpu, retrive_run, try_func, int2str, txtprocess, log_all_metrics)
from lung_function.modules.trans import batch_bbox2_3D
from lung_function.scripts.run import run, main

args = get_args()
global_lock = threading.Lock()



log_metric = try_func(log_metric)
log_metrics = try_func(log_metrics)







        

if __name__ == "__main__":
    main()
