#!python3
"""
This file is only used for quick iterative development
"""

import pandas as pd
import numpy as np
from pprint import pprint
import os
import logging as log
from pprint import pprint
# drop weather fo now and just check what you can get
# or cap everything to the length of the weather
# or cap
from datasets.grid_dataset import GridDataGroup
from datasets.flat_dataset import FlatDataGroup
from utils.plots import im
import pandas as pd
from pprint import pprint
import numpy as np
from utils.configs import BaseConf
import matplotlib.pyplot as plt
from models.baseline_models import ExponentialMovingAverage, UniformMovingAverage, \
    TriangularMovingAverage, HistoricAverage
from utils.metrics import CellPlotter

# data_dim_strs = [
#     "T12H-X850M-Y880M/",
#     "T1H-X1700M-Y1760M/",
#     "T24H-X425M-Y440M/",
#     "T24H-X850M-Y880M/",
#     "T24H-X85M-Y110M/",
#     "T3H-X850M-Y880M/",
#     "T4H-X850M-Y880M/",
#     "T6H-X850M-Y880M/"
# ]

data_dim_strs = os.listdir("./data/processed")[1:]
pprint(data_dim_strs)

data_dim_str = "T24H-X850M-Y880M"
data_path = f"./data/processed/{data_dim_str}/"

conf_dict = {
    "seed": 3,
    "resume": False,
    "early_stopping": False,
    "use_cuda": False,
    "val_ratio": 0.1,
    "tst_ratio": 0.2,
    "sub_sample_train_set": True,
    "sub_sample_validation_set": True,
    "sub_sample_test_set": False,
    "flatten_grid": True,
    "lr": 1e-3,
    "weight_decay": 1e-8,
    "max_epochs": 10,
    "batch_size": 64,
    "dropout": 0,
    "shuffle": False,
    "num_workers": 6,
    "seq_len": 10
}

conf = BaseConf(conf_dict=conf_dict)

#  todo (bernard): when creating the objects print out their shapes and values on the create...
data_group = FlatDataGroup(data_path=data_path, conf=conf)

validation_set = data_group.validation_set
spc_feats, tmp_feats, env_feats, target = validation_set[validation_set.min_index:validation_set.min_index+10]

d = {"spc_feats": spc_feats, "tmp_feats": tmp_feats, "env_feats": env_feats, "target": target}
for k in d.keys():
    print(k, d[k].shape)
    if k == "tmp_feats":
        print("tmp_feats ->", d[k])


# batch_loader - test loader only the test data no sub-sampling and all the data in sequential order this way
