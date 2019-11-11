import os
import logging as log
from time import strftime
from copy import deepcopy
from torch import nn, optim
import torch.nn.functional as F
from utils.data_processing import *
from logger.logger import setup_logging
from utils.configs import BaseConf
from utils.utils import write_json, Timer
from models.kangkang_fnn_models import KangFeedForwardNetwork
from dataloaders.flat_loader import FlatDataLoaders
from datasets.flat_dataset import FlatDataGroup, BaseDataGroup
from utils.metrics import PRCurvePlotter, ROCCurvePlotter, LossPlotter
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from models.model_result import ModelResult
from utils.mock_data import mock_data
import matplotlib.pyplot as plt
from utils.plots import im
from torch.nn.utils import clip_grad_norm_
from utils.mock_data import mock_adding_problem_data, mock_rnn_data

if __name__ == '__main__':

    data_dim_str = "T24H-X850M-Y880M"  # "T1H-X1700M-Y1760M"  # needs to exist
    model_name = "RNN-CRIME-MODEL"  # needs to be created

    data_path = f"./data/processed/{data_dim_str}/"
    model_path = f"{data_path}models/{model_name}/"
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # logging config is set globally thus we only need to call this in this file
    # imported function logs will follow the configuration
    setup_logging(save_dir=model_path, log_config='./logger/standard_logger_config.json', default_level=log.INFO)

    # manually set the config
    conf_dict = {
        "seed": 3,
        "use_cuda": False,

        "use_crime_types": False,

        # data group/data set related
        "val_ratio": 0.1,  # ratio of the total dataset
        "tst_ratio": 0.2,  # ratio of the total dataset
        "seq_len": 1,
        "flatten_grid": True,  # if the shaper should be used to squeeze the data

        # shaper related
        "shaper_top_k": -1,  # if less then 0, top_k will not be applied
        "shaper_threshold": 0,

        # data loader related
        "sub_sample_train_set": 1,
        "sub_sample_validation_set": 1,
        "sub_sample_test_set": 0,

        # training parameters
        "resume": False,
        "early_stopping": False,
        "tolerance": 1e-8,
        "lr": 1e-3,
        "weight_decay": 1e-8,
        "max_epochs": 1,
        "batch_size": 64,
        "dropout": 0.2,
        "shuffle": False,
        "num_workers": 6,

        # attached global variables - bad practice -find alternative
        "device": None,  # pytorch device object [CPU|GPU]
        "timer": Timer(),
        "model_name": model_name,
        "model_path": model_path,
    }

    conf = BaseConf(conf_dict=conf_dict)

    info = deepcopy(conf.__dict__)
    info["start_time"] = strftime("%Y-%m-%dT%H:%M:%S")

    # DATA LOADER SETUP
    np.random.seed(conf.seed)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(conf.seed)
    else:
        torch.manual_seed(conf.seed)

    conf.device = torch.device("cuda:0" if use_cuda else "cpu")

    base_data_group = BaseDataGroup(data_path=data_path, conf=conf) \
        .with_time_vectors() \
        .with_weather_vectors() \
        .with_demog_grid() \
        .with_street_grid()

    data_group = FlatDataGroup(data_path=data_path, conf=conf)
