from utils.plots import plot, displot
from seaborn import distplot
import matplotlib.pyplot as plt
from models.baseline_models import historic_average

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
from models.kangkang_fnn_models import KangFeedForwardNetwork, SimpleKangFNN, evaluate_fnn
from dataloaders.flat_loader import FlatDataLoaders, MockLoader, MockLoaders
from datasets.flat_dataset import FlatDataGroup
from utils.metrics import PRCurvePlotter, ROCCurvePlotter, LossPlotter, PerTimeStepPlotter
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from models.model_result import ModelResult, ModelMetrics, save_results, save_metrics, \
    compare_all_models, get_models_metrics
from utils.mock_data import mock_fnn_data_classification
from utils.plots import im
from utils.utils import pshape, get_data_sub_paths, by_ref
from trainers.generic_trainer import train_model
from models.kangkang_fnn_models import train_epoch_for_fnn

from utils.metrics import best_threshold, get_y_pred, get_y_pred_by_thresholds, best_thresholds
from time import time
from torch.optim import lr_scheduler

from pprint import pprint
import logging
from utils.plots import plot_time_signals

from utils.forecasting import compare_time_series_metrics


if __name__ == '__main__':
    os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())

    data_sub_paths = get_data_sub_paths()
    print(np.sort(data_sub_paths))

    data_sub_paths = by_ref("1f1")
    print(data_sub_paths)

    data_sub_path = data_sub_paths[0]
    time_steps_per_day = 24 / int(data_sub_path[data_sub_path.find('T') + 1:data_sub_path.find('H')])
    print(time_steps_per_day)

    # manually set
    conf = BaseConf()
    conf.seed = int(time())  # 3
    conf.model_name = "CityCount"  # "SimpleKangFNN" # "KangFNN"  # needs to be created
    conf.data_path = f"./data/processed/{data_sub_path}/"

    if not os.path.exists(conf.data_path):
        raise Exception(f"Directory ({conf.data_path}) needs to exist.")

    conf.model_path = f"{conf.data_path}models/{conf.model_name}/"
    os.makedirs(conf.data_path, exist_ok=True)
    os.makedirs(conf.model_path, exist_ok=True)

    # logging config is set globally thus we only need to call this in this file
    # imported function logs will follow the configuration
    setup_logging(save_dir=conf.model_path, log_config='./logger/standard_logger_config.json', default_level=log.INFO)
    log.info("=====================================BEGIN=====================================")

    info = deepcopy(conf.__dict__)
    info["start_time"] = strftime("%Y-%m-%dT%H:%M:%S")

    # DATA LOADER SETUP
    np.random.seed(conf.seed)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(conf.seed)
    else:
        torch.manual_seed(conf.seed)

    device = torch.device("cuda:0" if use_cuda else "cpu")
    log.info(f"Device: {device}")
    info["device"] = device.type
    conf.device = device

    # SET THE HYPER PARAMETERS
    conf.shaper_top_k = -1
    conf.use_classification = False
    conf.train_set_first = True

    conf.use_crime_types = True
    data_group = FlatDataGroup(data_path=conf.data_path, conf=conf)
    data_group.crimes.shape
    # loaders = FlatDataLoaders(data_group=data_group, conf=conf)


