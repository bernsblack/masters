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
from models.hawkes_model import HawkesModelGeneral, IndHawkesModel
from dataloaders.flat_loader import FlatDataLoaders
from datasets.flat_dataset import FlatDataGroup

from dataloaders.grid_loader import GridDataLoaders
from datasets.grid_dataset import GridDataGroup

from dataloaders.cell_loader import CellDataLoaders
from datasets.cell_dataset import CellDataGroup


from utils.plots import DistributionPlotter
from utils.metrics import PRCurvePlotter, ROCCurvePlotter, LossPlotter, CellPlotter
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, classification_report
import pickle
from utils.utils import pshape, pmax, pmin, pmean, get_data_sub_paths
import os
from utils.mock_data import mock_fnn_data_classification
import matplotlib.pyplot as plt
from utils.plots import im
from utils.metrics import best_threshold, get_y_pred
from models.model_result import ModelResult, ModelMetrics, save_metrics, compare_all_models,\
                                get_models_metrics, get_models_results, get_metrics_table
from models.baseline_models import ExponentialMovingAverage, UniformMovingAverage, \
                                    TriangularMovingAverage, HistoricAverage
from pprint import pprint

if __name__ == '__main__':
    data_sub_paths = get_data_sub_paths()
    pprint(data_sub_paths)

    data_sub_paths = ['T24H-X850M-Y880M_2013-01-01_2015-01-01']

    # run HAWKES INDEPENDENT MODEL for all data dimensions
    for data_sub_path in data_sub_paths:
        conf = BaseConf()

        conf.data_path = f"./data/processed/{data_sub_path}/"
        setup_logging(save_dir=conf.data_path,
                      log_config='./logger/standard_logger_config.json',
                      default_level=log.INFO)

        log.info("\n====================================================" + \
                 "===============================================================\n" + \
                 f"CALCULATING BASE LINE RESULTS FOR {data_sub_path}\n" + \
                 "==============================================================" + \
                 "=====================================================")

        if not os.path.exists(conf.data_path):
            raise Exception(f"Directory ({conf.data_path}) needs to exist.")

        # LOAD DATA
        conf.shaper_threshold = 0
        conf.shaper_top_k = -1
        data_group = FlatDataGroup(data_path=conf.data_path, conf=conf)
        loaders = FlatDataLoaders(data_group=data_group, conf=conf)

        # ------------ HAWKES INDEPENDENT MODEL
        conf.model_name = f"Ind-Hawkes Model"  # tod add the actual parameters as well
        conf.model_path = f"{conf.data_path}models/{conf.model_name}/"
        os.makedirs(conf.model_path, exist_ok=True)
        setup_logging(save_dir=conf.model_path,
                      log_config='./logger/standard_logger_config.json',
                      default_level=log.INFO)

        log.info("=====================================BEGIN=====================================")
        test_set_size = data_group.testing_set.target_shape[0]

        trn_crimes = data_group.training_set.crimes[:, 0]
        trn_y_true = data_group.training_set.targets

        tst_crimes = data_group.testing_set.crimes[:, 0]
        tst_y_true = data_group.testing_set.targets
        tst_t_range = data_group.testing_set.t_range

        # time step in this context is used for
        freqstr = data_group.t_range.freqstr
        if freqstr == "H":
            freqstr = "1H"
        time_step = int(24 / int(freqstr[:freqstr.find("H")]))

        kernel_size = time_step * 30

        N, L = np.shape(trn_crimes)

        model = IndHawkesModel(kernel_size=kernel_size)
        trn_probas_pred = model.fit_transform(trn_crimes)

        thresh = best_threshold(y_true=trn_y_true,
                                probas_pred=trn_probas_pred)

        tst_probas_pred = model.transform(tst_crimes)
        tst_y_pred = get_y_pred(thresh, tst_probas_pred)

        tst_y_true = tst_y_true[-test_set_size:]
        tst_y_pred = tst_y_pred[-test_set_size:]
        tst_y_pred = np.expand_dims(tst_y_pred, axis=1)
        tst_probas_pred = tst_probas_pred[-test_set_size:]
        tst_probas_pred = np.expand_dims(tst_probas_pred, axis=1)
        tst_t_range = tst_t_range[-test_set_size:]

        save_metrics(y_true=tst_y_true,
                     y_pred=tst_y_pred,
                     probas_pred=tst_probas_pred,
                     t_range=tst_t_range,
                     shaper=data_group.shaper,
                     conf=conf)

        # ------------HAWKES INDEPENDENT MODEL