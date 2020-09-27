"""
File contains a series of classes that can be used as the config/base settings for the models and their
hyper parameters for training
"""
from copy import deepcopy
from pprint import pformat
from typing import Dict

from utils import Timer


class BaseConf:  # args from arg-parser to over write values
    """
    Configure class used to set/get global variable and
    """

    def __init__(self, conf_dict: Dict = None):  # get conf_dict either from a file, construct it or set it

        # default values
        # if the regression or classification version of the data loader should be used
        self.use_classification = True
        self.use_historic_average = True
        self.use_crime_types = False

        self.seed = 3
        # self.use_cuda = False

        # data related hyper-params
        self.val_ratio = 0.1
        self.tst_ratio = 0.3
        self.sub_sample_train_set = 1  # will sample class 0 and 1 with 1:1 ratio
        self.sub_sample_validation_set = 1  # will sample class 0 and 1 with 1:1 ratio
        self.sub_sample_test_set = 0  # will not sub sample class 0 and 1
        self.flatten_grid = True  # if the shaper should be used to squeeze the data

        self.seq_len = 1
        self.shaper_top_k = -1  # if less then 0, top_k will not be applied
        self.shaper_threshold = 0

        # training parameters
        self.resume = False
        self.early_stopping = False
        self.tolerance = 1e-8  # Convergence tolerance: difference between the past two validation losses
        self.lr = 1e-3
        self.weight_decay = 1e-8
        self.max_epochs = 1
        self.batch_size = 64
        self.dropout = 0
        self.shuffle = False
        self.num_workers = 6

        # attached global variables
        self.device = None  # pytorch device object [CPU|GPU]
        self.timer = Timer()
        self.model_name = ""
        self.model_path = ""  # is data_path/models/{model_name}
        self.data_path = ""
        self.checkpoint = "best"  # ['latest'|'best'] checkpoint to resume from

        #  used when train GRU - if loss should be calculated over whole sequence or only last output/prediction
        self.use_seq_loss = True

        # cnn values
        self.n_layers = 3  # number of res-unit layers
        self.n_channels = 3  # inner channel size of the res-units

        self.n_steps_q = 3
        self.n_steps_p = 3
        self.n_steps_c = 3

        self.pad_width = 0

        self.patience = 10

        if conf_dict:
            for k, v in conf_dict.items():
                if self.__dict__.get(k, None) is not None:
                    self.__dict__[k] = v

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()

#  create different confs that can be set and easily used from any where
#  argparsers should be able to overwrite some of confs values
