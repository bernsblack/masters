"""
File contains a series of classes that can be used as the config/base settings for the models and their
hyperparameters for training
"""
from copy import deepcopy
from os import cpu_count
from pprint import pformat
from typing import Dict, Optional

from torch.types import Device

from utils import Timer


class BaseConf:  # args from arg-parser to over write values
    """
    Configure class used to set/get global variable and
    """

    def __init__(self, conf_dict: Dict = None):  # get conf_dict either from a file, construct it or set it

        # default values
        # if the regression or classification version of the data loader should be used
        self.use_classification: bool = False
        self.use_crime_types: bool = False
        self.log_norm_scale: bool = True
        self.use_periodic_average: bool = False  # only add the periodic average to a channel - does not normalise
        # the data at all.

        self.scale_axis: int = 1  # which axis should be scaled to ensure values lie between min and max

        self.seed: float = 3
        # self.use_cuda = False

        # data related hyper-params
        self.val_ratio: float = 0.25
        self.tst_ratio: float = 0.25
        self.sub_sample_train_set: int = 1  # will sample class 0 and 1 with 1:1 ratio
        self.sub_sample_validation_set: int = 1  # will sample class 0 and 1 with 1:1 ratio
        self.sub_sample_test_set: int = 0  # will not sub sample class 0 and 1
        self.flatten_grid: bool = True  # if the shaper should be used to squeeze the data
        self.test_set_size_days: int = 360  # should be changed to 64 if freq is less than one day

        self.seq_len: int = 1
        self.shaper_top_k: int = -1  # if less then 0, top_k will not be applied
        self.shaper_threshold: float = 0

        # training parameters
        self.resume: bool = False
        self.early_stopping: bool = False
        self.tolerance: float = 1e-8  # Convergence tolerance: difference between the past two validation losses
        self.lr: float = 1e-3
        self.weight_decay: float = 1e-8
        self.max_epochs: int = 1
        self.min_epochs: int = 1
        self.batch_size: int = 64
        self.dropout: float = 0.0  # dropout probability
        self.shuffle: bool = True
        self.num_workers: int = cpu_count()

        # attached global variables
        self.device: Optional[Device] = None  # pytorch device object [CPU|GPU]
        self.timer: Timer = Timer()
        self.model_name: str = ""
        self.model_path: str = ""  # is data_path/models/{model_name}
        self.data_path: str = ""
        self.plots_path: str = ""
        self.checkpoint: str = "best"  # ['latest'|'best'] checkpoint to resume from

        #  used when train GRU - if loss should be calculated over whole sequence or only last output/prediction
        self.use_seq_loss: bool = True

        # cnn values
        self.n_layers: int = 3  # number of res-unit layers
        self.n_channels: int = 3  # inner channel size of the res-units

        self.n_steps_q: int = 3
        self.n_steps_p: int = 3
        self.n_steps_c: int = 3

        self.pad_width: int = 0

        self.patience: int = 10  # how many epochs the training continues after validation metrics have been increasing

        # when train and valid sets are split in time: if the train set should be earlier in time than the validation
        # set
        self.train_set_first: bool = True

        # Will cap the maximum value to be equal to the 99.9 percentile - to limit the outliers and remove
        # unnecessary scaling
        self.cap_crime_percentile: float = 0  # 99.95 # if zero then no cap takes place

        self._freq: Optional[str] = None  # data time step frequency
        self.freq_title: str = ""  # string description of time series frequency
        self.time_steps_per_day: int = 1

        self.hidden_size: int = 100
        self.num_layers: int = 2

        if conf_dict:
            for k, v in conf_dict.items():
                if self.__dict__.get(k, None) is not None:
                    self.__dict__[k] = v

    def set_freq(self, freq: str):
        self._freq = freq

    @property
    def freq(self):
        if self._freq is None:
            raise Exception(f"conf.freq has not been set")
        else:
            return self._freq

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()

#  create different confs that can be set and easily used from any where
#  argparsers should be able to overwrite some of confs values
