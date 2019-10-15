"""
File contains a series of classes that can be used as the config/base settings for the models and their
hyper parameters for training
"""
from pprint import pformat
from copy import deepcopy


class BaseConf:  # args from arg-parser to over write values
    """
    Configure class used to set/get global variable and
    """

    def __init__(self, conf_dict=None):  # get conf_dict either from a file, construct it or set it
        if conf_dict:
            self.__dict__ = deepcopy(conf_dict)
        else:  # default values
            self.seed = 3
            self.use_cuda = False

            # data related hyper-params
            self.val_ratio = 0.1
            self.tst_ratio = 0.2
            self.sub_sample_train_set = True
            self.sub_sample_validation_set = True
            self.sub_sample_test_set = False
            self.flatten_grid = True  # if the shaper should be used to squeeze the data

            self.seq_len = 1
            self.shaper_top_k = -1  # if less then 0, top_k will not be applied
            self.shaper_threshold = 0

            # training parameters
            self.resume = False
            self.early_stopping = False
            self.lr = 1e-3
            self.weight_decay = 1e-8
            self.max_epochs = 1
            self.batch_size = 64
            self.dropout = 0
            self.shuffle = False
            self.num_workers = 6

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()


#  create different confs that can be set and easily used from any where
#  argparsers should be able to overwrite some of confs values
