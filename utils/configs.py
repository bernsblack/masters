"""
File contains a series of classes that can be used as the config/base settings for the models and their
hyper parameters for training
"""


class BaseConf:  # args from arg-parser to over write values
    """
    Configure class used to set/get global variable and
    """
    def __init__(self, conf_file=None, args=None):
        self.seed = 3
        self.resume = False
        self.early_stopping = False
        self.use_weighted_sampler = True
        self.use_cuda = False

        # data related hyper-params
        self.val_ratio = 0.1
        self.tst_ratio = 0.2

        # training parameters
        self.lr = 1e-3
        self.weight_decay = 1e-8
        self.max_epochs = 10
        self.batch_size = 80
        self.shuffle = False
        self.num_workers = 6

        if conf_file is not None:
            raise NotImplemented  # todo over-write internal values

        if args is not None:
            raise NotImplemented  # todo over-write internal values

#  create different confs that can be set and easily used from any where
#  argparsers should be able to overwrite some of confs values
