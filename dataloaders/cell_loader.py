from dataloaders.batch_loader import BatchLoader
from datasets.cell_dataset import CellDataGroup
from datasets.flat_dataset import FlatDataGroup
from utils.configs import BaseConf
import numpy as np
import pandas as pd

from utils.mock_data import mock_fnn_data_classification


class CellDataLoaders:
    """
    Container for the data group and the TRAIN/TEST/VALIDATION batch loaders
    The data group loads data from disk, splits in separate sets and normalises according to train set.
    The data group class also handles reshaping of data.
    """

    def __init__(self, data_group: CellDataGroup, conf: BaseConf):
        # have the train, validation and testing data available im memory
        # (maybe not necessary to have test set in memory tpp)
        # DATA LOADER SETUP

        self.data_group = data_group

        self.train_loader = BatchLoader(dataset=self.data_group.training_set,
                                        batch_size=conf.batch_size,
                                        sub_sample=conf.sub_sample_train_set)

        self.validation_loader = BatchLoader(dataset=self.data_group.validation_set,
                                             batch_size=conf.batch_size,
                                             sub_sample=conf.sub_sample_validation_set)

        self.test_loader = BatchLoader(dataset=self.data_group.testing_set,
                                       batch_size=conf.batch_size,
                                       sub_sample=conf.sub_sample_test_set)
