from dataloaders.batch_loader import BatchLoader
from datasets.flat_dataset import FlatDataGroup
from utils.configs import BaseConf


# todo (rename) to Loader
class FlatDataLoaders:
    """
    Container for the data group and the TRAIN/TEST/VALIDATION batch loaders
    The data group loads data from disk, splits in separate sets and normalises according to train set.
    The data group class also handles reshaping of data.
    """

    def __init__(self, data_path, conf: BaseConf):
        # have the train, validation and testing data available im memory
        # (maybe not necessary to have test set in memory tpp)
        # DATA LOADER SETUP

        self.data_group = FlatDataGroup(data_path=data_path, conf=conf)

        self.training_loader = BatchLoader(dataset=self.data_group.training_set,
                                           batch_size=conf.batch_size,
                                           seq_len=conf.seq_len,
                                           sub_sample=True)

        self.validation_loader = BatchLoader(dataset=self.data_group.validation_set,
                                             batch_size=conf.batch_size,
                                             seq_len=conf.seq_len,
                                             sub_sample=True)

        self.testing_loader = BatchLoader(dataset=self.data_group.testing_set,
                                          batch_size=conf.batch_size,
                                          seq_len=conf.seq_len,
                                          sub_sample=conf.sub_sample_test_set)
