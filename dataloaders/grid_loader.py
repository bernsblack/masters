from utils.configs import BaseConf
from datasets.grid_dataset import GridDataGroup
from dataloaders.batch_loader import BatchLoader


class GridDataLoaders:
    """
    Container for the data group and the TRAIN/TEST/VALIDATION batch loaders
    The data group loads data from disk, splits in separate sets and normalises according to train set.
    The data group class also handles reshaping of data.
    """

    def __init__(self, data_path, conf: BaseConf):
        # have the train, validation and testing data available im memory
        # (maybe not necessary to have test set in memory tpp)
        # DATA LOADER SETUP

        data_group = GridDataGroup(data_path=data_path, conf=conf)

        training_set = data_group.training_set
        validation_set = data_group.validation_set
        testing_set = data_group.testing_set
        self.training_generator = BatchLoader(dataset=training_set,
                                              batch_size=conf.batch_size,
                                              sub_sample=True)

        self.validation_generator = BatchLoader(dataset=validation_set,
                                                batch_size=conf.batch_size,
                                                sub_sample=True)

        self.testing_generator = BatchLoader(dataset=testing_set,
                                             batch_size=conf.batch_size,
                                             sub_sample=conf.sub_sample_test_set)
