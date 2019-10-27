from dataloaders.batch_loader import BatchLoader
from datasets.flat_dataset import FlatDataGroup
from utils.configs import BaseConf
import numpy as np
import pandas as pd

# todo (rename) to Loader
from utils.mock_data import mock_fnn_data_classification


class FlatDataLoaders:
    """
    Container for the data group and the TRAIN/TEST/VALIDATION batch loaders
    The data group loads data from disk, splits in separate sets and normalises according to train set.
    The data group class also handles reshaping of data.
    """

    def __init__(self, data_group: FlatDataGroup, conf: BaseConf):
        # have the train, validation and testing data available im memory
        # (maybe not necessary to have test set in memory tpp)
        # DATA LOADER SETUP

        self.data_group = data_group

        self.train_loader = BatchLoader(dataset=self.data_group.training_set,
                                        batch_size=conf.batch_size,
                                        sub_sample=True)

        self.validation_loader = BatchLoader(dataset=self.data_group.validation_set,
                                             batch_size=conf.batch_size,
                                             sub_sample=True)

        self.test_loader = BatchLoader(dataset=self.data_group.testing_set,
                                       batch_size=conf.batch_size,
                                       sub_sample=conf.sub_sample_test_set)


# -------------------------------------------------- MockLoaders --------------------------------------------------
class MockLoaders:
    def __init__(self, train_loader=None, validation_loader=None, test_loader=None):
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader


class MockLoader:
    def __init__(self, vector_size, batch_size, n_samples, class_split):
        n_feats = np.sum(vector_size)
        X, y = mock_fnn_data_classification(n_samples=n_samples, n_feats=n_feats, class_split=class_split)

        y = np.expand_dims(y, axis=1)
        y = np.expand_dims(y, axis=0)

        # (N,C,L) in the dataset but (seq_len, batch_size, n_features) in loader
        self.targets = np.copy(y).swapaxes(0, 1)  # format is now in (N,C,L)
        self.t_range = pd.date_range(start='2012', end='2016', periods=n_samples)
        self.shaper = None
        X = np.expand_dims(X, axis=0)

        self.indices = np.array([(i, 0, 0) for i in np.arange(n_samples)])  # (N,C,L) format

        vectors = []
        i = 0
        for j in vector_size:
            vectors.append(X[:, :, i:i + j])
            i += j
        vectors.append(y)

        self.n_samples = n_samples
        self.n_feats = n_feats
        self.vectors = vectors
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(self.n_samples / self.batch_size))
        self.current_batch = 0

        self.max_index = n_samples
        self.min_index = 0
        self.dataset = self  # just to act as an interface

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
        else:
            self.current_batch += 1
            start_index = (self.current_batch - 1) * self.batch_size
            stop_index = self.current_batch * self.batch_size
            if stop_index > len(self):
                stop_index = len(self)
            return self[start_index:stop_index]

    def __getitem__(self, index):
        r = tuple(map(lambda x: x[:, index], self.vectors))
        return (self.indices[index], *r)
