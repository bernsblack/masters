from dataloaders.grid_loader import BaseDataLoaders
from datasets.flat_dataset import FlatDataGroup, FlatDataset
from utils.configs import BaseConf
import numpy as np
import pandas as pd

from utils.mock_data import mock_fnn_data_classification


class FlatDataLoaders(BaseDataLoaders):
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

        self.train_loader = FlatBatchLoader(dataset=self.data_group.training_set,
                                            batch_size=conf.batch_size,
                                            sub_sample=conf.sub_sample_train_set)

        self.validation_loader = FlatBatchLoader(dataset=self.data_group.validation_set,
                                                 batch_size=conf.batch_size,
                                                 sub_sample=conf.sub_sample_validation_set)

        self.test_loader = FlatBatchLoader(dataset=self.data_group.testing_set,
                                           batch_size=conf.batch_size,
                                           sub_sample=conf.sub_sample_test_set)


# todo add sequence loader - runs through the entire sets sequence first then
class FlatBatchLoader:
    def __init__(self, dataset: FlatDataset, batch_size: int, sub_sample: int = 0):
        """
        BatchLoader is a iterator that produces batches of data that can be fed into a model during a training
        loop. BatchLoader is used to iterate through and sample the indices for a whole epoch.

        :param dataset: PyTorch dataset
        :param batch_size: number of records the models see in a training step
        :param sub_sample: used to decide if we limit class imbalance by sub-sampling the classes to be equal
        """
        # SET DATA
        self.dataset = dataset

        # SET LIMITS
        # values use for checking boundaries for the dataset
        self.min_index = dataset.min_index
        self.max_index = dataset.max_index

        # SET SAMPLE INDICES
        flat_targets = self.dataset.targets.flatten()
        class0_args = np.argwhere(flat_targets == 0)
        class0_args = class0_args[class0_args >= self.min_index]

        class1_args = np.argwhere(flat_targets > 0)
        class1_args = class1_args[class1_args >= self.min_index]

        # todo - set subsample ratio_cls0_cls1 = 2 - and let np.choose(class0_args, take_len_class1)
        sub_sample = int(sub_sample)
        if sub_sample > 0:  # options: 0, 1, 2, 3 should always be int
            np.random.shuffle(class0_args)
            np.random.shuffle(class1_args)
            # class0_args = class0_args[:int(ratio_cls0_cls1*len(class1_args))]
            class0_args = class0_args[:sub_sample * len(class1_args)]
            self.indices = np.array(list(zip(class0_args, class1_args))).flatten()
        else:
            self.indices = np.concatenate((class0_args, class1_args), axis=0).flatten()  # [0,0,1,1] format
            np.random.shuffle(self.indices)

        self.batch_size = batch_size

        self.num_batches = int(np.ceil(len(self.indices) / self.batch_size))
        self.current_batch = 0

    def __len__(self):
        """
        number of batches in the batch loader
        :return:
        """
        return self.num_batches

    def __iter__(self):
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            np.random.shuffle(self.indices)  # reshuffle indices after each epoch
            raise StopIteration
        else:
            self.current_batch += 1
            start_index = (self.current_batch - 1) * self.batch_size
            stop_index = self.current_batch * self.batch_size
            if stop_index > len(self.indices):
                stop_index = len(self.indices)
            batch_indices = self.indices[start_index:stop_index]  # array of the indices - thus getitem should cater

            return self.dataset[batch_indices]

    def __getitem__(self, index):
        """
        :param index: the current batch
        :return: batch of data where batch == index
        """
        if 0 <= index < self.num_batches and isinstance(index, int):
            start_index = index * self.batch_size
            stop_index = (index + 1) * self.batch_size
            if stop_index > len(self.indices):
                stop_index = len(self.indices)
            batch_indices = self.indices[start_index:stop_index]  # array of the indices - thus getitem should cater

            return self.dataset[batch_indices]
        else:
            raise IndexError(f"index {index} must be in range(0{self.num_batches})")


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
        return self.indices[index], *r


def reconstruct_from_flat_loader(batch_loader: FlatBatchLoader):
    reconstructed_targets = np.zeros(batch_loader.dataset.target_shape)
    y_true = batch_loader.dataset.targets[-len(reconstructed_targets):]
    t_range = batch_loader.dataset.t_range[-len(reconstructed_targets):]
    y_class = np.zeros(y_true.shape)  # classes {0, 1}

    for indices, spc_feats, tmp_feats, env_feats, targets, labels in batch_loader:
        for i in range(len(indices)):
            n, c, l = indices[i]
            reconstructed_targets[n, c, l] = targets[-1, i]
            y_class[n, c, l] = 1

    return y_true, reconstructed_targets, t_range
