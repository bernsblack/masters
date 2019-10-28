from datasets.grid_dataset import GridDataGroup
from utils.configs import BaseConf
import numpy as np


class GridDataLoaders:
    def __init__(self, data_group: GridDataGroup, conf: BaseConf):
        self.data_group = data_group

        self.train_loader = GridBatchLoader(dataset=self.data_group.training_set,
                                            batch_size=conf.batch_size,
                                            shuffle=True)

        self.validation_loader = GridBatchLoader(dataset=self.data_group.validation_set,
                                                 batch_size=conf.batch_size,
                                                 shuffle=True)

        self.test_loader = GridBatchLoader(dataset=self.data_group.testing_set,
                                           batch_size=conf.batch_size,
                                           shuffle=False)


class GridBatchLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset

        self.min_index = dataset.min_index
        self.max_index = dataset.max_index

        self.indices = np.arange(self.min_index, self.max_index, dtype=int)
        if shuffle:
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
