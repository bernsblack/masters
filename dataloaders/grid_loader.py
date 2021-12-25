import numpy as np

from datasets.grid_dataset import GridDataGroup, GridDataset
from utils.configs import BaseConf


class BaseDataLoaders:
    """
    Abstract Class for type hinting
    """

    def __init__(self):
        self.data_group = None
        self.train_loader = None
        self.validation_loader = None
        self.test_loader = None


class GridDataLoaders(BaseDataLoaders):
    def __init__(self, data_group: GridDataGroup, conf: BaseConf):
        super().__init__()
        self.data_group = data_group

        self.train_loader = GridBatchLoader(dataset=self.data_group.training_set,
                                            batch_size=conf.batch_size,
                                            shuffle=conf.shuffle)

        self.validation_loader = GridBatchLoader(dataset=self.data_group.validation_set,
                                                 batch_size=conf.batch_size,
                                                 shuffle=conf.shuffle)

        self.test_loader = GridBatchLoader(dataset=self.data_group.testing_set,
                                           batch_size=conf.batch_size,
                                           shuffle=conf.shuffle)


class GridBatchLoader:
    def __init__(self: 'GridBatchLoader', dataset: GridDataset, batch_size: int, shuffle: bool = True):
        self.dataset: GridDataset = dataset

        self.min_index: int = dataset.min_index
        self.max_index: int = dataset.max_index

        self.indices: np.ndarray = np.arange(self.min_index, self.max_index, dtype=int)

        self.shuffle: bool = shuffle
        if self.shuffle:
            np.random.shuffle(self.indices)

        self.batch_size: int = batch_size

        self.num_batches = int(np.ceil(len(self.indices) / self.batch_size))
        self.current_batch: int = 0

    def __len__(self) -> int:
        """
        number of batches in the batch loader
        :return:
        """
        return self.num_batches

    def __iter__(self: 'GridBatchLoader') -> 'GridBatchLoader':
        self.current_batch = 0
        return self

    def __next__(self: 'GridBatchLoader'):
        if self.current_batch >= self.num_batches:
            if self.shuffle:  # ensure reshuffling after each epoch
                np.random.shuffle(self.indices)
            raise StopIteration
        else:
            self.current_batch += 1
            start_index = (self.current_batch - 1) * self.batch_size
            stop_index = self.current_batch * self.batch_size
            if stop_index > len(self.indices):
                stop_index = len(self.indices)
            batch_indices = self.indices[start_index:stop_index]  # array of the indices - thus getitem should cater

            return self.dataset[batch_indices]

    def __getitem__(self: 'GridBatchLoader', index: int):
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


def reconstruct_from_grid_loader(batch_loader: GridBatchLoader):
    reconstructed_targets = np.zeros(batch_loader.dataset.target_shape, dtype=np.float)
    y_counts = batch_loader.dataset.targets[-len(reconstructed_targets):]
    t_range = batch_loader.dataset.t_range[-len(reconstructed_targets):]

    for batch_indices, batch_seq_c, batch_seq_p, batch_seq_q, batch_seq_e, batch_seq_t in batch_loader:

        for i, v in zip(batch_indices, batch_seq_t):
            reconstructed_targets[i] = v

    return y_counts, reconstructed_targets, t_range
