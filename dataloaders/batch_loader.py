import numpy as np


class BatchLoader:
    def __init__(self, dataset, batch_size, seq_len, sub_sample=True):
        """
        BatchLoader is a iterator that produces batches of data that can be fed into a model during a training
        loop. BatchLoader is used to iterate through and sample the indices for a whole epoch.

        :param dataset: PyTorch dataset
        :param batch_size: number of records the models see in a training step
        :param seq_len: length of historic data fed into the model
        :param sub_sample: used to decide if we limit class imbalance by sub-sampling the classes to be equal
        """
        self.dataset = dataset

        # values use for checking boundaries for the dataset
        self.min_index = dataset.min_index
        self.max_index = dataset.max_index

        flat_targets = self.dataset.targets.flatten()
        class0_args = np.argwhere(flat_targets == 0)
        class0_args = class0_args[class0_args > self.min_index]

        class1_args = np.argwhere(flat_targets > 0)
        class1_args = class1_args[class1_args > self.min_index]

        if sub_sample:
            np.random.shuffle(class0_args)
            np.random.shuffle(class1_args)
            class0_args = class0_args[:len(class1_args)]
            self.indices = np.array(list(zip(class0_args, class1_args))).flatten()
        else:
            self.indices = np.concatenate((class0_args, class1_args), axis=0).flatten()
            np.random.shuffle(self.indices)

        self.len = len(self.indices)

        self.batch_size = batch_size

        self.num_batches = int(np.ceil(len(self.indices) / self.batch_size))
        self.current_batch = 0

    def __len__(self):
        return self.len

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
                stop_index = self.len
            batch_indices = self.indices[start_index:stop_index]  # array of the indices - thus getitem should cater

            return self.dataset[batch_indices]
