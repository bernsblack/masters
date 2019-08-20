import numpy as np


class BatchLoader:
    def __init__(self, dataset, batch_size, sub_sample=True):
        """
        sub_sample: used to decide if we limit class imbalance by sub-sampling the classes to be equal
        """
        self.dataset = dataset
        flat_targets = self.dataset.targets.flatten()
        class0_args = np.argwhere(flat_targets == 0)
        class1_args = np.argwhere(flat_targets > 0)

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
            batch_indices = self.indices[start_index:stop_index]

            return self.dataset[batch_indices]