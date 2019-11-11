import numpy as np
# todo add sequence loader - runs through the entire sets sequence first then
# todo rename to flat BatchLoader
class BatchLoader:
    def __init__(self, dataset, batch_size, sub_sample=0):
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
        if sub_sample > 0: # options: 0, 1, 2, 3 should always be int
            np.random.shuffle(class0_args)
            np.random.shuffle(class1_args)
            # class0_args = class0_args[:int(ratio_cls0_cls1*len(class1_args))]
            class0_args = class0_args[:sub_sample*len(class1_args)]
            self.indices = np.array(list(zip(class0_args, class1_args))).flatten()
        else:
            self.indices = np.concatenate((class0_args, class1_args), axis=0).flatten() # [0,0,1,1] format
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
