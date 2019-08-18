import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from utils.configs import BaseConf
from datasets.generic_dataset import CrimeDataGroup
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


class CrimeDataLoaders:
    def __init__(self, data_path, conf: BaseConf):
        # have the train, validation and testing data available im memory
        # (maybe not necessary to have test set in memory tpp)
        # DATA LOADER SETUP

        data_group = CrimeDataGroup(data_path=data_path, conf=conf)

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


class OLDCrimeDataLoaders:
    def __init__(self, conf: BaseConf, data_path):

        # have the train, validation and testing data available im memory
        # (maybe not necessary to have test set in memory tpp)
        # DATA LOADER SETUP
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.manual_seed(conf.seed)
            pin_memory = True  # makes RAM -> GPU transfers quicker
        else:
            torch.manual_seed(conf.seed)
            pin_memory = False

        data_group = CrimeDataGroup(data_path=data_path, conf=conf)

        """
        Important note on sampling
        ====
        The entire dataset will still be seen by the model, but the distribution of labels
        should tend to be uniformly distributed for the majority of the batches. If we want 
        the batches to always remain uniformly distributed set -> replacement=True
        """
        if conf.use_weighted_sampler:
            training_weights = data_group.training_set.sample_weights
            validation_weights = data_group.validation_set.sample_weights
            testing_weights = data_group.training_set.sample_weights

            train_sampler = WeightedRandomSampler(weights=training_weights,
                                                  num_samples=len(training_weights),
                                                  replacement=False)
            validation_sampler = WeightedRandomSampler(weights=validation_weights,
                                                       num_samples=len(validation_weights),
                                                       replacement=False)
            testing_sampler = WeightedRandomSampler(weights=testing_weights,
                                                    num_samples=len(testing_weights),
                                                    replacement=False)
        else:
            train_sampler = None
            validation_sampler = None
            testing_sampler = None

        training_set = data_group.training_set
        validation_set = data_group.validation_set
        testing_set = data_group.testing_set
        self.training_generator = DataLoader(dataset=training_set,
                                             batch_size=conf.batch_size,
                                             shuffle=conf.shuffle,
                                             num_workers=conf.num_workers,
                                             pin_memory=pin_memory,
                                             sampler=train_sampler)

        self.validation_generator = DataLoader(dataset=validation_set,
                                               batch_size=conf.batch_size,
                                               shuffle=conf.shuffle,
                                               num_workers=conf.num_workers,
                                               pin_memory=pin_memory,
                                               sampler=validation_sampler)

        self.testing_generator = DataLoader(dataset=testing_set,
                                            batch_size=conf.batch_size,
                                            shuffle=conf.shuffle,
                                            num_workers=conf.num_workers,
                                            pin_memory=pin_memory,
                                            sampler=testing_sampler)
