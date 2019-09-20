import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from utils.configs import BaseConf
from utils.data_processing import map_to_weights, inv_weights
from datasets.mock_dataset import MockDataset
import numpy as np
import logging as log

"""
Data loader using the PyTorch default data loaders
This causes problems seeing that we cannot use our customer samplers
Thus is will not be used
"""


class MockDataLoaders:
    def __init__(self, conf: BaseConf, file_path=None):

        # have the train, validation and testing data available im memory
        # (maybe not necessary to have test set in memory tpp)
        if file_path:
            # load data from file
            in_features, targets = None, None
            self.in_size = 0
            self.out_size = 2
            raise NotImplemented
        else:  # generate generic data
            N, d = 1000, 5
            bias = -0.05
            in_features = np.random.randn(N, d) + bias
            targets = np.array(0.5 * (np.sign(np.sum(in_features, axis=1)) + 1), dtype=int)
            log.info(f"Inverted Class Weights: {inv_weights(targets)}")

            self.in_size = d
            self.out_size = 2

        # DATA LOADER SETUP
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.manual_seed(conf.seed)
            pin_memory = True  # makes RAM -> GPU transfers quicker
        else:
            torch.manual_seed(conf.seed)
            pin_memory = False

        device = torch.device("cuda:0" if use_cuda else "cpu")

        val_size = int(len(in_features) * conf.val_ratio)
        tst_size = int(len(in_features) * conf.tst_ratio)

        trn_index = (0, len(in_features) - tst_size - val_size)
        val_index = (trn_index[1], len(in_features) - tst_size)
        tst_index = (val_index[1], len(in_features))

        training_set = MockDataset(in_features[trn_index[0]:trn_index[1]], targets[trn_index[0]:trn_index[1]])
        validation_set = MockDataset(in_features[val_index[0]:val_index[1]], targets[val_index[0]:val_index[1]])
        testing_set = MockDataset(in_features[tst_index[0]:tst_index[1]], targets[tst_index[0]:tst_index[1]])

        training_weights = map_to_weights(targets[trn_index[0]:trn_index[1]])
        validation_weights = map_to_weights(targets[val_index[0]:val_index[1]])
        testing_weights = map_to_weights(targets[tst_index[0]:tst_index[1]])

        """
        Important note on sampling
        ====
        The entire dataset will still be seen by the model, but the distribution of labels
        should tend to be uniformly distributed for the majority of the batches. If we want 
        the batches to always remain uniformly distributed set -> replacement=True
        """
        if conf.use_weighted_sampler:
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
