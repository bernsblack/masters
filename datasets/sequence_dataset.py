from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class SequenceDataset(Dataset):
    def __init__(self, input_data, target_data, t_range, seq_len=30):
        """
        data: shape (N,n_features)
        seq_len: sequence length
        """
        assert len(input_data) == len(target_data)

        self.input_data = input_data
        self.target_data = target_data
        self.t_range = t_range

        self.seq_len = seq_len

        self.target_shape = (len(self.input_data) - self.seq_len - 1, self.target_data.shape[-1])

        self.length = len(self.input_data) - self.seq_len - 1

        self.input_data = torch.Tensor(self.input_data)
        self.target_data = torch.Tensor(self.target_data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        start = idx
        end = idx + self.seq_len

        inp = self.input_data[start:end]
        trg = self.target_data[start:end]

        return idx, inp, trg


class SequenceDataLoaders:
    def __init__(self, input_data, target_data, t_range, seq_len=30, batch_size=1, shuffle=True,
                 val_ratio=0.2, tst_ratio=0.2, tst_size=None, num_workers=0, overlap_sequences=False):
        """
        Indices will be chosen so that training, validation and test sets only overlap by sequence length, this way
        we can feed in data from the train_val set into the evaluation model, preserving more data to evaluate without
        leaking evaluation target data into our train_val set because we only look at the final forecasted value given
        the sequence for the evaluation model.

        :param input_data: all input sequence data, train val en test sets included in nd array format
        :param target_data: all target sequence data, train val en test sets included in nd array format
        :param t_range: all datetime sequence data, train val en test sets included in nd array format
        :param seq_len: sequence length used to feed sequenced data into models
        :param batch_size: size of batches when training models
        :param shuffle: if the indices should be shuffled between epochs
        :param val_ratio: validation set ratio of the train_val set
        :param tst_ratio: test set ration of the total dataset
        :param tst_size: explicit test set size. tst_ratio is ignored when tst_size is set
        :param num_workers: number cpu's used to load data when iterating over set
        :param overlap_sequences: of datasets should overlap by the sequence length - only of concern if the loss functions in the training loop use the whole sequence outputs to calculate the loss - which in turn leads to quicker train times.
        """
        assert len(input_data) == len(target_data)

        total_len = len(input_data)

        if tst_size is None:
            tst_size = int(total_len * tst_ratio)

        trn_val_size = total_len - tst_size
        val_size = int(trn_val_size * val_ratio)
        trn_size = trn_val_size - val_size

        # overlapping by sequence length - if loss values are calculated using entire sequence output
        # instead of just the final sequence value there will be issues training and validation values that overlap
        # which means validation loss will decrease as training starts to over-fit because they share values in their
        # respective loss calculations
        if overlap_sequences:
            trn_idx = np.array([0, trn_size])
            val_idx = np.array([trn_idx[1] - seq_len, trn_idx[1] + val_size])
            trn_val_idx = np.array([0, val_idx[1]])
            tst_idx = np.array([val_idx[1] - seq_len, val_idx[1] + tst_size])
        else:
            trn_idx = np.array([0, trn_size])
            val_idx = np.array([trn_idx[1], trn_idx[1] + val_size])
            trn_val_idx = np.array([0, val_idx[1]])
            tst_idx = np.array([val_idx[1], val_idx[1] + tst_size])

        trn_input_data = input_data[trn_idx[0]:trn_idx[1]]
        val_input_data = input_data[val_idx[0]:val_idx[1]]
        trn_val_input_data = input_data[trn_val_idx[0]:trn_val_idx[1]]
        tst_input_data = input_data[tst_idx[0]:tst_idx[1]]

        trn_target_data = target_data[trn_idx[0]:trn_idx[1]]
        val_target_data = target_data[val_idx[0]:val_idx[1]]
        trn_val_target_data = target_data[trn_val_idx[0]:trn_val_idx[1]]
        tst_target_data = target_data[tst_idx[0]:tst_idx[1]]

        trn_t_range = t_range[trn_idx[0]:trn_idx[1]]
        val_t_range = t_range[val_idx[0]:val_idx[1]]
        trn_val_t_range = t_range[trn_val_idx[0]:trn_val_idx[1]]
        tst_t_range = t_range[tst_idx[0]:tst_idx[1]]

        trn_set = SequenceDataset(input_data=trn_input_data, target_data=trn_target_data,
                                  t_range=trn_t_range, seq_len=seq_len)

        val_set = SequenceDataset(input_data=val_input_data, target_data=val_target_data,
                                  t_range=val_t_range, seq_len=seq_len)

        trn_val_set = SequenceDataset(input_data=trn_val_input_data, target_data=trn_val_target_data,
                                      t_range=trn_val_t_range, seq_len=seq_len)

        tst_set = SequenceDataset(input_data=tst_input_data, target_data=tst_target_data,
                                  t_range=tst_t_range, seq_len=seq_len)

        self.train_validation_loader = DataLoader(
            dataset=trn_val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        )
        self.train_loader = DataLoader(
            dataset=trn_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        )
        self.validation_loader = DataLoader(
            dataset=val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        )
        self.test_loader = DataLoader(
            dataset=tst_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        )
