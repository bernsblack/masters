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

        self.target_shape = (len(self.input_data) - self.seq_len, self.target_data.shape[-1])

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
    def __init__(self, input_data, target_data, t_range,
                 seq_len=30, batch_size=1, shuffle=True, val_ratio=0.2, tst_ratio=0.2, num_workers=0):
        """
        data ndarray (N,d)
        """
        assert len(input_data) == len(target_data)

        total_len = len(input_data)

        tst_size = int(total_len * tst_ratio)

        trn_val_size = total_len - tst_size
        val_size = int(trn_val_size * val_ratio)
        trn_size = trn_val_size - val_size

        trn_idx = np.array([0, trn_size])
        val_idx = np.array([trn_idx[1] - seq_len, trn_idx[1] + val_size])
        trn_val_idx = np.array([0, val_idx[1]])
        tst_idx = np.array([val_idx[1] - seq_len, val_idx[1] + tst_size])

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
