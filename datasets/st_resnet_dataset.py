import numpy as np
import torch

from utils.constants import *

"""
IMPORTANT NOTE: RATHER USE GRID DATASET
"""


# train/test/validation split should be done before hand
class STResNetDataLoader:  # add test and train data and validation set
    def __init__(self, S, E, lc=3, lp=3, lq=3, c=1, p=HOURS_IN_DAY, q=HOURS_IN_WEEK, shuffle=False, trn_tst_split=0.8,
                 trn_val_split=0.8,
                 overlap_in_out=True, norm='none'):
        """
        Args:
            S: Total Sequence size -> (N,rows,cols)
            E: External Info -> (N, n_features)
            lc,lp,lq: number of closeness, period and trend frames respectively
            c,p,q: time step of closeness, period and trend frames respectively

            shuffle: if true shuffles the training batch after each epoch
            trn_tst_split: ratio between train and test data split
            trn_val_split: ratio between train and val data split
            overlap_in_out: if false, make sure that none of the train/validation/test sets' input and output data overlaps
            norm: [minmax | meanstd | none]

        """
        # DATA
        self.S = S
        self.E = E

        # DATA FORMAT PARAMETERS
        self.lc = lc
        self.lp = lp
        self.lq = lq

        self.c = c
        self.p = p
        self.q = q

        # TRAIN/VAL/TEST DATA INDICES
        self.max_t = len(self.S)
        self.min_t = self.lq * self.q  # might not be the min incase p >> q

        self.max_t_val = int(
            np.floor((len(self.S) - self.min_t) * trn_tst_split) + self.min_t)  # max values for validation set
        self.max_t_trn = int(
            np.floor((self.max_t_val - self.min_t) * trn_val_split) + self.min_t)  # max values for train set

        self.trn_times = np.arange(self.min_t, self.max_t_trn, dtype=int)
        self.trn_val_times = np.arange(self.min_t, self.max_t_val, dtype=int)  # train and validation together

        if overlap_in_out == False:
            gap = self.min_t
        else:
            gap = 0

        self.val_times = np.arange(self.max_t_trn + gap, self.max_t_val, dtype=int)
        self.tst_times = np.arange(self.max_t_val + gap, self.max_t, dtype=int)

        # TRAIN MEAN,STD,MIN,MAX FOR SCALING
        trn_data = S[self.trn_times]
        self.trn_mean = trn_data.mean()
        self.trn_std = trn_data.std()
        self.trn_min = trn_data.min()
        self.trn_max = trn_data.max()

        # NORMALIZE THE DATA
        if norm == 'minmax':
            self.S = self.minmax_norm(S)

        if norm == 'meanstd':
            self.S = self.meanstd_norm(S)

        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.trn_times)

        self.current_t = 0

        # TODO: Add t_range so that we get a concept of time
        print('Total data: ', len(S))
        print('Useable data: ', len(self.tst_times) + len(self.trn_times) + len(self.val_times))
        print('Train data: ', len(self.trn_times))
        print('Validation data: ', len(self.val_times))
        print('Test data: ', len(self.tst_times))
        print()

    def getTrainDataLength(self):
        return len(self.trn_times)

    def __call__(self, batch_size):
        """
        Returns a random batch from the training data
        return Dbc,Dbp,Dbq,Dbt,times
        """
        return self.getTrainBatch(batch_size)

    def getData(self, times):
        """
        returns data sequences given a set of Xt times
        return Dbc,Dbp,Dbq,Dbe,Dbt,times
        """
        Dbc = []  # batch of Xc (closeness)
        Dbp = []  # batch of Xp (period)
        Dbq = []  # batch of Xq (trend)
        Dbt = []  # batch of Xt (current time slot)
        Dbe = []  # batch of Et (external factors of current time slot)

        for t in times:
            Sc = self.S[t - self.c * self.lc:t:self.c]
            Sp = self.S[t - self.p * self.lp:t:self.p]
            Sq = self.S[t - self.q * self.lq:t:self.q]
            Dbc.append(Sc)
            Dbp.append(Sp)
            Dbq.append(Sq)

            Xt = self.S[t:t + 1]
            Dbt.append(Xt)
            # if self.E is not None:
            Et = self.E[t:t + 1]
            Dbe.append(Et)

        Dbc = torch.stack(Dbc)
        Dbp = torch.stack(Dbp)
        Dbq = torch.stack(Dbq)
        Dbe = torch.stack(Dbe)
        Dbt = torch.stack(Dbt)

        return Dbc, Dbp, Dbq, Dbe, Dbt, times

        # if self.E != None:
        #     Dbe = torch.stack(Dbe)
        #     return Dbc,Dbp,Dbq,Dbe,Dbt
        # else:
        #     return Dbc,Dbp,Dbq,Dbt

    def getTrainBatch(self, batch_size):  # TODO fix offset
        """
        Returns a random batch from the train set
        format: Dbc,Dbp,Dbq,Dbe,Dbt,times

        """
        # times = np.random.choice(range(self.min_t,self.max_t_trn),size=batch_size,replace=False)
        times = self.trn_times[self.current_t:self.current_t + batch_size]
        if len(times) < 1:
            if self.shuffle:
                np.random.shuffle(self.trn_times)  # Shuffle after every epoch
            self.current_t = 0
            times = self.trn_times[self.current_t:self.current_t + batch_size]
        else:
            self.current_t = self.current_t + batch_size

        return self.getData(times)

    def getTestSet(self):
        """
        returns all data in test set
        format: Dbc,Dbp,Dbq,Dbe,Dbt,times
        """
        #         times = np.arange(self.max_t_val,self.max_t)
        times = self.tst_times

        return self.getData(times)

    def getTrainSet(self):
        """
        returns all data in train set
        format: Dbc,Dbp,Dbq,Dbe,Dbt,times
        """
        #         times = np.arange(self.min_t,self.max_t_trn)
        times = self.trn_times

        return self.getData(times)

    # TODO: Implement cross validation
    def getValidationSet(self):
        """
        returns all data in train set
        format: Dbc,Dbp,Dbq,Dbe,Dbt and times
        """
        #         times = np.arange(self.max_t_trn,self.max_t_val)
        times = self.val_times

        return self.getData(times)

    def reset_current_t(self):
        self.current_t = 0

    def minmax_norm(self, data):
        #       return (data-self.trn_min)/(self.trn_max-self.trn_min)
        return (data - self.trn_min) / (self.trn_max - self.trn_min)

    def meanstd_norm(self, data):
        return (data - self.trn_mean) / (self.trn_std)

    def minmax_norm_r(self, data):
        """
        reverse minmax_norm transformation
        """
        #       return data*(self.trn_max-self.trn_min) + self.trn_min
        return (data) * (self.trn_max - self.trn_min) + self.trn_min

    def meanstd_norm_r(self, data):
        """
        reverse meanstd_norm transformation
        """
        return data * self.trn_std + self.trn_mean
