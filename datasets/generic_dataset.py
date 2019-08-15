import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import logging as log


# todo import shaper

class GenericDataset(Dataset):
    """
    Characterizes a dataset for PyTorch
    """

    def __init__(self, features, labels):
        """Initialization"""
        self.labels = labels
        self.feats = features

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.feats)

    def __getitem__(self, index):
        """Generates one sample of data"""
        X = self.feats[index]
        y = self.labels[index]

        return X, y


# todo add set of valid / living sells / to be able to sample the right ones
class CrimeDataGroup:
    """
    Object that contains the test/validation/training sets
    """

    def __init__(self, data_path, conf):
        """
        Args:
            data_path (string): Path to the data folder with all spatial and temporal data.
        """
        # todo normalise the data
        # todo cap the crime grids at a certain level - instead use np.log2(1 + x) to normalise
        # todo flatten the crime grids and normalise according to the features
        # dont use function to get values each time - create join and add table
        # [√] number of incidents of crime occurrence by sampling point in 2013 (1-D)
        # [√] number of incidents of crime occurrence by census tract in 2013 (1-D)
        # [√] number of incidents of crime occurrence by census tract yesterday (1-D).
        # [√] number of incidents of crime occurrence by date in 2013 (1-D)

        zip_file = np.load(data_path + "generated_data.npz")
        # used in determine what columns in crime_types_grids represent
        self.crime_feature_indices = zip_file["crime_feature_indices"]
        self.crime_types_grids = zip_file["crime_types_grids"]  # todo rather use sparse matrix??

        # getting the total sum over the whole data set
        total_crimes_index = 0  # self.crime_feature_indices["TOTAL"] # todo dict implementation

        self.total_crimes = self.crime_types_grids[:, total_crimes_index].sum(1).sum(1)  # total crimes per time step
        self.tract_count_grids = zip_file["tract_count_grids"]
        self.demog_grid = zip_file["demog_grid"]
        self.street_grid = zip_file["street_grid"]
        self.time_vectors = zip_file["time_vectors"]
        self.x_range = zip_file["x_range"]
        self.y_range = zip_file["y_range"]
        self.t_range = pd.read_pickle(data_path + "t_range.pkl")

        self.seq_len = conf.seq_len

        self.total_len = len(self.t_range)  # length of the whole time series

        #  sanity check if time matches up with our grids
        if len(self.t_range) != len(self.crime_types_grids):
            log.error("time series and time range lengths do not match up -> " +
                      "len(self.t_range) != len(self.crime_types_grids)")
            raise RuntimeError

        val_size = int(self.total_len * conf.val_ratio)
        tst_size = int(self.total_len * conf.tst_ratio)

        trn_index = (0, self.total_len - tst_size - val_size)
        val_index = (trn_index[1], self.total_len - tst_size)
        tst_index = (val_index[1], self.total_len)

        # todo check if time independent  values aren't copied every time
        self.training_set = CrimeDataset(
            crime_types_grids=self.crime_types_grids[trn_index[0]:trn_index[1]],
            t_range=self.t_range[trn_index[0]:trn_index[1]],
            time_vectors=self.time_vectors[trn_index[0]:trn_index[1]],
            tract_count_grids=self.tract_count_grids,
            demog_grid=self.demog_grid,
            street_grid=self.street_grid,
            x_range=self.x_range,
            y_range=self.y_range,
            crime_feature_indices=self.crime_feature_indices,
            seq_len=self.seq_len,
        )

        self.validation_set = CrimeDataset(
            crime_types_grids=self.crime_types_grids[val_index[0]:val_index[1]],
            t_range=self.t_range[val_index[0]:val_index[1]],
            time_vectors=self.time_vectors[val_index[0]:val_index[1]],
            tract_count_grids=self.tract_count_grids,
            demog_grid=self.demog_grid,
            street_grid=self.street_grid,
            x_range=self.x_range,
            y_range=self.y_range,
            crime_feature_indices=self.crime_feature_indices,
            seq_len=self.seq_len,
        )

        self.testing_set = CrimeDataset(
            crime_types_grids=self.crime_types_grids[tst_index[0]:tst_index[1]],
            t_range=self.t_range[tst_index[0]:tst_index[1]],
            time_vectors=self.time_vectors[tst_index[0]:tst_index[1]],
            tract_count_grids=self.tract_count_grids,
            demog_grid=self.demog_grid,
            street_grid=self.street_grid,
            x_range=self.x_range,
            y_range=self.y_range,
            crime_feature_indices=self.crime_feature_indices,
            seq_len=self.seq_len,
        )


from sklearn.preprocessing import MinMaxScaler


class CrimeDataset(Dataset):
    def __init__(
            self,
            crime_types_grids,  # time and space dependent
            t_range,  # time dependent
            time_vectors,  # time dependent
            tract_count_grids,  # space dependent
            demog_grid,  # space dependent
            street_grid,  # space dependent
            x_range,  # space dependent
            y_range,  # space dependent
            crime_feature_indices,
            seq_len,


    ):
        self.seq_len = seq_len
        # self.crime_types_grids = crime_types_grids     #  (N,C,H,W)
        self.shaper = Shaper(data=crime_types_grids)  # (N,C,L) where `L=H*W
        self.crime_types_flat = self.shaper.squeeze(crime_types_grids)  # (N,C,L)

        self.targets = np.copy(self.crime_types_flat[1:])
        self.targets[self.targets > 1] = 1

        self.crime_types_flat = self.crime_types_flat[:-1]

        self.tract_count_grids = tract_count_grids  # mostly for kang and kang
        self.demog_grid = demog_grid
        self.street_grid = street_grid
        # self.x_range = x_range
        # self.y_range = y_range
        self.time_vectors = time_vectors[:-1]
        self.t_range = t_range[:-1]  # time of the crime not the prediction
        # todo add weather -  remember weather should be the info of the next time step
        # self.y_size = len(y_range)
        # self.x_size = len(x_range)
        self.t_size, _, self.l_size = np.shape(self.crime_types_flat)

        self.total_crimes = self.crime_types_flat[:, 0].sum(1)  # or self.crime_types_grids[0].sum(1).sum(1)
        self.total_crimes = np.log2(1 + self.total_crimes)

        # todo add min -1 max +1 norm
        # normalisation should be set with the train set min max - scaler should be fed in


        self.crime_feature_indices = crime_feature_indices

    def __len__(self):
        """Denotes the total number of samples"""
        return (self.t_size - self.seq_len) * self.l_size

    def __getitem__(self, index):
        """Generates one sample of data"""
        t, l = np.unravel_index(index, (self.t_size, self.l_size))

        # todo teacher forcing - if we are using this then we need to return sequence of targets
        feats = self.crime_types_flat[t:t + self.seq_len, :, l]
        target = self.targets[t:t + self.seq_len, :, l]

        # when using no teacher forcing
        # target = self.targets[t+self.seq_len, :, l]

        return feats, target
