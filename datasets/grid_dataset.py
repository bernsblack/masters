import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import logging as log
from utils.preprocessing import Shaper, MinMaxScaler, minmax_scale
from utils.configs import BaseConf


# todo add set of valid / living sells / to be able to sample the right ones
class GridDataGroup:
    """
    GridDataGroup class acts as a collection of datasets (training/validation/test)
    The data group loads data from disk, splits in separate sets and normalises according to train set.
    Crime count related data is first scaled using f(x) = log2(1 + x) and then scaled between -1 and 1.
    """

    def __init__(self, data_path, conf):
        """
        Args:
            data_path (string): Path to the data folder with all spatial and temporal data.
        """
        # dont use function to get values each time - create join and add table
        # [√] number of incidents of crime occurrence by sampling point in 2013 (1-D)
        # [√] number of incidents of crime occurrence by census tract in 2013 (1-D)
        # [√] number of incidents of crime occurrence by census tract yesterday (1-D).
        # [√] number of incidents of crime occurrence by date in 2013 (1-D)

        zip_file = np.load(data_path + "generated_data.npz")

        # print info on the read data
        log.info("Data shapes of files in generated_data.npz")
        for k, v in zip_file.items():
            log.info(f"\t{k} shape {np.shape(v)}")
        t_range = pd.read_pickle(data_path + "t_range.pkl")
        log.info(f"\tt_range shape {np.shape(t_range)}")

        # shaper can bes used to shape the data into teh same format - ignoring cells where nothing ever takes place,
        # e.g. lakes, air-strips, etc.
        self.shaper = Shaper(data=zip_file["crime_types_grids"])

        # squeeze all spatially related data
        self.crimes = zip_file["crime_types_grids"]

        self.targets = np.copy(self.crimes)[1:, 0]  # only check for totals > 0
        self.targets[self.targets > 0] = 1

        self.crimes = self.crimes[:-1]
        self.total_crimes = np.expand_dims(self.crimes[:, 0].sum(1), axis=1)

        self.time_vectors = zip_file["time_vectors"][1:]  # already normalised
        self.weather_vectors = zip_file["weather_vectors"][1:]  # get weather for target date
        self.x_range = zip_file["x_range"]
        self.y_range = zip_file["y_range"]
        self.t_range = t_range[:-1]

        self.demog_grid = zip_file["demog_grid"]
        self.street_grid = zip_file["street_grid"]

        self.seq_len = conf.seq_len
        self.total_len = len(self.crimes)  # length of the whole time series

        #  sanity check if time matches up with our grids
        if len(self.t_range) - 1 != len(self.crimes):
            log.error("time series and time range lengths do not match up -> " +
                      "len(self.t_range) != len(self.crime_types_grids)")
            raise RuntimeError(f"len(self.t_range) - 1 {len(self.t_range) - 1} != len(self.crimes) {len(self.crimes)} ")

        val_size = int(self.total_len * conf.val_ratio)
        tst_size = int(self.total_len * conf.tst_ratio)

        trn_index = (0, self.total_len - tst_size - val_size)
        val_index = (trn_index[1] - self.seq_len, self.total_len - tst_size)
        tst_index = (val_index[1] - self.seq_len, self.total_len)

        trn_t_range = self.t_range[trn_index[0]:trn_index[1]]
        val_t_range = self.t_range[val_index[0]:val_index[1]]
        tst_t_range = self.t_range[tst_index[0]:tst_index[1]]

        # split and normalise the crime data
        # log2 scaling to count data to make less disproportionate
        self.crimes = np.log2(1 + self.crimes)
        self.crime_scaler = MinMaxScaler(feature_range=(-1, 1))
        # should be axis of the channels - only fit scaler on training data
        self.crime_scaler.fit(self.crimes[trn_index[0]:trn_index[1]], axis=1)
        self.crimes = self.crime_scaler.transform(self.crimes)
        trn_crimes = self.crimes[trn_index[0]:trn_index[1]]
        val_crimes = self.crimes[val_index[0]:val_index[1]]
        tst_crimes = self.crimes[tst_index[0]:tst_index[1]]

        # targets
        trn_targets = self.targets[trn_index[0]:trn_index[1]]
        val_targets = self.targets[val_index[0]:val_index[1]]
        tst_targets = self.targets[tst_index[0]:tst_index[1]]

        # total crimes - added separately because all spatial
        self.total_crimes = np.log2(1 + self.total_crimes)
        self.total_crimes_scaler = MinMaxScaler(feature_range=(-1, 1))
        # should be axis of the channels - only fit scaler on training data
        self.total_crimes_scaler.fit(self.total_crimes[trn_index[0]:trn_index[1]], axis=1)
        self.total_crimes = self.total_crimes_scaler.transform(self.total_crimes)
        trn_total_crimes = self.total_crimes[trn_index[0]:trn_index[1]]
        val_total_crimes = self.total_crimes[val_index[0]:val_index[1]]
        tst_total_crimes = self.total_crimes[tst_index[0]:tst_index[1]]

        # time_vectors is already scaled between 0 and 1
        trn_time_vectors = self.time_vectors[trn_index[0]:trn_index[1]]
        val_time_vectors = self.time_vectors[val_index[0]:val_index[1]]
        tst_time_vectors = self.time_vectors[tst_index[0]:tst_index[1]]

        # splitting and normalisation of weather data
        self.weather_vector_scaler = MinMaxScaler(feature_range=(-1, 1))
        # todo maybe scale weather with all data so it's season independent
        # self.weather_vector_scaler.fit(self.weather_vectors[trn_index[0]:trn_index[1]], axis=1)  # norm with trn data
        self.weather_vector_scaler.fit(self.weather_vectors, axis=1)  # norm with all data
        self.weather_vectors = self.weather_vector_scaler.transform(self.weather_vectors)
        trn_weather_vectors = self.weather_vectors[trn_index[0]:trn_index[1]]
        val_weather_vectors = self.weather_vectors[val_index[0]:val_index[1]]
        tst_weather_vectors = self.weather_vectors[tst_index[0]:tst_index[1]]

        # normalise space dependent data - using minmax_scale - no need to save train data norm values
        # todo concat the space dependent values so long
        self.demog_grid = minmax_scale(data=self.demog_grid, feature_range=(-1, 1), axis=1)
        self.street_grid = minmax_scale(data=self.street_grid, feature_range=(-1, 1), axis=1)

        # todo check if time independent  values aren't copied every time
        self.training_set = GridDataset(
            crimes=trn_crimes,
            targets=trn_targets,
            total_crimes=trn_total_crimes,
            t_range=trn_t_range,
            time_vectors=trn_time_vectors,
            weather_vectors=trn_weather_vectors,
            demog_grid=self.demog_grid,
            street_grid=self.street_grid,
            seq_len=self.seq_len,
        )

        self.validation_set = GridDataset(
            crimes=val_crimes,
            targets=val_targets,
            total_crimes=val_total_crimes,
            t_range=val_t_range,
            time_vectors=val_time_vectors,
            weather_vectors=val_weather_vectors,
            demog_grid=self.demog_grid,
            street_grid=self.street_grid,
            seq_len=self.seq_len,
        )

        self.testing_set = GridDataset(
            crimes=tst_crimes,
            targets=tst_targets,
            total_crimes=tst_total_crimes,
            t_range=tst_t_range,
            time_vectors=tst_time_vectors,
            weather_vectors=tst_weather_vectors,
            demog_grid=self.demog_grid,
            street_grid=self.street_grid,
            seq_len=self.seq_len,
        )


class GridDataset(Dataset):
    """
    # todo (bernard): add documentation
    """

    def __init__(
            self,
            crimes,  # time and space dependent
            targets,  # time and space dependent
            total_crimes,  # time dependent
            t_range,  # time dependent
            time_vectors,  # time dependent
            weather_vectors,  # time dependent
            demog_grid,  # space dependent
            street_grid,  # space dependent
            seq_len,
    ):
        self.seq_len = seq_len

        #  time and space dependent
        self.crimes = crimes
        self.targets = targets
        self.t_size, _, self.y_size, self.x_size = np.shape(self.crimes)

        #  space dependent
        self.demog_grid = demog_grid
        self.street_grid = street_grid

        # time dependent
        self.total_crimes = total_crimes
        self.time_vectors = time_vectors
        self.weather_vectors = weather_vectors
        self.t_range = t_range

    def __len__(self):
        """Denotes the total number of samples"""
        return self.t_size - self.seq_len

    def __getitem__(self, index):
        start_index = index
        stop_index = index + self.seq_len if self.seq_len > 0 else index + 1

        # todo teacher forcing - if we are using this then we need to return sequence of targets
        target_grid = self.targets[start_index:stop_index]
        crime_grid = self.crimes[start_index:stop_index]
        demog_grid = self.demog_grid
        street_grid = self.street_grid

        time_vec = self.time_vectors[start_index:stop_index]
        weather_vec = self.weather_vectors[start_index:stop_index]

        # stack all grids on channel axis
        # grids = np.concatenate((crime_grid, demog_grid, street_grid), axis=0)

        tmp_feats = np.concatenate((time_vec, weather_vec, self.total_crimes[start_index]),
                                   axis=-1)  # todo add historical values
        return crime_grid, demog_grid, street_grid, tmp_feats, target_grid
