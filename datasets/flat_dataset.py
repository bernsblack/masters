import logging as log

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils.configs import BaseConf
from utils.preprocessing import Shaper, MinMaxScaler, minmax_scale
from utils.utils import if_none

"""
Not really faster as the FlatDataset
"""


class BaseDataGroup:  # using the builder pattern
    """
    Simple class to build up a data-group
    Crime set is always necessary - other meta data can be added later on - makes simplifying models easier
    """

    def __init__(self, data_path: str, conf: BaseConf):
        # Init optionals
        self.time_vectors = None
        self.weather_vectors = None
        self.demog_grid = None
        self.street_grid = None

        self.weather_vector_scaler = None
        self.trn_time_vectors = None
        self.val_time_vectors = None
        self.tst_time_vectors = None

        self.trn_weather_vectors = None
        self.val_weather_vectors = None
        self.tst_weather_vectors = None

        # with context helper ensures zip_file is closed
        with np.load(data_path + "generated_data.npz") as zip_file:
            self.data_path = data_path
            self.conf = conf

            t_range = pd.read_pickle(data_path + "t_range.pkl")
            log.info(f"\tt_range shape {np.shape(t_range)}")

            if conf.use_crime_types:
                self.crimes = zip_file["crime_types_grids"]
            else:
                self.crimes = zip_file["crime_grids"]

            self.shaper = Shaper(data=self.crimes,
                                 threshold=conf.shaper_threshold,
                                 top_k=conf.shaper_top_k)

            # squeeze all spatially related data
            # reshaped into (N, C, L) from (N, C, H, W)
            self.crimes = self.shaper.squeeze(self.crimes)
            # used for display purposes - and grouping similarly active cells together.

            self.sorted_indices = np.argsort(self.crimes[:, 0].sum(0))[::-1]  # todo move to shaper

            self.targets = np.copy(self.crimes[1:, 0])  # only check for totals > 0
            # todo (reminder) ensure that the self.crimes are not scaled to -1,1 before
            self.targets[self.targets > 0] = 1

            self.crimes = self.crimes[:-1]

            self.x_range = zip_file["x_range"]
            self.y_range = zip_file["y_range"]
            self.t_range = t_range[1:]  # todo: t_range match the time of the target

            self.seq_len = conf.seq_len
            self.total_len = len(self.crimes)  # length of the whole time series

            #  sanity check if time matches up with our grids
            if len(self.t_range) - 1 != len(self.crimes):
                log.error("time series and time range lengths do not match up -> " +
                          "len(self.t_range) != len(self.crime_types_grids)")
                raise RuntimeError(
                    f"len(self.t_range) - 1 {len(self.t_range) - 1} != len(self.crimes) {len(self.crimes)} ")

            #  split the data into ratios
            val_size = int(self.total_len * conf.val_ratio)
            tst_size = int(self.total_len * conf.tst_ratio)

            #  start and stop t_index of each dataset - can be used outside of loader/group
            self.trn_indices = (0, self.total_len - tst_size - val_size)
            self.val_indices = (self.trn_indices[1] - self.seq_len, self.total_len - tst_size)
            self.tst_indices = (self.val_indices[1] - self.seq_len, self.total_len)

            self.trn_t_range = self.t_range[self.trn_indices[0]:self.trn_indices[1]]
            self.val_t_range = self.t_range[self.val_indices[0]:self.val_indices[1]]
            self.tst_t_range = self.t_range[self.tst_indices[0]:self.tst_indices[1]]

            # split and normalise the crime data
            # log2 scaling to count data to make less disproportionate
            self.crimes = np.log2(1 + self.crimes)  # todo: add hot-encoded option
            self.crime_scaler = MinMaxScaler(feature_range=(0, 1))
            # should be axis of the channels - only fit scaler on training data
            self.crime_scaler.fit(self.crimes[self.trn_indices[0]:self.trn_indices[1]], axis=1)
            self.crimes = self.crime_scaler.transform(self.crimes)
            self.trn_crimes = self.crimes[self.trn_indices[0]:self.trn_indices[1]]
            self.val_crimes = self.crimes[self.val_indices[0]:self.val_indices[1]]
            self.tst_crimes = self.crimes[self.tst_indices[0]:self.tst_indices[1]]

            # targets
            self.trn_targets = self.targets[self.trn_indices[0]:self.trn_indices[1]]
            self.val_targets = self.targets[self.val_indices[0]:self.val_indices[1]]
            self.tst_targets = self.targets[self.tst_indices[0]:self.tst_indices[1]]

    def with_time_vectors(self):
        with np.load(self.data_path + "generated_data.npz") as zip_file:
            self.time_vectors = zip_file["time_vectors"][1:]  # already normalised - time vector of future crime
            # time_vectors is already scaled between 0 and 1
            self.trn_time_vectors = self.time_vectors[self.trn_indices[0]:self.trn_indices[1]]
            self.val_time_vectors = self.time_vectors[self.val_indices[0]:self.val_indices[1]]
            self.tst_time_vectors = self.time_vectors[self.tst_indices[0]:self.tst_indices[1]]

    def with_weather_vectors(self):
        with np.load(self.data_path + "generated_data.npz") as zip_file:
            self.weather_vectors = zip_file["weather_vectors"][1:]  # get weather for target date
            # splitting and normalisation of weather data
            self.weather_vector_scaler = MinMaxScaler(feature_range=(0, 1))

            # scaling weather with all data so it's season independent, not just the training data
            self.weather_vector_scaler.fit(self.weather_vectors, axis=1)  # norm with all data
            self.weather_vectors = self.weather_vector_scaler.transform(self.weather_vectors)
            self.trn_weather_vectors = self.weather_vectors[self.trn_indices[0]:self.trn_indices[1]]
            self.val_weather_vectors = self.weather_vectors[self.val_indices[0]:self.val_indices[1]]
            self.tst_weather_vectors = self.weather_vectors[self.tst_indices[0]:self.tst_indices[1]]

    def with_demog_grid(self):
        with np.load(self.data_path + "generated_data.npz") as zip_file:
            self.demog_grid = self.shaper.squeeze(zip_file["demog_grid"])
            self.demog_grid = minmax_scale(data=self.demog_grid, feature_range=(0, 1), axis=1)

    def with_street_grid(self):
        with np.load(self.data_path + "generated_data.npz") as zip_file:
            self.street_grid = self.shaper.squeeze(zip_file["street_grid"])
            self.street_grid = minmax_scale(data=self.street_grid, feature_range=(0, 1), axis=1)


# todo add set of valid / living sells / to be able to sample the right ones
class FlatDataGroup:
    """
    FlatDataGroup class acts as a collection of datasets (training/validation/test)
    The data group loads data from disk, splits in separate sets and normalises according to train set.
    Crime count related data is first scaled using f(x) = log2(1 + x) and then scaled between -1 and 1.
    The data group class also handles reshaping of data.
    """

    def __init__(self, data_path: str, conf: BaseConf):
        """
        Args:
            data_path (string): Path to the data folder with all spatial and temporal data.
        """
        # todo (note): dont use function to get values each time - create join and add table
        # [√] number of incidents of crime occurrence by sampling point in 2013 (1-D)
        # [√] number of incidents of crime occurrence by census tract in 2013 (1-D)
        # [√] number of incidents of crime occurrence by census tract yesterday (1-D).
        # [√] number of incidents of crime occurrence by date in 2013 (1-D)

        with np.load(data_path + "generated_data.npz") as zip_file:  # context helper ensures zip_file is closed
            # print info on the read data
            log.info("Data shapes of files in generated_data.npz")
            for k, v in zip_file.items():
                log.info(f"\t{k} shape {np.shape(v)}")
            t_range = pd.read_pickle(data_path + "t_range.pkl")
            log.info(f"\tt_range shape {np.shape(t_range)}")

            if conf.use_crime_types:
                self.crimes = zip_file["crime_types_grids"]
            else:
                self.crimes = zip_file["crime_grids"]

            # should make shaper on only the testing data so that we don't look at testing cells where nothing actually
            # happens
            self.shaper = Shaper(data=self.crimes,
                                 threshold=conf.shaper_threshold,
                                 top_k=conf.shaper_top_k)

            # squeeze all spatially related data
            # reshaped into (N, C, L) from (N, C, H, W)
            self.crimes = self.shaper.squeeze(self.crimes)
            # used for display purposes - and grouping similarly active cells together.
            self.sorted_indices = np.argsort(self.crimes[:, 0].sum(0))[::-1]

            self.targets = np.copy(self.crimes[1:, 0:1])  # only check for totals > 0
            self.targets[
                self.targets > 0] = 1  # todo (reminder) ensure that the self.crimes are not scaled to -1,1 before

            self.crimes = self.crimes[:-1]
            self.total_crimes = np.expand_dims(self.crimes[:, 0].sum(1), axis=1)  # todo move to where data is generated

            self.time_vectors = zip_file["time_vectors"][1:]  # already normalised - time vector of future crime
            self.weather_vectors = zip_file["weather_vectors"][1:]  # get weather for target date
            self.x_range = zip_file["x_range"]
            self.y_range = zip_file["y_range"]
            self.t_range = t_range[1:]  # todo: t_range match the time of the target

            self.demog_grid = self.shaper.squeeze(zip_file["demog_grid"])
            self.street_grid = self.shaper.squeeze(zip_file["street_grid"])

        self.seq_len = conf.seq_len
        self.total_len = len(self.crimes)  # length of the whole time series

        #  sanity check if time matches up with our grids
        if len(self.t_range) - 1 != len(self.crimes):
            log.error("time series and time range lengths do not match up -> " +
                      "len(self.t_range) != len(self.crimes)")
            raise RuntimeError(f"len(self.t_range) - 1 {len(self.t_range) - 1} != len(self.crimes) {len(self.crimes)} ")

        #  split the data into ratios
        val_size = int(self.total_len * conf.val_ratio)
        tst_size = int(self.total_len * conf.tst_ratio)

        #  start and stop t_index of each dataset - can be used outside of loader/group
        self.trn_indices = (0, self.total_len - tst_size - val_size)
        self.val_indices = (self.trn_indices[1] - self.seq_len, self.total_len - tst_size)
        self.tst_indices = (self.val_indices[1] - self.seq_len, self.total_len)

        trn_t_range = self.t_range[self.trn_indices[0]:self.trn_indices[1]]
        val_t_range = self.t_range[self.val_indices[0]:self.val_indices[1]]
        tst_t_range = self.t_range[self.tst_indices[0]:self.tst_indices[1]]

        # split and normalise the crime data
        # log2 scaling to count data to make less disproportionate
        self.crimes = np.log2(1 + self.crimes)  # todo: add hot-encoded option
        self.crime_scaler = MinMaxScaler(feature_range=(0, 1))
        # should be axis of the channels - only fit scaler on training data
        print("shape crimes_train -> ", self.crimes[self.trn_indices[0]:self.trn_indices[1]].shape)
        print("shape crimes -> ", self.crimes.shape)

        self.crime_scaler.fit(self.crimes[self.trn_indices[0]:self.trn_indices[1]], axis=1)
        self.crimes = self.crime_scaler.transform(self.crimes)
        trn_crimes = self.crimes[self.trn_indices[0]:self.trn_indices[1]]
        val_crimes = self.crimes[self.val_indices[0]:self.val_indices[1]]
        tst_crimes = self.crimes[self.tst_indices[0]:self.tst_indices[1]]

        # targets
        trn_targets = self.targets[self.trn_indices[0]:self.trn_indices[1]]
        val_targets = self.targets[self.val_indices[0]:self.val_indices[1]]
        tst_targets = self.targets[self.tst_indices[0]:self.tst_indices[1]]

        # total crimes - added separately because all spatial
        self.total_crimes = np.log2(1 + self.total_crimes)
        self.total_crimes_scaler = MinMaxScaler(feature_range=(0, 1))
        # should be axis of the channels - only fit scaler on training data
        self.total_crimes_scaler.fit(self.total_crimes[self.trn_indices[0]:self.trn_indices[1]], axis=1)
        self.total_crimes = self.total_crimes_scaler.transform(self.total_crimes)
        trn_total_crimes = self.total_crimes[self.trn_indices[0]:self.trn_indices[1]]
        val_total_crimes = self.total_crimes[self.val_indices[0]:self.val_indices[1]]
        tst_total_crimes = self.total_crimes[self.tst_indices[0]:self.tst_indices[1]]

        # time_vectors is already scaled between 0 and 1
        trn_time_vectors = self.time_vectors[self.trn_indices[0]:self.trn_indices[1]]
        val_time_vectors = self.time_vectors[self.val_indices[0]:self.val_indices[1]]
        tst_time_vectors = self.time_vectors[self.tst_indices[0]:self.tst_indices[1]]

        # splitting and normalisation of weather data
        self.weather_vector_scaler = MinMaxScaler(feature_range=(0, 1))
        # todo maybe scale weather with all data so it's season independent
        # self.weather_vector_scaler.fit(self.weather_vectors[self.trn_index[0]:self.trn_index[1]], axis=1)  # norm with trn data
        self.weather_vector_scaler.fit(self.weather_vectors, axis=1)  # norm with all data
        self.weather_vectors = self.weather_vector_scaler.transform(self.weather_vectors)
        trn_weather_vectors = self.weather_vectors[self.trn_indices[0]:self.trn_indices[1]]
        val_weather_vectors = self.weather_vectors[self.val_indices[0]:self.val_indices[1]]
        tst_weather_vectors = self.weather_vectors[self.tst_indices[0]:self.tst_indices[1]]

        # normalise space dependent data - using minmax_scale - no need to save train data norm values
        self.demog_grid = minmax_scale(data=self.demog_grid, feature_range=(0, 1), axis=1)
        self.street_grid = minmax_scale(data=self.street_grid, feature_range=(0, 1), axis=1)

        # target index - also the index given to the
        # todo check if time independent values aren't copied every time
        # todo dependency injection of the different types of datasets
        self.training_set = FlatDataset(
            crimes=trn_crimes,
            targets=trn_targets,
            total_crimes=trn_total_crimes,
            t_range=trn_t_range,  # t_range is matched to the target index
            time_vectors=trn_time_vectors,
            weather_vectors=trn_weather_vectors,
            demog_grid=self.demog_grid,
            street_grid=self.street_grid,
            seq_len=self.seq_len,
            shaper=self.shaper,
        )

        self.validation_set = FlatDataset(
            crimes=val_crimes,
            targets=val_targets,
            total_crimes=val_total_crimes,
            t_range=val_t_range,  # t_range is matched to the target index
            time_vectors=val_time_vectors,
            weather_vectors=val_weather_vectors,
            demog_grid=self.demog_grid,
            street_grid=self.street_grid,
            seq_len=self.seq_len,
            shaper=self.shaper,
        )

        self.testing_set = FlatDataset(
            crimes=tst_crimes,
            targets=tst_targets,
            total_crimes=tst_total_crimes,
            t_range=tst_t_range,  # t_range is matched to the target index
            time_vectors=tst_time_vectors,
            weather_vectors=tst_weather_vectors,
            demog_grid=self.demog_grid,
            street_grid=self.street_grid,
            seq_len=self.seq_len,
            shaper=self.shaper,
        )


class FlatDataset(Dataset):
    """
    Flat datasets operate on flattened data where the map/grid of data has been reshaped
    from (N,C,H,W) -> (N,C,L). These re-shaped values have also been formatted/squeezed to
    ignore all locations where there never occurs any crimes
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
            shaper,
    ):
        self.seq_len = seq_len

        self.crimes = crimes
        self.targets = targets
        self.t_size, _, self.l_size = np.shape(self.crimes)
        self.total_crimes = total_crimes

        self.demog_grid = demog_grid
        self.street_grid = street_grid

        self.time_vectors = time_vectors
        self.weather_vectors = weather_vectors  # remember weather should be the info of the next time step
        self.t_range = t_range

        self.shaper = shaper

        #  [min_index, max_index) are limits of flattened targets
        self.max_index = self.t_size * self.l_size
        self.min_index = self.seq_len * self.l_size
        self.len = self.min_index - self.min_index  # todo WARNING WON'T LINE UP WITH BATCH LOADERS IF SUB-SAMPLING

        self.shape = self.t_size, self.l_size  # used when saving the model results

    def __len__(self):
        """Denotes the total number of samples"""
        return self.len  # todo WARNING WON'T LINE UP WITH BATCH LOADERS IF SUB-SAMPLING

    def __getitem__(self, index):  # tensorflow version of the pasring function
        # when using no teacher forcing
        # target = self.targets[t+self.seq_len, :, l]
        # todo add all other data - should be done in data-generation?
        # [√] number of incidents of crime occurrence by sampling point in 2013 (1-D)
        # [√] number of incidents of crime occurrence by census tract in 2013 (1-D)
        # [√] number of incidents of crime occurrence by census tract yesterday (1-D).
        # [√] number of incidents of crime occurrence by date in 2013 (1-D)
        """Generates one sample of data"""
        if isinstance(index, slice):
            index = range(if_none(index.start, self.min_index),
                          if_none(index.stop, self.max_index),
                          if_none(index.step, 1))

        indices = np.array([index]).flatten()  # brackets and flatten caters where index is a single number
        # todo review the code below - list are bad find a better way!!
        stack_spc_feats = []
        stack_tmp_feats = []
        stack_env_feats = []
        stack_targets = []

        result_indices = []

        for i in indices:
            if not (self.min_index <= i < self.max_index):
                raise IndexError(f"index value {i} is not in range({self.min_index},{self.max_index})")

            t_index, l_index = np.unravel_index(i, (self.t_size, self.l_size))
            t_start = t_index - self.seq_len
            t_stop = t_index

            # return shape should be (seq_len, batch, input_size)

            crime_vec = self.crimes[t_start:t_stop, :, l_index]
            # todo add more historical values
            crime_vec = np.concatenate((crime_vec, self.total_crimes[t_start:t_stop]), axis=-1)

            time_vec = self.time_vectors[t_start:t_stop]
            weather_vec = self.weather_vectors[t_start:t_stop]
            demog_vec = self.demog_grid[:, :, l_index]
            street_vec = self.street_grid[:, :, l_index]
            tmp_vec = np.concatenate((time_vec, weather_vec, crime_vec), axis=-1)  # todo add more historical values
            # todo teacher forcing - if we are using this then we need to return sequence of targets
            target_vec = self.targets[t_start:t_stop, :, l_index]

            stack_spc_feats.append(demog_vec)  # todo stacking the same grid might cause memory issues
            stack_env_feats.append(street_vec)  # todo stacking the same grid might cause memory issues

            stack_tmp_feats.append(tmp_vec)
            stack_targets.append(target_vec)

            result_indices.append((t_index, l_index))

        # spc_feats: [demog_vec]
        # env_feats: [street_vec]
        # tmp_feats: [time_vec, weather_vec, crime_vec]
        # targets: [targets]
        spc_feats = np.stack(stack_spc_feats)
        tmp_feats = np.stack(stack_tmp_feats)
        env_feats = np.stack(stack_env_feats)
        targets = np.stack(stack_targets)

        spc_feats = np.swapaxes(spc_feats, 0, 1)
        tmp_feats = np.swapaxes(tmp_feats, 0, 1)
        env_feats = np.swapaxes(env_feats, 0, 1)
        targets = np.swapaxes(targets, 0, 1)

        # output shapes should be - (seq_len, batch_size,, n_feats)
        return result_indices, spc_feats, tmp_feats, env_feats, targets

    ## todo have this as the base class - create new classes where we're just over loading
