import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import logging as log
from utils.preprocessing import Shaper, MinMaxScaler, minmax_scale
from utils.configs import BaseConf


# todo add set of valid / living sells / to be able to sample the right ones
class CrimeDataGroup:
    """
    Collection of datasets (training/validation/test)
    """

    def __init__(self, data_path, conf):
        """
        Args:
            data_path (string): Path to the data folder with all spatial and temporal data.
        """
        # todo normalise the data
        # todo cap the crime grids at a certain level - instead use np.log2(1 + x) to normalise
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

        self.shaper = Shaper(data=zip_file["crime_types_grids"])

        # squeeze all spatially related data
        self.crimes = self.shaper.squeeze(zip_file["crime_types_grids"])  # reshaped into (N, C, L)

        self.targets = np.copy(self.crimes)[1:, 0]  # only check for totals > 0
        self.targets[self.targets > 0] = 1

        self.crimes = self.crimes[:-1]

        self.total_crimes = self.crimes[:, 0].sum(1)

        self.time_vectors = zip_file["time_vectors"][:-1]  # already normalised
        self.weather_vectors = zip_file["weather_vectors"][1:]  # get weather for target date
        self.x_range = zip_file["x_range"]
        self.y_range = zip_file["y_range"]
        self.t_range = t_range[:-1]

        self.demog_grid = self.shaper.squeeze(zip_file["demog_grid"])
        self.street_grid = self.shaper.squeeze(zip_file["street_grid"])

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
        val_index = (trn_index[1], self.total_len - tst_size)
        tst_index = (val_index[1], self.total_len)

        trn_times = self.t_range[trn_index[0]:trn_index[1]]
        val_times = self.t_range[val_index[0]:val_index[1]]
        tst_times = self.t_range[tst_index[0]:tst_index[1]]

        # split and normalise the crime data
        # log2 scaling to count data to make less disproportionate
        self.crimes = np.log2(1 + self.crimes)
        trn_crimes = self.crimes[trn_index[0]:trn_index[1]]
        val_crimes = self.crimes[val_index[0]:val_index[1]]
        tst_crimes = self.crimes[tst_index[0]:tst_index[1]]
        self.crime_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.crime_scaler.fit(trn_crimes, axis=1)  # should be axis of the channels
        trn_crimes = self.crime_scaler.transform(trn_crimes)
        val_crimes = self.crime_scaler.transform(val_crimes)
        tst_crimes = self.crime_scaler.transform(tst_crimes)

        # targets
        trn_targets = self.targets[trn_index[0]:trn_index[1]]
        val_targets = self.targets[val_index[0]:val_index[1]]
        tst_targets = self.targets[tst_index[0]:tst_index[1]]

        # split and normalise the time vector data # todo concat the weather data
        trn_time_vectors = self.time_vectors[trn_index[0]:trn_index[1]]
        val_time_vectors = self.time_vectors[val_index[0]:val_index[1]]
        tst_time_vectors = self.time_vectors[tst_index[0]:tst_index[1]]
        # is already scaled between 0 and 1
        # self.time_vector_scaler = MinMaxScaler(feature_range=(-1, 1))
        # self.time_vector_scaler.fit(trn_time_vectors, axis=1)
        # trn_time_vectors = self.time_vector_scaler.transform(trn_time_vectors)
        # val_time_vectors = self.time_vector_scaler.transform(val_time_vectors)
        # tst_time_vectors = self.time_vector_scaler.transform(tst_time_vectors)

        # splitting and normalisation of weather data
        trn_weather_vectors = self.weather_vectors[trn_index[0]:trn_index[1]]
        val_weather_vectors = self.weather_vectors[val_index[0]:val_index[1]]
        tst_weather_vectors = self.weather_vectors[tst_index[0]:tst_index[1]]
        self.weather_vector_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.weather_vector_scaler.fit(trn_weather_vectors, axis=1)
        trn_weather_vectors = self.weather_vector_scaler.transform(trn_weather_vectors)
        val_weather_vectors = self.weather_vector_scaler.transform(val_weather_vectors)

        # normalise space dependent data - using minmax_scale - no need to save train data norm values
        # todo concat the space dependent values so long
        self.demog_grid = minmax_scale(data=self.demog_grid, feature_range=(-1, 1), axis=1)
        self.street_grid = minmax_scale(data=self.street_grid, feature_range=(-1, 1), axis=1)

        # todo check if time independent  values aren't copied every time
        self.training_set = CrimeDataset(
            crimes=trn_crimes,
            targets=trn_targets,
            t_range=trn_times,
            time_vectors=trn_time_vectors,
            weather_vectors=trn_weather_vectors,
            demog_grid=self.demog_grid,
            street_grid=self.street_grid,
            seq_len=self.seq_len,
        )

        self.validation_set = CrimeDataset(
            crimes=val_crimes,
            targets=val_targets,
            t_range=val_times,
            time_vectors=val_time_vectors,
            weather_vectors=val_weather_vectors,
            demog_grid=self.demog_grid,
            street_grid=self.street_grid,
            seq_len=self.seq_len,
        )

        self.testing_set = CrimeDataset(
            crimes=tst_crimes,
            targets=tst_targets,
            t_range=tst_times,
            time_vectors=tst_time_vectors,
            weather_vectors=tst_weather_vectors,
            demog_grid=self.demog_grid,
            street_grid=self.street_grid,
            seq_len=self.seq_len,
        )


class CrimeDataset(Dataset):
    def __init__(
            self,
            crimes,  # time and space dependent
            targets,  # time and space dependent
            t_range,  # time dependent
            time_vectors,  # time dependent
            weather_vectors,  # time dependent
            demog_grid,  # space dependent
            street_grid,  # space dependent
            seq_len,
    ):
        # normalize
        self.seq_len = seq_len

        self.crimes = crimes
        self.targets = targets
        self.t_size, _, self.l_size = np.shape(self.crimes)

        self.demog_grid = demog_grid
        self.street_grid = street_grid

        self.time_vectors = time_vectors
        self.weather_vectors = weather_vectors
        self.t_range = t_range

        self.total_crimes = self.crimes[:, 0].sum(1)  # or self.crime_types_grids[0].sum(1).sum(1)

    # todo add weather -  remember weather should be the info of the next time step

    def __len__(self):
        """Denotes the total number of samples"""
        return self.t_size * self.l_size

    def __getitem__(self, index):
        #  todo try without seq len first - like fnn
        # todo separate vectors in tuples to make input to the models easier
        # when using no teacher forcing
        # target = self.targets[t+self.seq_len, :, l]
        # todo add all other data
        # [√] number of incidents of crime occurrence by sampling point in 2013 (1-D)
        # [√] number of incidents of crime occurrence by census tract in 2013 (1-D)
        # [√] number of incidents of crime occurrence by census tract yesterday (1-D).
        # [√] number of incidents of crime occurrence by date in 2013 (1-D)
        """Generates one sample of data"""
        try:
            t_index, l_index = np.unravel_index(index, (self.t_size, self.l_size))
            # todo teacher forcing - if we are using this then we need to return sequence of targets
        except TypeError:
            log.warning(f"Multi index unraveling\ntype(index): {type(index)} | index: {index}")
            index_slice = slice(index)
            indices = np.arange(index_slice.start, index_slice.stop, index_slice.step)
            t_index, l_index = np.unravel_index(indices, (self.t_size, self.l_size))

        # todo teacher forcing - if we are using this then we need to return sequence of targets
        target = self.targets[t_index, l_index]
        crime_vec = self.crimes[t_index, :, l_index]  # todo add historical values

        time_vec = self.time_vectors[t_index]
        weather_vec = self.weather_vectors[t_index]
        demog_vec = self.demog_grid[0, :, l_index]
        street_vec = self.street_grid[0, :, l_index]

        env_feats = street_vec
        spc_feats = demog_vec
        tmp_feats = np.concatenate((time_vec, weather_vec, crime_vec), axis=-1)  # todo add historical values
        return spc_feats, tmp_feats, env_feats, target
