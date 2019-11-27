import logging as log

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from models.baseline_models import HistoricAverage
from utils.configs import BaseConf
from utils.constants import TEST_SET_SIZE_DAYS
from utils.preprocessing import Shaper, MinMaxScaler, minmax_scale
from utils.utils import if_none


class BaseDataGroup:
    def __init__(self, data_path: str, conf: BaseConf):
        """
        :param data_path: Path to the data folder with all spatial and temporal data.
        :param conf: Config class with pre-set and global values
        """

        with np.load(data_path + "generated_data.npz") as zip_file:  # context helper ensures zip_file is closed
            if conf.use_crime_types:
                self.crimes = zip_file["crime_types_grids"]
            else:
                self.crimes = zip_file["crime_grids"]

            self.t_range = pd.read_pickle(data_path + "t_range.pkl")
            log.info(f"\tt_range shape {np.shape(self.t_range)}")

            freqstr = self.t_range.freqstr
            if freqstr == "H":
                freqstr = "1H"
            time_step_hrs = int(freqstr[:freqstr.find("H")])  # time step in hours
            time_step_days = int(24 / time_step_hrs)

            self.offset_year = int(365 * time_step_days)

            self.seq_len = conf.seq_len
            self.pad_width = conf.pad_width
            self.total_len = len(self.crimes)  # length of the whole time series

            #  sanity check if time matches up with our grids
            if len(self.t_range) - 1 != len(self.crimes):
                log.error("time series and time range lengths do not match up -> " +
                          "len(self.t_range) != len(self.crimes)")
                raise RuntimeError(
                    f"len(self.t_range) - 1 {len(self.t_range) - 1} != len(self.crimes) {len(self.crimes)} ")

            #  split the data into ratios - size represent the targets sizes not the number of time steps
            total_offset = self.seq_len + self.offset_year

            # target_len = self.total_len - total_offset
            tst_size = int((conf.tst_ratio / conf.tst_ratio) * TEST_SET_SIZE_DAYS * time_step_days)  # int(target_len * conf.tst_ratio)
            val_size = int((conf.val_ratio / conf.tst_ratio) * tst_size)  # int(target_len * conf.val_ratio)
            trn_size = int(((1 - conf.val_ratio - conf.tst_ratio) / conf.tst_ratio) * tst_size) # int(target_len - tst_size - val_size)

            #  start and stop t_index of each dataset - can be used outside of loader/group
            # runs -> val_set, trn_set, tst_set: trn and tst set are more correlated
            self.tst_indices = np.array([self.total_len - tst_size, self.total_len])
            self.trn_indices = np.array([self.tst_indices[0] - trn_size, self.tst_indices[0]])
            self.val_indices = np.array([self.trn_indices[0] - val_size, self.trn_indices[0]])

            # used to create the shaper so that all datasets have got values in them
            tmp_trn_crimes = self.crimes[self.trn_indices[0]:self.trn_indices[1], 0:1]
            tmp_val_crimes = self.crimes[self.val_indices[0]:self.val_indices[1], 0:1]
            tmp_tst_crimes = self.crimes[self.tst_indices[0]:self.tst_indices[1], 0:1]
            shaper_crimes = np.max(tmp_trn_crimes, axis=0, keepdims=True) * \
                            np.max(tmp_val_crimes, axis=0, keepdims=True) * \
                            np.max(tmp_tst_crimes, axis=0, keepdims=True)

            # fit crime data to shaper
            self.shaper = Shaper(data=shaper_crimes,
                                 conf=conf)

            # squeeze all spatially related data
            # reshaped (N, C, H, W) -> (N, C, L)
            self.crimes = self.shaper.squeeze(self.crimes)
            # add tract count to crime grids - done separately in case we do not want crime types or arrests
            tract_count_grids = zip_file["tract_count_grids"]
            tract_count_grids = self.shaper.squeeze(tract_count_grids)

            self.crimes = np.concatenate((self.crimes, tract_count_grids), axis=1)

            # used for display purposes - and grouping similarly active cells together.
            self.sorted_indices = np.argsort(self.crimes[:, 0].sum(0))[::-1]

            # (reminder) ensure that the self.crimes are not scaled to -1,1 before
            self.targets = np.copy(self.crimes[1:, 0:1])  # only check for totals > 0
            if conf.use_classification:
                self.targets[self.targets > 0] = 1

            self.crimes = self.crimes[:-1]
            self.total_crimes = np.expand_dims(self.crimes[:, 0].sum(1), axis=1)

            self.time_vectors = zip_file["time_vectors"][1:]  # already normalised - time vector of future crime
            # self.weather_vectors = zip_file["weather_vectors"][1:]  # get weather for target date
            self.x_range = zip_file["x_range"]
            self.y_range = zip_file["y_range"]
            self.t_range = self.t_range[1:]

            self.demog_grid = self.shaper.squeeze(zip_file["demog_grid"])
            self.street_grid = self.shaper.squeeze(zip_file["street_grid"])

        self.tst_indices[0] = self.tst_indices[0] - total_offset
        self.val_indices[0] = self.val_indices[0] - total_offset
        self.trn_indices[0] = self.trn_indices[0] - total_offset

        self.trn_t_range = self.t_range[self.trn_indices[0]:self.trn_indices[1]]
        self.val_t_range = self.t_range[self.val_indices[0]:self.val_indices[1]]
        self.tst_t_range = self.t_range[self.tst_indices[0]:self.tst_indices[1]]

        # split and normalise the crime data
        # log2 scaling to count data to make less disproportionate
        self.crimes = np.log2(1 + self.crimes)
        self.targets = np.log2(1 + self.targets)

        # get historic average on the log2+1 normed values
        if conf.use_historic_average:
            ha = HistoricAverage(step=time_step_days)
            historic_average = ha.fit_transform(+ self.crimes[:, 0:1])
            self.crimes = np.concatenate((self.crimes, historic_average), axis=1)

        self.crime_scaler = MinMaxScaler(feature_range=(0, 1))
        # should be axis of the channels - only fit scaler on training data
        self.crime_scaler.fit(self.crimes[self.trn_indices[0]:self.trn_indices[1]], axis=1)
        self.crimes = self.crime_scaler.transform(self.crimes)
        self.trn_crimes = self.crimes[self.trn_indices[0]:self.trn_indices[1]]
        self.val_crimes = self.crimes[self.val_indices[0]:self.val_indices[1]]
        self.tst_crimes = self.crimes[self.tst_indices[0]:self.tst_indices[1]]

        # targets
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        # should be axis of the channels - only fit scaler on training data
        self.target_scaler.fit(self.targets[self.trn_indices[0]:self.trn_indices[1]], axis=1)
        self.targets = self.target_scaler.transform(self.targets)

        self.trn_targets = self.targets[self.trn_indices[0]:self.trn_indices[1]]
        self.val_targets = self.targets[self.val_indices[0]:self.val_indices[1]]
        self.tst_targets = self.targets[self.tst_indices[0]:self.tst_indices[1]]

        # total crimes - added separately because all spatial
        self.total_crimes = np.log2(1 + self.total_crimes)
        self.total_crimes_scaler = MinMaxScaler(feature_range=(0, 1))
        # should be axis of the channels - only fit scaler on training data
        self.total_crimes_scaler.fit(self.total_crimes[self.trn_indices[0]:self.trn_indices[1]], axis=1)
        self.total_crimes = self.total_crimes_scaler.transform(self.total_crimes)
        self.trn_total_crimes = self.total_crimes[self.trn_indices[0]:self.trn_indices[1]]
        self.val_total_crimes = self.total_crimes[self.val_indices[0]:self.val_indices[1]]
        self.tst_total_crimes = self.total_crimes[self.tst_indices[0]:self.tst_indices[1]]

        # time_vectors is already scaled between 0 and 1
        self.trn_time_vectors = self.time_vectors[self.trn_indices[0]:self.trn_indices[1]]
        self.val_time_vectors = self.time_vectors[self.val_indices[0]:self.val_indices[1]]
        self.tst_time_vectors = self.time_vectors[self.tst_indices[0]:self.tst_indices[1]]

        # # splitting and normalisation of weather data
        # self.weather_vector_scaler = MinMaxScaler(feature_range=(0, 1))
        # # self.weather_vector_scaler.fit(self.weather_vectors[self.trn_index[0]:self.trn_index[1]], axis=1)  # norm with trn data
        # self.weather_vector_scaler.fit(self.weather_vectors, axis=1)  # norm with all data
        # self.weather_vectors = self.weather_vector_scaler.transform(self.weather_vectors)
        # trn_weather_vectors = self.weather_vectors[self.trn_indices[0]:self.trn_indices[1]]
        # val_weather_vectors = self.weather_vectors[self.val_indices[0]:self.val_indices[1]]
        # tst_weather_vectors = self.weather_vectors[self.tst_indices[0]:self.tst_indices[1]]

        # normalise space dependent data - using minmax_scale - no need to save train data norm values
        self.demog_grid = minmax_scale(data=self.demog_grid, feature_range=(0, 1), axis=1)
        self.street_grid = minmax_scale(data=self.street_grid, feature_range=(0, 1), axis=1)

        self.training_set = None

        self.validation_set = None

        self.testing_set = None

