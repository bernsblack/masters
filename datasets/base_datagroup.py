import logging as log
import numpy as np
import pandas as pd

from models.baseline_models import HistoricAverage
from utils.configs import BaseConf
from utils.preprocessing import Shaper, MinMaxScaler, min_max_scale
from utils.preprocessing import get_hours_per_time_step


class BaseDataGroup:
    def __init__(self, data_path: str, conf: BaseConf):
        """
        BaseDataGroup acts as base class for collections of datasets (training/validation/test)
        The data group loads data from disk, splits in separate sets and normalises according to train set.

        :param data_path: Path to the data folder with all spatial and temporal data.
        :param conf: Config class with pre-set and global values
        """

        with np.load(data_path + "generated_data.npz") as zip_file:  # context helper ensures zip_file is closed
            if conf.use_crime_types:
                log.info(f"loading crimes grids WITH crime types")
                self.crimes = zip_file["crime_types_grids"]
            else:
                log.info(f"loading crimes grids WITHOUT crime types")
                self.crimes = zip_file["crime_grids"]  # original crime counts

            self.crime_feature_indices = list(zip_file["crime_feature_indices"])

            self.t_range: pd.DatetimeIndex = pd.read_pickle(data_path + "t_range.pkl")
            log.info(f"\tt_range: {np.shape(self.t_range)} {self.t_range[0]} -> {self.t_range[-1]}")

            # freqstr = self.t_range.freqstr
            # if freqstr == "D":
            #     freqstr = "24H"
            # if freqstr == "H":
            #     freqstr = "1H"
            # hours_per_time_step = int(freqstr[:freqstr.find("H")])  # time step in hours
            hours_per_time_step = get_hours_per_time_step(self.t_range.freq)  # time step in hours
            time_steps_per_day = 24 / hours_per_time_step

            self.offset_year = int(365 * time_steps_per_day)

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
            self.total_offset = self.seq_len + self.offset_year

            target_len = self.total_len - self.total_offset

            # OPTION 1:
            # train/test split sizes is dependent on total_offset - means we can't compare models easily
            # tst_size = int(target_len * conf.tst_ratio)
            # val_size = int(target_len * conf.val_ratio)
            # trn_size = int(target_len - tst_size - val_size)
            # trn_val_size = trn_size + val_size

            # OPTION 2:
            # # train/test split sizes is dependent on TEST_SET_SIZE_DAYS - sizes will be same for all models: makes it comparable
            # tst_size = int((conf.tst_ratio/conf.tst_ratio) * TEST_SET_SIZE_DAYS * time_steps_per_day)
            # val_size = int((conf.val_ratio/conf.tst_ratio) * tst_size)
            # trn_size = int(((1 - conf.val_ratio - conf.tst_ratio)/conf.tst_ratio) * tst_size)
            # trn_val_size = trn_size + val_size

            # OPTION 3:
            # constant test set size and varying train and validation sizes
            # test set can be set to be specific # days instead of just a year
            tst_size = int(conf.test_set_size_days * time_steps_per_day)
            # tst_size = int(HOURS_IN_YEAR / hours_per_time_step)  # conf.test_set_size # the last year in the data set
            trn_val_size = target_len - tst_size
            val_size = int(conf.val_ratio * trn_val_size)
            trn_size = trn_val_size - val_size

            log.info(f"\ttarget_len:\t{target_len}\t({(100 * target_len / target_len):.3f}%)")
            log.info(f"\ttrn_val_size:\t{trn_val_size}\t({(100 * trn_val_size / target_len):.3f}%)")
            log.info(f"\ttrn_size:\t{trn_size}\t({(100 * trn_size / target_len):.3f}%)")
            log.info(f"\tval_size:\t{val_size}\t({(100 * val_size / target_len):.3f}%)")
            log.info(f"\ttst_size:\t{tst_size} \t({(100 * tst_size / target_len):.3f}%)")

            #  start and stop t_index of each dataset - can be used outside of loader/group
            # runs -> val_set, trn_set, tst_set: trn and tst set are more correlated
            self.tst_indices = np.array([self.total_len - tst_size, self.total_len])
            self.trn_val_indices = np.array([self.tst_indices[0] - trn_val_size,
                                             self.tst_indices[0]])  # used to train model with validation set included
            if conf.train_set_first:
                # train first option
                self.val_indices = np.array([self.tst_indices[0] - val_size, self.tst_indices[0]])
                self.trn_indices = np.array([self.val_indices[0] - trn_size, self.val_indices[0]])
            else:
                # val first option
                self.trn_indices = np.array([self.tst_indices[0] - trn_size, self.tst_indices[0]])
                self.val_indices = np.array([self.trn_indices[0] - val_size, self.trn_indices[0]])

            # used to create the shaper so that all datasets have got values in them
            tmp_trn_crimes = self.crimes[self.trn_indices[0]:self.trn_indices[1], 0:1]
            tmp_val_crimes = self.crimes[self.val_indices[0]:self.val_indices[1], 0:1]
            tmp_tst_crimes = self.crimes[self.tst_indices[0]:self.tst_indices[1], 0:1]

            # only using shaper on test crimes - ensures loaders line up
            shaper_crimes = np.max(tmp_tst_crimes, axis=0, keepdims=True)
            # shaper_crimes = self.crimes
            # shaper_crimes = np.max(tmp_tst_crimes, axis=0, keepdims=True) * np.max(tmp_val_crimes, axis=0,
            #                                                                        keepdims=True) * np.max(
            #     tmp_trn_crimes, axis=0, keepdims=True)

            assert np.sum(shaper_crimes) > 0
            # fit crime data to shaper
            self.shaper = Shaper(data=shaper_crimes,
                                 conf=conf)

            # squeeze all spatially related data
            # reshaped (N, C, H, W) -> (N, C, L)
            self.crimes = self.shaper.squeeze(self.crimes)

            # cap any values above conf.cap_crime_percentile percentile as outliers
            if conf.cap_crime_percentile > 0:
                cap = np.percentile(self.crimes.flatten(), conf.cap_crime_percentile)
                self.crimes[self.crimes > cap] = cap

            # add tract count to crime grids - done separately in case we do not want crime types or arrests
            tract_count_grids = zip_file["tract_count_grids"]
            tract_count_grids = self.shaper.squeeze(tract_count_grids)

            self.crimes = np.concatenate((self.crimes, tract_count_grids), axis=1)
            self.crime_feature_indices.append("tract_count")

            # used for display purposes - and grouping similarly active cells together.
            self.sorted_indices = np.argsort(self.crimes[:, 0].sum(0))[::-1]

            # (reminder) ensure that the self.crimes are not scaled to -1,1 before
            self.targets = np.copy(self.crimes[1:, 0:1])  # only check for totals > 0
            self.labels = np.copy(self.crimes[1:, 0:1])  # only check for totals > 0
            self.labels[self.labels > 0] = 1

            self.crimes = self.crimes[:-1]

            # if conf.use_classification:  # use this if statement in the training loops to determine if we should use y_class or y_countj
            #     self.targets[self.targets > 0] = 1

            self.total_crimes = np.expand_dims(self.crimes[:, 0].sum(1), axis=1)

            self.time_vectors = zip_file["time_vectors"][1:]  # already normalised - time vector of future crime
            # self.weather_vectors = zip_file["weather_vectors"][1:]  # get weather for target date
            self.x_range = zip_file["x_range"]
            self.y_range = zip_file["y_range"]
            self.t_range = self.t_range[1:]

            self.demog_grid = self.shaper.squeeze(zip_file["demog_grid"])
            self.street_grid = self.shaper.squeeze(zip_file["street_grid"])

        self.tst_indices[0] = self.tst_indices[0] - self.total_offset
        self.val_indices[0] = self.val_indices[0] - self.total_offset
        self.trn_indices[0] = self.trn_indices[0] - self.total_offset
        self.trn_val_indices[0] = self.trn_val_indices[0] - self.total_offset

        self.trn_val_t_range = self.t_range[self.trn_val_indices[0]:self.trn_val_indices[1]]
        self.trn_t_range = self.t_range[self.trn_indices[0]:self.trn_indices[1]]
        self.val_t_range = self.t_range[self.val_indices[0]:self.val_indices[1]]
        self.tst_t_range = self.t_range[self.tst_indices[0]:self.tst_indices[1]]

        self.log_norm_scale = conf.log_norm_scale
        # split and normalise the crime data
        # log2 scaling to count data to make less disproportionate
        # self.crimes = np.round(np.log2(1 + self.crimes)) # by round the values we cannot inverse to get the original counts
        # self.targets = np.round(np.log2(1 + self.targets)) # by round the values we cannot inverse to get the original counts
        if self.log_norm_scale:
            self.crimes = np.log2(1 + self.crimes)
            self.targets = np.log2(1 + self.targets)

        assert len(self.crimes) == len(self.targets)

        # get historic average on the log2+1 normed values
        if conf.use_historic_average:
            if time_steps_per_day == 1:
                ha = HistoricAverage(step=7)
            elif time_steps_per_day == 24:
                ha = HistoricAverage(step=24)
            else:
                ha = HistoricAverage(step=1)

            historic_average = ha.fit_transform(self.crimes[:, 0:1])
            self.crimes = np.concatenate((self.crimes, historic_average), axis=1)
            self.crime_feature_indices.append("historic_average")

        self.crime_scaler = MinMaxScaler(feature_range=(0, 1))
        # should be axis of the channels - only fit scaler on training data
        # self.crime_scaler.fit(self.crimes[self.trn_val_indices[0]:self.trn_val_indices[1]], axis=1) # scale only on training and validation data
        # scale on all data because training set is not the same for grid and cell data groups because of sequence offsets
        self.crime_scaler.fit(self.crimes, axis=conf.scale_axis)
        self.crimes = self.crime_scaler.transform(self.crimes)
        self.trn_val_crimes = self.crimes[self.trn_val_indices[0]:self.trn_val_indices[1]]
        self.trn_crimes = self.crimes[self.trn_indices[0]:self.trn_indices[1]]
        self.val_crimes = self.crimes[self.val_indices[0]:self.val_indices[1]]
        self.tst_crimes = self.crimes[self.tst_indices[0]:self.tst_indices[1]]

        # targets
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        # should be axis of the channels - only fit scaler on training data
        # use train and val sizes to determine the scale - validation set forms part of the training set technically
        # self.target_scaler.fit(self.targets[self.trn_val_indices[0]:self.trn_val_indices[1]], axis=1) # scale only on training and validation data
        self.target_scaler.fit(self.targets,
                               axis=1)  # scale on all data because training set is not the same for grid and cell data groups because of sequence offsets
        self.targets = self.target_scaler.transform(self.targets)

        self.trn_val_targets = self.targets[self.trn_val_indices[0]:self.trn_val_indices[1]]
        self.trn_targets = self.targets[self.trn_indices[0]:self.trn_indices[1]]
        self.val_targets = self.targets[self.val_indices[0]:self.val_indices[1]]
        self.tst_targets = self.targets[self.tst_indices[0]:self.tst_indices[1]]

        self.trn_val_labels = self.labels[self.trn_val_indices[0]:self.trn_val_indices[1]]
        self.trn_labels = self.labels[self.trn_indices[0]:self.trn_indices[1]]
        self.val_labels = self.labels[self.val_indices[0]:self.val_indices[1]]
        self.tst_labels = self.labels[self.tst_indices[0]:self.tst_indices[1]]

        # total crimes - added separately because all spatial
        # if self.log_norm_scale:
        #     self.total_crimes = np.log2(1 + self.total_crimes)
        self.total_crimes_scaler = MinMaxScaler(feature_range=(0, 1))
        # should be axis of the channels - only fit scaler on training data
        self.total_crimes_scaler.fit(self.total_crimes[self.trn_indices[0]:self.trn_indices[1]], axis=1)
        self.total_crimes = self.total_crimes_scaler.transform(self.total_crimes)
        self.trn_val_total_crimes = self.total_crimes[self.trn_val_indices[0]:self.trn_val_indices[1]]
        self.trn_total_crimes = self.total_crimes[self.trn_indices[0]:self.trn_indices[1]]
        self.val_total_crimes = self.total_crimes[self.val_indices[0]:self.val_indices[1]]
        self.tst_total_crimes = self.total_crimes[self.tst_indices[0]:self.tst_indices[1]]

        # time_vectors is already scaled between 0 and 1
        self.trn_val_time_vectors = self.time_vectors[self.trn_val_indices[0]:self.trn_val_indices[1]]
        self.trn_time_vectors = self.time_vectors[self.trn_indices[0]:self.trn_indices[1]]
        self.val_time_vectors = self.time_vectors[self.val_indices[0]:self.val_indices[1]]
        self.tst_time_vectors = self.time_vectors[self.tst_indices[0]:self.tst_indices[1]]

        # # splitting and normalisation of weather data
        # self.weather_vector_scaler = MinMaxScaler(feature_range=(0, 1))
        # # self.weather_vector_scaler.fit(self.weather_vectors[self.trn_index[0]:self.trn_index[1]], axis=1)  # norm with trn data
        # self.weather_vector_scaler.fit(self.weather_vectors, axis=1)  # norm with all data
        # self.weather_vectors = self.weather_vector_scaler.transform(self.weather_vectors)
        # trn_val_weather_vectors = self.weather_vectors[self.trn_val_indices[0]:self.trn_val_indices[1]]
        # trn_weather_vectors = self.weather_vectors[self.trn_indices[0]:self.trn_indices[1]]
        # val_weather_vectors = self.weather_vectors[self.val_indices[0]:self.val_indices[1]]
        # tst_weather_vectors = self.weather_vectors[self.tst_indices[0]:self.tst_indices[1]]

        # normalise space dependent data - using min_max_scale - no need to save train data norm values
        self.demog_grid = min_max_scale(data=self.demog_grid, feature_range=(0, 1), axis=1)
        self.street_grid = min_max_scale(data=self.street_grid, feature_range=(0, 1), axis=1)

        self.training_validation_set = None
        self.training_set = None
        self.validation_set = None
        self.testing_set = None
