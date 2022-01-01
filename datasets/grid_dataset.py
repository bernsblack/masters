import logging as log
import numpy as np
import pandas as pd
from pandas.tseries.offsets import Hour as OffsetHour
from torch.utils.data import Dataset

from constants.date_time import DatetimeFreq
from models.baseline_models import PeriodicAverage
from utils.configs import BaseConf
from utils.preprocessing import Shaper, MinMaxScaler, min_max_scale, get_hours_per_time_step
from utils.types.arrays import ArrayNCHW, ArrayNC, ArrayHWC, ArrayNHW
from utils.utils import if_none

HOUR_NANOS = OffsetHour().nanos


# todo: split train/val/test first then make shaper with prod of the max of each set - ensuring one crime for all cells
class GridDataGroup:
    """
    GridDataGroup class acts as a collection of datasets (training/validation/test)
    The data group loads data from disk, splits in separate sets and normalises according to train set.
    Crime count related data is first scaled using f(x) = log2(1 + x) and then scaled between 0 and 1.
    The data group class also handles reshaping of data.
    """

    def __init__(self, data_path: str, conf: BaseConf):
        """
        Crime should be in format (N,H,W)
        :param data_path: path toe file
        :param conf:
        """
        log.info('Initialising Grid Data Group')

        with np.load(data_path + "generated_data.npz") as zip_file:  # context helper ensures zip_file is closed
            if conf.use_crime_types:
                self.crimes: ArrayNCHW = zip_file["crime_types_grids"]
                assert len(self.crimes.shape) == 4, f"self.crimes.shape is {self.crimes.shape} not ArrayNCHW"
            else:
                self.crimes: ArrayNCHW = zip_file["crime_grids"]
                assert len(self.crimes.shape) == 4, f"self.crimes.shape is {self.crimes.shape} not ArrayNCHW"
            self.crime_feature_indices = list(zip_file["crime_feature_indices"])

            self.total_len: int = len(self.crimes)  # length of the whole time series

            self.t_range: pd.DatetimeIndex = pd.read_pickle(data_path + "t_range.pkl")
            log.info(f"\tt_range: {np.shape(self.t_range)} {self.t_range[0]} -> {self.t_range[-1]}")

            # todo change to use DatetimeFreq.convert
            freqstr = DatetimeFreq.convert(self.t_range)

            hours_per_time_step = get_hours_per_time_step(self.t_range.freq)  # time step in hours
            time_steps_per_day = 24 / hours_per_time_step

            if freqstr == DatetimeFreq.Week:
                self.step_c: int = 1
                self.step_p: int = 5
                self.step_q: int = 10
            elif freqstr == DatetimeFreq.Day:
                self.step_c: int = 1
                self.step_p: int = 7  # jumps in days
                self.step_q: int = 14
            else:
                self.step_c: int = 1
                self.step_p: int = int(24 / hours_per_time_step)  # jumps in days
                self.step_q: int = int(168 / hours_per_time_step)  # jumps in weeks -  maximum offset

            self.n_steps_c: int = conf.n_steps_c  # 3
            self.n_steps_p: int = conf.n_steps_p  # 3
            self.n_steps_q: int = conf.n_steps_q  # 3

            #  split the data into ratios - size represent the targets sizes not the number of time steps
            self.total_offset: int = self.step_q * self.n_steps_q
            # self.total_offset = np.max([self.step_q * self.n_steps_q, 365 * time_steps_per_day])

            target_len = self.total_len - self.total_offset

            # OPTION 1:
            # train/test split sizes is dependent on total_offset - means we can't compare models easily
            # tst_size = int(target_len * conf.tst_ratio)
            # val_size = int(target_len * conf.val_ratio)
            # trn_size = int(target_len - tst_size - val_size)
            # trn_val_size = trn_size + val_size

            # OPTION 2:
            # # train/test split sizes is dependent on TEST_SET_SIZE_DAYS - sizes will be same for all models: makes it comparable
            # tst_size = int((conf.tst_ratio/conf.tst_ratio) * TEST_SET_SIZE_DAYS * time_step_days)
            # val_size = int((conf.val_ratio/conf.tst_ratio) * tst_size)
            # trn_size = int(((1 - conf.val_ratio - conf.tst_ratio)/conf.tst_ratio) * tst_size)
            # trn_val_size = trn_size + val_size

            # OPTION 3:
            # constant test set size and varying train and validation sizes
            # test set can be set to be specific # days instead of just a year
            tst_size = int(24 * conf.test_set_size_days / hours_per_time_step)
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
            self.trn_val_indices = np.array([self.tst_indices[0] - trn_val_size, self.tst_indices[0]])
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

            # fit crime data to shaper - only used to squeeze results when calculating the results
            self.shaper = Shaper(
                data=shaper_crimes,
                conf=conf,
            )

            squeezed_crimes = self.shaper.squeeze(self.crimes)

            # cap any values above conf.cap_crime_percentile percentile as outliers
            if conf.cap_crime_percentile > 0:
                cap = np.percentile(squeezed_crimes.flatten(), conf.cap_crime_percentile)
                squeezed_crimes[squeezed_crimes > cap] = cap

                log.info(f"capping max crime values to {cap} which is percentile {conf.cap_crime_percentile}")

            # get periodic average on the log2+1 normed values
            if conf.use_periodic_average:
                if time_steps_per_day == 1:
                    ha = PeriodicAverage(step=7)
                elif time_steps_per_day == 24:
                    ha = PeriodicAverage(step=24)
                else:
                    ha = PeriodicAverage(step=1)

                periodic_average = ha.fit_transform(squeezed_crimes[:, 0:1])
                squeezed_crimes = np.concatenate((squeezed_crimes, periodic_average), axis=1)

            self.crimes = self.shaper.unsqueeze(
                dense_data=squeezed_crimes,
            )  # ensures we ignore areas that do not have crime through out and ensures crime data in different data loaders are exactly the same
            # add tract count to crime grids - done separately in case we do not want crime types or arrests
            # tract_count_grids = zip_file["tract_count_grids"]
            # self.crimes = np.concatenate((self.crimes, tract_count_grids), axis=1)

            # (reminder) ensure that the self.crimes are not scaled to -1,1 before
            self.targets = np.copy(self.crimes[1:, 0:1])  # only check for totals > 0 # N,1,H,W
            self.labels = np.copy(self.crimes[1:, 0:1])  # only check for totals > 0
            self.labels[self.labels > 0] = 1

            self.crimes = self.crimes[:-1]

            self.time_vectors = zip_file["time_vectors"][1:]  # already normalised - time vector of future crime

            self.t_range = self.t_range[1:]

            self.demog_grid = zip_file["demog_grid"]
            self.street_grid = zip_file["street_grid"]

        assert len(self.crimes) == len(self.targets)

        #  sanity check if time matches up with our grids
        if len(self.t_range) - 1 != len(self.crimes):
            log.error("time series and time range lengths do not match up -> " +
                      "len(self.t_range) != len(self.crimes)")
            raise RuntimeError(f"len(self.t_range) - 1 {len(self.t_range) - 1} != len(self.crimes) {len(self.crimes)} ")

        self.tst_indices[0] = self.tst_indices[0] - self.total_offset
        self.val_indices[0] = self.val_indices[0] - self.total_offset
        self.trn_indices[0] = self.trn_indices[0] - self.total_offset
        self.trn_val_indices[0] = self.trn_val_indices[0] - self.total_offset

        trn_val_t_range = self.t_range[self.trn_val_indices[0]:self.trn_val_indices[1]]
        trn_t_range = self.t_range[self.trn_indices[0]:self.trn_indices[1]]
        val_t_range = self.t_range[self.val_indices[0]:self.val_indices[1]]
        tst_t_range = self.t_range[self.tst_indices[0]:self.tst_indices[1]]

        self.log_norm_scale = conf.log_norm_scale
        if self.log_norm_scale:
            # self.crimes = np.round(np.log2(1 + self.crimes)) # by rounding we cannot retrieve original count
            # self.targets = np.round(np.log2(1 + self.targets)) # by rounding we cannot retrieve original count
            self.crimes = np.log2(1 + self.crimes)
            self.targets = np.log2(1 + self.targets)

        self.crime_scaler = MinMaxScaler(feature_range=(0, 1))
        # self.crime_scaler.fit(self.crimes[self.trn_indices[0]:self.trn_indices[1]], axis=1) # scale only on training and validation data
        # scale on all data because training set is not the same for grid and cell data groups because of sequence offsets
        self.crime_scaler.fit(self.crimes, axis=conf.scale_axis)
        self.crimes = self.crime_scaler.transform(self.crimes)
        trn_val_crimes = self.crimes[self.trn_val_indices[0]:self.trn_val_indices[1]]
        trn_crimes = self.crimes[self.trn_indices[0]:self.trn_indices[1]]
        val_crimes = self.crimes[self.val_indices[0]:self.val_indices[1]]
        tst_crimes = self.crimes[self.tst_indices[0]:self.tst_indices[1]]

        # targets
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        # self.target_scaler.fit(self.targets[self.trn_indices[0]:self.trn_indices[1]], axis=1) # scale only on training and validation data
        self.target_scaler.fit(self.targets,
                               axis=1)  # scale on all data because training set is not the same for grid and cell data groups because of sequence offsets
        self.targets = self.target_scaler.transform(self.targets)  # N,1,H,W

        trn_val_targets = self.targets[self.trn_val_indices[0]:self.trn_val_indices[1]]
        trn_targets = self.targets[self.trn_indices[0]:self.trn_indices[1]]
        val_targets = self.targets[self.val_indices[0]:self.val_indices[1]]
        tst_targets = self.targets[self.tst_indices[0]:self.tst_indices[1]]

        # time_vectors is already scaled between 0 and 1
        trn_val_time_vectors = self.time_vectors[self.trn_val_indices[0]:self.trn_val_indices[1]]
        trn_time_vectors = self.time_vectors[self.trn_indices[0]:self.trn_indices[1]]
        val_time_vectors = self.time_vectors[self.val_indices[0]:self.val_indices[1]]
        tst_time_vectors = self.time_vectors[self.tst_indices[0]:self.tst_indices[1]]

        trn_val_labels = self.labels[self.trn_val_indices[0]:self.trn_val_indices[1]]
        trn_labels = self.labels[self.trn_indices[0]:self.trn_indices[1]]
        val_labels = self.labels[self.val_indices[0]:self.val_indices[1]]
        tst_labels = self.labels[self.tst_indices[0]:self.tst_indices[1]]

        # # splitting and normalisation of weather data
        # self.weather_vector_scaler = MinMaxScaler(feature_range=(0, 1))
        # # self.weather_vector_scaler.fit(self.weather_vectors[self.trn_index[0]:self.trn_index[1]], axis=1)  # norm with trn data
        # self.weather_vector_scaler.fit(self.weather_vectors, axis=1)  # norm with all data
        # self.weather_vectors = self.weather_vector_scaler.transform(self.weather_vectors)
        # trn_val_weather_vectors = self.weather_val_vectors[self.trn_val_indices[0]:self.trn_val_indices[1]]
        # trn_weather_vectors = self.weather_vectors[self.trn_indices[0]:self.trn_indices[1]]
        # val_weather_vectors = self.weather_vectors[self.val_indices[0]:self.val_indices[1]]
        # tst_weather_vectors = self.weather_vectors[self.tst_indices[0]:self.tst_indices[1]]

        # normalise space dependent data - using min_max_scale - no need to save train data norm values
        self.demog_grid = min_max_scale(data=self.demog_grid, feature_range=(0, 1), axis=1)
        self.street_grid = min_max_scale(data=self.street_grid, feature_range=(0, 1), axis=1)

        # target index - also the index given to the
        # todo dependency injection of the different types of datasets
        self.training_validation_set = GridDataset(
            crimes=trn_val_crimes,
            targets=trn_val_targets,
            labels=trn_val_labels,
            t_range=trn_val_t_range,  # t_range is matched to the target index
            time_vectors=trn_val_time_vectors,
            # weather_vectors=trn_val_weather_vectors,
            demog_grid=self.demog_grid,
            street_grid=self.street_grid,

            step_c=self.step_c,
            step_p=self.step_p,
            step_q=self.step_q,

            n_steps_c=self.n_steps_c,
            n_steps_p=self.n_steps_p,
            n_steps_q=self.n_steps_q,

            shaper=self.shaper,
        )

        self.training_set = GridDataset(
            crimes=trn_crimes,
            targets=trn_targets,
            labels=trn_labels,
            t_range=trn_t_range,  # t_range is matched to the target index
            time_vectors=trn_time_vectors,
            # weather_vectors=trn_weather_vectors,
            demog_grid=self.demog_grid,
            street_grid=self.street_grid,

            step_c=self.step_c,
            step_p=self.step_p,
            step_q=self.step_q,

            n_steps_c=self.n_steps_c,
            n_steps_p=self.n_steps_p,
            n_steps_q=self.n_steps_q,

            shaper=self.shaper,
        )

        self.validation_set = GridDataset(
            crimes=val_crimes,
            targets=val_targets,
            labels=val_labels,
            t_range=val_t_range,  # t_range is matched to the target index
            time_vectors=val_time_vectors,
            # weather_vectors=val_weather_vectors,
            demog_grid=self.demog_grid,
            street_grid=self.street_grid,

            step_c=self.step_c,
            step_p=self.step_p,
            step_q=self.step_q,

            n_steps_c=self.n_steps_c,
            n_steps_p=self.n_steps_p,
            n_steps_q=self.n_steps_q,
            shaper=self.shaper,
        )

        self.testing_set = GridDataset(
            crimes=tst_crimes,
            targets=tst_targets,
            labels=tst_labels,
            t_range=tst_t_range,  # t_range is matched to the target index
            time_vectors=tst_time_vectors,
            # weather_vectors=tst_weather_vectors,
            demog_grid=self.demog_grid,
            street_grid=self.street_grid,

            step_c=self.step_c,
            step_p=self.step_p,
            step_q=self.step_q,

            n_steps_c=self.n_steps_c,
            n_steps_p=self.n_steps_p,
            n_steps_q=self.n_steps_q,

            shaper=self.shaper,
        )

    def to_counts(self, sparse_data: np.ndarray):
        """
        Convert data ndarray values to original count scale so that mae and mse metric calculations can be done.
        :param sparse_data: ndarray (N,1,H,W)
        :return: count_data (N,1,H,W)
        """
        assert len(sparse_data.shape) == 4
        _, _c, _, _ = sparse_data.shape
        assert _c == 1

        sparse_count = self.target_scaler.inverse_transform(sparse_data)[:, 0:1]
        if self.log_norm_scale:
            sparse_count = np.round(2 ** sparse_count - 1)

        return sparse_count  # (N,1,H,W)


class GridDataset(Dataset):
    """
    Grid datasets operate on un-flattened data where the map/grid of data has been is in (N,C,H,W) format.
    """

    def __init__(
            self,
            crimes: ArrayNCHW,  # time and space dependent
            targets: ArrayNCHW,  # time and space dependent
            labels: ArrayNCHW,  # time and space dependent
            t_range: pd.DatetimeIndex,  # time dependent
            time_vectors: ArrayNC,  # time dependent (optional)
            # weather_vectors,  # time dependent (optional)
            demog_grid: ArrayHWC,  # space dependent (optional)
            street_grid: ArrayHWC,  # space dependent (optional)

            step_c: int,
            step_p: int,
            step_q: int,

            n_steps_c: int,
            n_steps_p: int,
            n_steps_q: int,

            shaper: Shaper,
    ):

        self.n_steps_c = n_steps_c
        self.n_steps_p = n_steps_p
        self.n_steps_q = n_steps_q

        self.step_c = step_c
        self.step_p = step_p
        self.step_q = step_q

        self.crimes = crimes  # N,C,H,W
        self.targets = targets  # N,1,H,W
        self.labels = labels  # N,1,H,W
        self.shape = self.t_size, self.c_size, self.h_size, self.w_size = np.shape(self.crimes)  # N,C,H,W

        self.demog_grid = demog_grid
        self.street_grid = street_grid

        self.time_vectors = time_vectors
        # self.weather_vectors = weather_vectors  # remember weather should be the info of the next time step
        self.t_range = t_range

        self.shaper = shaper

        #  [min_index, max_index) are limits of flattened targets
        self.max_index = self.t_size
        self.min_index = self.step_q * self.n_steps_q
        self.len = self.max_index - self.min_index

        # used to map the predictions to the actual targets
        self.target_shape = list(self.targets.shape)
        self.target_shape[0] = self.target_shape[0] - self.min_index

    def __len__(self):
        """Denotes the total number of samples"""
        return self.len

    def __getitem__(self, index):

        if isinstance(index, slice):
            index = range(if_none(index.start, self.min_index),
                          if_none(index.stop, self.max_index),
                          if_none(index.step, 1))

        indices = np.array([index]).flatten()  # brackets and flatten caters where index is a single number

        batch_seq_c = []  # batch of Xc (closeness)
        batch_seq_p = []  # batch of Xp (period)
        batch_seq_q = []  # batch of Xq (trend)
        batch_seq_t = []  # batch of Xt (current time slot)
        batch_seq_e = []  # batch of Et (external factors of current time slot) - weather, demographics, google street view

        batch_indices = []

        for i in indices:
            # i + 1 has a plus one because the index shift between crimes and targets happens in data group
            seq_c = self.crimes[i + 1 - self.step_c * self.n_steps_c:i + 1:self.step_c] \
                .reshape(self.n_steps_c * self.c_size, self.h_size, self.w_size)
            seq_p = self.crimes[i + 1 - self.step_p * self.n_steps_p:i + 1:self.step_p] \
                .reshape(self.n_steps_p * self.c_size, self.h_size, self.w_size)
            seq_q = self.crimes[i + 1 - self.step_q * self.n_steps_q:i + 1:self.step_q] \
                .reshape(self.n_steps_q * self.c_size, self.h_size, self.w_size)
            batch_seq_c.append(seq_c)
            batch_seq_p.append(seq_p)
            batch_seq_q.append(seq_q)

            seq_t = self.targets[i:i + 1, 0]  # targets
            batch_seq_t.append(seq_t)

            seq_e = self.time_vectors[i:i + 1]  # external vectors
            batch_seq_e.append(seq_e)

            batch_indices.append(i - self.min_index)

        # output data should be in format (N,C,H,W)
        # todo look at including the channels, i.e. crime types and stack times as well, conv3d?

        # concatenate on the channel axis
        batch_seq_c = np.stack(batch_seq_c)
        batch_seq_p = np.stack(batch_seq_p)
        batch_seq_q = np.stack(batch_seq_q)
        batch_seq_e = np.stack(batch_seq_e)
        batch_seq_t = np.stack(batch_seq_t)

        return batch_indices, batch_seq_c, batch_seq_p, batch_seq_q, batch_seq_e, batch_seq_t
