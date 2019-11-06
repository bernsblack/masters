import logging as log

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils.configs import BaseConf
from utils.preprocessing import Shaper, MinMaxScaler, minmax_scale
from utils.utils import if_none


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
        :param data_path:
        :param conf:
        """

        with np.load(data_path + "generated_data.npz") as zip_file:  # context helper ensures zip_file is closed
            if conf.use_crime_types:
                self.crimes = zip_file["crime_types_grids"]
            else:
                self.crimes = zip_file["crime_grids"]

            self.total_len = len(self.crimes)  # length of the whole time series

            t_range = pd.read_pickle(data_path + "t_range.pkl")
            log.info(f"\tt_range shape {np.shape(t_range)}")

            freqstr = t_range.freqstr
            if freqstr == "H":
                freqstr = "1H"
            time_step_hrs = int(freqstr[:freqstr.find("H")])  # time step in hours

            self.step_c = 1
            self.step_p = int(24 / time_step_hrs)
            self.step_q = int(168 / time_step_hrs)  # maximum offset

            self.n_steps_c = conf.n_steps_c  # 3
            self.n_steps_p = conf.n_steps_p  # 3
            self.n_steps_q = conf.n_steps_q  # 3

            #  split the data into ratios - size represent the targets sizes not the number of time steps
            total_offset = self.step_q * self.n_steps_q

            target_len = self.total_len - total_offset
            val_size = int(target_len * conf.val_ratio)
            tst_size = int(target_len * conf.tst_ratio)
            trn_size = int(target_len - tst_size - val_size)

            #  start and stop t_index of each dataset - can be used outside of loader/group
            self.tst_indices = np.array([self.total_len - tst_size, self.total_len])
            self.val_indices = np.array([self.tst_indices[0] - val_size, self.tst_indices[0]])
            self.trn_indices = np.array([self.val_indices[0] - trn_size, self.val_indices[0]])

            # used to create the shaper so that all datasets have got values in them
            tmp_trn_crimes = self.crimes[self.trn_indices[0]:self.trn_indices[1], 0:1]
            tmp_val_crimes = self.crimes[self.val_indices[0]:self.val_indices[1], 0:1]
            tmp_tst_crimes = self.crimes[self.tst_indices[0]:self.tst_indices[1], 0:1]
            shaper_crimes = np.max(tmp_trn_crimes,axis=0, keepdims=True) * \
                            np.max(tmp_val_crimes, axis=0, keepdims=True) * \
                            np.max(tmp_tst_crimes, axis=0, keepdims=True)

            # fit crime data to shaper - only used to squeeze results when calculating the results
            self.shaper = Shaper(data=shaper_crimes,
                                 conf=conf)

            # add tract count to crime grids - done separately in case we do not want crime types or arrests
            # tract_count_grids = zip_file["tract_count_grids"]
            # self.crimes = np.concatenate((self.crimes, tract_count_grids), axis=1)

            # (reminder) ensure that the self.crimes are not scaled to -1,1 before
            self.targets = np.copy(self.crimes[1:])  # only check for totals > 0

            self.crimes = self.crimes[:-1]

            self.time_vectors = zip_file["time_vectors"][1:]  # already normalised - time vector of future crime

            self.t_range = t_range[1:]

            self.demog_grid = zip_file["demog_grid"]
            self.street_grid = zip_file["street_grid"]

        #  sanity check if time matches up with our grids
        if len(self.t_range) - 1 != len(self.crimes):
            log.error("time series and time range lengths do not match up -> " +
                      "len(self.t_range) != len(self.crimes)")
            raise RuntimeError(f"len(self.t_range) - 1 {len(self.t_range) - 1} != len(self.crimes) {len(self.crimes)} ")


        self.tst_indices[0] = self.tst_indices[0] - total_offset
        self.val_indices[0] = self.val_indices[0] - total_offset
        self.trn_indices[0] = self.trn_indices[0] - total_offset

        trn_t_range = self.t_range[self.trn_indices[0]:self.trn_indices[1]]
        val_t_range = self.t_range[self.val_indices[0]:self.val_indices[1]]
        tst_t_range = self.t_range[self.tst_indices[0]:self.tst_indices[1]]

        self.crimes = np.log2(1 + self.crimes)
        self.crime_scaler = MinMaxScaler(feature_range=(0, 1))
        self.crime_scaler.fit(self.crimes[self.trn_indices[0]:self.trn_indices[1]], axis=1)
        self.crimes = self.crime_scaler.transform(self.crimes)

        self.crimes = self.crimes[:, 0]  # only select totals after scaling channel wise

        trn_crimes = self.crimes[self.trn_indices[0]:self.trn_indices[1]]
        val_crimes = self.crimes[self.val_indices[0]:self.val_indices[1]]
        tst_crimes = self.crimes[self.tst_indices[0]:self.tst_indices[1]]

        # targets
        self.targets = np.log2(1 + self.targets)
        self.targets = self.crime_scaler.transform(self.targets)
        self.targets = self.targets[:, 0]

        trn_targets = self.targets[self.trn_indices[0]:self.trn_indices[1]]
        val_targets = self.targets[self.val_indices[0]:self.val_indices[1]]
        tst_targets = self.targets[self.tst_indices[0]:self.tst_indices[1]]

        # time_vectors is already scaled between 0 and 1
        trn_time_vectors = self.time_vectors[self.trn_indices[0]:self.trn_indices[1]]
        val_time_vectors = self.time_vectors[self.val_indices[0]:self.val_indices[1]]
        tst_time_vectors = self.time_vectors[self.tst_indices[0]:self.tst_indices[1]]

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

        # target index - also the index given to the
        # todo dependency injection of the different types of datasets
        self.training_set = GridDataset(
            crimes=trn_crimes,
            targets=trn_targets,
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


class GridDataset(Dataset):
    """
    Grid datasets operate on un-flattened data where the map/grid of data has been is in
    (N,C,H,W) format.
    """

    def __init__(
            self,
            crimes,  # time and space dependent
            targets,  # time and space dependent
            t_range,  # time dependent
            time_vectors,  # time dependent (optional)
            # weather_vectors,  # time dependent (optional)
            demog_grid,  # space dependent (optional)
            street_grid,  # space dependent (optional)

            step_c,
            step_p,
            step_q,

            n_steps_c,
            n_steps_p,
            n_steps_q,

            shaper,
    ):

        self.n_steps_c = n_steps_c
        self.n_steps_p = n_steps_p
        self.n_steps_q = n_steps_q

        self.step_c = step_c
        self.step_p = step_p
        self.step_q = step_q

        self.crimes = crimes
        self.targets = targets
        self.shape = self.t_size, self.h_size, self.w_size = np.shape(self.crimes)  # N,H,W

        self.demog_grid = demog_grid
        self.street_grid = street_grid

        self.time_vectors = time_vectors
        # self.weather_vectors = weather_vectors  # remember weather should be the info of the next time step
        self.t_range = t_range

        self.shaper = shaper

        #  [min_index, max_index) are limits of flattened targets
        self.max_index = self.t_size
        self.min_index = self.step_q * self.n_steps_q
        self.len = self.min_index - self.min_index

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
            seq_c = self.crimes[i + 1 - self.step_c * self.n_steps_c:i + 1:self.step_c]
            seq_p = self.crimes[i + 1 - self.step_p * self.n_steps_p:i + 1:self.step_p]
            seq_q = self.crimes[i + 1 - self.step_q * self.n_steps_q:i + 1:self.step_q]
            batch_seq_c.append(seq_c)
            batch_seq_p.append(seq_p)
            batch_seq_q.append(seq_q)

            seq_t = self.targets[i:i + 1]  # targets
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
