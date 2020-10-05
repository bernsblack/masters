import numpy as np
from torch.utils.data import Dataset
from utils.configs import BaseConf
from utils.utils import if_none
from datasets.base_datagroup import BaseDataGroup
import logging as log

class FlatDataGroup(BaseDataGroup):
    """
    FlatDataGroup class acts as a collection of datasets (training/validation/test)
    The data group loads data from disk, splits in separate sets and normalises according to train set.
    Crime count related data is first scaled using f(x) = log2(1 + x) and then scaled between 0 and 1.
    The data group class also handles reshaping of data.
    """

    def __init__(self, data_path: str, conf: BaseConf):
        """

        :param data_path: Path to the data folder with all spatial and temporal data.
        :param conf: Config class with pre-set and global values
        """
        log.info('Initialising Flat Data Group')
        super(FlatDataGroup, self).__init__(data_path, conf)


        self.training_set = FlatDataset(
            crimes=self.trn_crimes,
            targets=self.trn_targets,
            labels=self.trn_labels,
            total_crimes=self.trn_total_crimes,
            t_range=self.trn_t_range,  # t_range is matched to the target index
            time_vectors=self.trn_time_vectors,
            # weather_vectors=trn_weather_vectors,
            demog_grid=self.demog_grid,
            street_grid=self.street_grid,
            seq_len=self.seq_len,
            offset_year=self.offset_year,
            shaper=self.shaper,
        )

        self.validation_set = FlatDataset(
            crimes=self.val_crimes,
            targets=self.val_targets,
            labels=self.val_labels,
            total_crimes=self.val_total_crimes,
            t_range=self.val_t_range,  # t_range is matched to the target index
            time_vectors=self.val_time_vectors,
            # weather_vectors=val_weather_vectors,
            demog_grid=self.demog_grid,
            street_grid=self.street_grid,
            seq_len=self.seq_len,
            offset_year=self.offset_year,
            shaper=self.shaper,
        )

        self.testing_set = FlatDataset(
            crimes=self.tst_crimes,
            targets=self.tst_targets,
            labels=self.tst_labels,
            total_crimes=self.tst_total_crimes,
            t_range=self.tst_t_range,  # t_range is matched to the target index
            time_vectors=self.tst_time_vectors,
            # weather_vectors=tst_weather_vectors,
            demog_grid=self.demog_grid,
            street_grid=self.street_grid,
            seq_len=self.seq_len,
            offset_year=self.offset_year,
            shaper=self.shaper,
        )

        self.training_validation_set = FlatDataset(
            crimes=self.trn_val_crimes,
            targets=self.trn_val_targets,
            labels=self.trn_val_labels,
            total_crimes=self.trn_val_total_crimes,
            t_range=self.trn_val_t_range,  # t_range is matched to the target index
            time_vectors=self.trn_val_time_vectors,
            # weather_vectors=trn_val_weather_vectors,
            demog_grid=self.demog_grid,
            street_grid=self.street_grid,
            seq_len=self.seq_len,
            offset_year=self.offset_year,
            shaper=self.shaper,
        )

    def to_counts(self, dense_data):
        """
        convert data ndarray values to original count scale so that mae and mse metric calculations can be done.
        :param dense_data: ndarray (N,1,L)
        :return: count_data (N,1,L)
        """

        dense_descaled = self.target_scaler.inverse_transform(dense_data)[:, 0:1]
        dense_count = np.round(2 ** dense_descaled - 1)

        return dense_count  # (N,1,L)


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
            labels,  # time and space dependent
            total_crimes,  # time dependent
            t_range,  # time dependent
            time_vectors,  # time dependent
            # weather_vectors,  # time dependent
            demog_grid,  # space dependent
            street_grid,  # space dependent
            seq_len,
            offset_year,
            shaper,
    ):
        self.seq_len = seq_len
        self.offset_year = offset_year

        self.crimes = crimes
        self.targets = targets
        self.labels = labels
        self.t_size, _, self.l_size = np.shape(self.crimes)
        self.total_crimes = total_crimes

        self.demog_grid = demog_grid
        self.street_grid = street_grid

        self.time_vectors = time_vectors
        # self.weather_vectors = weather_vectors  # remember weather should be the info of the next time step
        self.t_range = t_range

        self.shaper = shaper

        #  [min_index, max_index) are limits of flattened targets
        self.max_index = self.t_size * self.l_size
        self.min_index = (self.offset_year + self.seq_len) * self.l_size
        self.len = self.max_index - self.min_index  # todo WARNING WON'T LINE UP WITH BATCH LOADERS IF SUB-SAMPLING

        self.shape = self.t_size, self.l_size  # used when saving the model results

        # used to map the predictions to the actual targets
        self.target_shape = list(self.targets.shape)
        self.target_shape[0] = self.target_shape[0] - self.seq_len - self.offset_year

    def __len__(self):
        """Denotes the total number of samples"""
        return self.len  # todo WARNING WON'T LINE UP WITH BATCH LOADERS IF SUB-SAMPLING

    def __getitem__(self, index):
        # when using no teacher forcing
        # target = self.targets[t+self.seq_len, :, l]
        # todo add all other data - should be done in data-generation?
        # [√] number of incidents of crime occurrence by sampling point in 2013 (1-D) - crime_grid[t-365]
        # [√] number of incidents of crime occurrence by census tract in 2013 (1-D) - crime_tract[t-365]
        # [√] number of incidents of crime occurrence by sampling point yesterday (1-D) - crime_grid[t-1]
        # [√] number of incidents of crime occurrence by census tract yesterday (1-D) - crime_tract[t-1]
        # [√] number of incidents of crime occurrence by date in 2013 (1-D) - total[t-365]
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
            t_start = t_index - self.seq_len + 1
            t_stop = t_index + 1

            crime_vec = self.crimes[t_start:t_stop, :, l_index]

            crimes_last_year = self.crimes[t_start - self.offset_year:t_stop - self.offset_year, :, l_index]
            crimes_total = self.total_crimes[t_start:t_stop]

            crime_vec = np.concatenate((crime_vec, crimes_total, crimes_last_year), axis=-1)

            time_vec = self.time_vectors[t_start:t_stop]
            demog_vec = self.demog_grid[:, :, l_index]
            street_vec = self.street_grid[:, :, l_index]
            # weather_vec = self.weather_vectors[t_start:t_stop]
            # tmp_vec = np.concatenate((time_vec, weather_vec, crime_vec), axis=-1)  # todo add more historical values
            tmp_vec = np.concatenate((crime_vec, time_vec), axis=-1)  # todo add more historical values

            # todo teacher forcing - if we are using this then we need to return sequence of targets
            target_vec = self.targets[t_start:t_stop, :, l_index]

            stack_spc_feats.append(demog_vec)  # todo stacking the same grid might cause memory issues
            stack_env_feats.append(street_vec)  # todo stacking the same grid might cause memory issues

            stack_tmp_feats.append(tmp_vec)
            stack_targets.append(target_vec)

            result_indices.append((t_index - self.offset_year - self.seq_len, 0,
                                   l_index))  # extra dimension C, makes it easier for the shaper

        # spc_feats: [demog_vec]
        # env_feats: [street_vec]
        # tmp_feats: [time_vec, weather_vec, crime_vec]  - no more weather for now
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
