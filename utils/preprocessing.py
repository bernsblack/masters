import pickle
import unittest

import numpy as np

from utils import deprecated
from utils.configs import BaseConf

from pandas.tseries.offsets import Hour as OffsetHour

from utils.data_processing import safe_divide

HOUR_NANOS = OffsetHour().nanos


def get_hours_per_time_step(freq):
    return freq.nanos / HOUR_NANOS


def get_index_mask(data: np.ndarray, threshold: float = 0, top_k: int = -1) -> np.ndarray:
    """
    :param data: array shaped (N, C, H, W)
    :param threshold: sum over all time should be above this threshold
    :param top_k: if larger than 0 we filter out only the top k most active cells of the data grid
    :return trans_mat: transition matrix used to filter out cells where nothing occurs over time
    """
    N, C, H, W = data.shape
    new_shape = N, H * W

    # only looks at the first channel
    flat_data_sum = data[:, 0].reshape(new_shape).sum(0).flatten()

    indices = np.argwhere(flat_data_sum > threshold)

    if top_k > 0:  # zero where data_sum is not
        top_indices = np.argsort(flat_data_sum)[:-top_k - 1:-1]  # ensures the
        indices = np.intersect1d(top_indices, indices)  # sort the array for us

    return indices.flatten()


@deprecated
def get_trans_mat(data, threshold=0, top_k=-1):
    """
    :param data: array shaped (N, C, H, W)
    :param threshold: sum over all time should be above this threshold
    :param top_k: if larger than 0 we filter out only the top k most active cells of the data grid
    :return trans_mat: transition matrix used to filter out cells where nothing occurs over time
    """
    old_shape = list(data.shape)
    new_shape = old_shape[:-2]
    new_shape.append(old_shape[-1] * old_shape[-2])

    flat_data_sum = data.reshape(new_shape).sum(0)[0]

    indices = np.argwhere(flat_data_sum > threshold)

    if top_k > 0:  # zero where data_sum is not
        top_indices = np.argsort(flat_data_sum)[:-top_k - 1:-1]  # ensures the
        indices = np.intersect1d(top_indices, indices)  # sort the array for us

    trans_mat = np.zeros((new_shape[-1], len(indices)))
    for i, j in enumerate(indices):
        trans_mat[j, i] = 1

    return trans_mat


# utils functions because we're having issues importing utils
@deprecated
def get_trans_mat_old(data, threshold=0, top_k=-1):
    """
    :param data: array shaped (N, C, H, W)
    :param threshold: sum over all time should be above this threshold
    :param top_k: if larger than 0 we filter out only the top k most active cells of the data grid
    :return trans_mat: transition matrix used to filter out cells where nothing occurs over time
    """
    shape_old = np.shape(data)
    num_axis = len(shape_old)
    if num_axis != 4:
        raise ValueError(f"data has {num_axis} axis and shape {shape_old} should be 4: (N,C,H,W)")

    data = data[:, 0]  # select only the TOTAL chanel
    n, h, w = np.shape(data)
    shape_new = (n, h * w)

    data = np.reshape(data, shape_new)

    data_sum = np.sum(data, 0)

    if top_k > 0:  # zero where data_sum is not
        bottom_indices = np.sort(np.argsort(data.mean(0))[:-top_k])  # ensures the
        data_sum[bottom_indices] = 0
    data_sum[data_sum <= threshold] = 0
    data_sum[data_sum > threshold] = 1

    data_sum = np.expand_dims(data_sum, 1)
    trans_mat = []
    for i in range(len(data_sum)):
        if data_sum[i] != 0:
            f = np.zeros(len(data_sum))
            f[i] = 1
            trans_mat.append(f)
    trans_mat = np.array(trans_mat).T
    return trans_mat


class Shaper:
    def __init__(self, data: np.ndarray, conf: BaseConf = BaseConf()):
        """

        :param data: array shaped (N, C, H, W)
        :param conf: contains: shaper_threshold and shaper_top_k used to determine cells to keep in dense representation
        """
        threshold = conf.shaper_threshold  # sum over all time should be above this threshold
        top_k = conf.shaper_top_k  # if larger than 0 we filter out only the top k most active cells of the data grid

        shape = list(np.shape(data))

        self.h, self.w = shape[-2:]

        self.index_mask = get_index_mask(data=data, threshold=threshold, top_k=top_k)

        self.l = len(self.index_mask)

        coords = []
        for y in range(self.h):
            for x in range(self.w):
                coords.append((y, x))
        coords = np.array(coords)[self.index_mask]
        self._yx2i_map = {tuple(yx): i for i, yx in enumerate(coords)}
        self._i2yx_map = {i: yx for yx, i in self._yx2i_map.items()}

    def __eq__(self, other):
        size_match = (self.h, self.w, self.l) == (other.h, other.w, other.l)
        if not size_match:
            return False
        mask_match = (self.index_mask == other.index_mask).all()
        return size_match and mask_match

    def i_to_yx(self, i):
        """
        maps squeezed coordinates to unsqueezed indices
        :param i: index of the cell in the squeezed format - (N,C, L) with i ∈ L
        :return: tuple of coordinate y,x in unsqueezed format - (N, C, H, W) with y ∈ H, x ∈ W

        will return None if i not in mapping
        """
        return self._i2yx_map.get(i)

    def yx_to_i(self, y, x):
        """
        maps unsqueezed coordinates to squeezed indices
        :param y: y index in unsqueezed format - (N, C, H, W) with y ∈ H
        :param x: x index in unsqueezed format - (N, C, H, W) with x ∈ W
        :return: index of the cell in the squeezed format - (N,C, L) with index ∈ L

        will return None if (y,x) not in mapping
        """
        return self._yx2i_map.get((y, x))

    def squeeze(self, sparse_data):
        """

        :param sparse_data: np.array with shape (N, C, H, W)
        :return dense_data: np.array with shape (N, C, L)
        """
        shape = list(np.shape(sparse_data))

        shape_new = shape[:-2]
        shape_new.append(int(np.product(shape[-2:])))

        reshaped_data = np.reshape(sparse_data, shape_new)
        dense_data = reshaped_data[:, :, self.index_mask]
        return dense_data

    def unsqueeze(self, dense_data):
        """
        :param dense_data: np.array with shape (N, C, L)
        :return sparse_data: np.array with shape (N, C, H, W):
        """
        N, C, L = np.shape(dense_data)
        shape = list(np.shape(dense_data))

        shape_old = shape[:-1]
        shape_old.extend([self.h, self.w])

        sparse_data = np.zeros((N, C, self.h * self.w))
        sparse_data[:, :, self.index_mask] = dense_data

        reshaped_data = np.reshape(sparse_data, shape_old)
        return reshaped_data

    def save(self, save_folder: str):
        save_shaper(self, save_folder)


def save_shaper(shaper: Shaper, save_folder: str):
    save_folder = save_folder.rstrip('/')
    with open(f"{save_folder}/shaper.pkl", "wb") as f:
        pickle.dump(shaper, f)


def load_shaper(load_folder: str):
    load_folder = load_folder.rstrip('/')
    with open(f"{load_folder}/shaper.pkl", "rb") as f:
        shaper_loaded = pickle.load(f)
    return shaper_loaded


class TestShaperEquals(unittest.TestCase):
    def test_shaper_equals(self):
        conf = BaseConf()
        data0 = np.random.binomial(1, 0.1, (10, 1, 100, 100))
        shaper0 = Shaper(data0, conf)
        shaper2 = Shaper(data0, conf)
        data1 = np.random.binomial(1, 0.1, (10, 1, 100, 100))
        shaper1 = Shaper(data1, conf)
        self.assertEqual(shaper0, shaper2)
        self.assertNotEqual(shaper0, shaper1)


class TestShaperIndexConversion(unittest.TestCase):
    def test_shaper_index_conversion(self):
        test_sparse = np.arange(40 * 50).reshape(1, 1, 40, 50)
        conf = BaseConf({"shaper_threshold": 0, "shaper_top_k": -1})
        shaper = Shaper(test_sparse, conf)

        test_dense = shaper.squeeze(test_sparse)

        i = 34
        y, x = shaper.i_to_yx(i)
        self.assertTrue(i == shaper.yx_to_i(y, x))  # test inversion
        self.assertTrue((test_dense[:, :, i] == test_sparse[:, :, y, x]).all())  # test values mapping


@deprecated
class ShaperDeprecated:  # was to slow has been replaced
    @deprecated
    def __init__(self, data, threshold=0, top_k=-1):
        """

        :param top_k: if larger than 0 we filter out only the top k most active cells of the data grid
        :param data: array shaped (N, C, H, W)
        :param threshold: sum over all time should be above this threshold
        """
        shape = list(np.shape(data))

        self.h, self.w = shape[-2:]

        self.trans_mat = get_trans_mat(data=data, threshold=threshold, top_k=top_k)

        self.l = self.trans_mat.shape[-1]

    @deprecated
    def squeeze(self, sparse_data):
        """

        :param sparse_data: np.array with shape (N, C, H, W)
        :return dense_data: np.array with shape (N, C, H * W)
        """
        shape = list(np.shape(sparse_data))

        shape_new = shape[:-2]
        shape_new.append(int(np.product(shape[-2:])))

        reshaped_data = np.reshape(sparse_data, shape_new)
        dense_data = np.matmul(reshaped_data, self.trans_mat)
        return dense_data

    @deprecated
    def unsqueeze(self, dense_data):
        """
        :param dense_data: np.array with shape (N, C, H * W)
        :return sparse_data: np.array with shape (N, C, H, W):
        """
        shape = list(np.shape(dense_data))

        shape_old = shape[:-1]
        shape_old.extend([self.h, self.w])

        sparse_data = np.matmul(dense_data, self.trans_mat.T)
        reshaped_data = np.reshape(sparse_data, shape_old)
        return reshaped_data


def min_max_scale(data, feature_range=(0, 1), axis=0):
    """
    'min_max_scale' scales values of the ndarray so that each index in given axis is scaled between feature_range.
    Example: if we have ndarray of stacked_images of  shape (n_images, n_channels, n_height, n_width) and we want to
    scaled each channel independently from the other use:
        min_max_scale(stacked_images, feature_range=(0, 1), axis=1)

    This ensures each channel has max and min of 0 and 1 respectively.

    'min_max_scale' can be used if we do not care about re-scaling data back into original values.
    To rescale or save min and max of a certain set then use MinMaxScaler class.

    :param data: numpy array to be scaled 
    :param feature_range: tuple of min and max values
    :param axis: axis in which the scaling will happen
    :return: scales numpy array in the axis provided
    """

    shape = np.shape(data)
    sum_axis = tuple(set(range(len(shape))) - {axis})

    min_old = np.min(data, axis=sum_axis, keepdims=True)
    max_old = np.max(data, axis=sum_axis, keepdims=True)
    scale_old = max_old - min_old
    # if np.any(scale_old == 0):
    #     raise ValueError(f"scale_old is {scale_old}. Division by zero is not allowed.")

    min_new = np.ones(min_old.shape) * feature_range[0]
    max_new = np.ones(max_old.shape) * feature_range[1]
    scale_new = max_new - min_new

    # return scale_new * (data - min_old) / scale_old + min_new
    return safe_divide(scale_new * (data - min_old), scale_old) + min_new


class MinMaxScaler:
    """
    Used to scale and inverse scale features of data

    'MinMaxScaler' scales values of the ndarray so that each index in given axis is scaled between feature_range.
    Example: if we have ndarray of stacked_images of  shape (n_images, n_channels, n_height, n_width) and we want to
    scaled each channel independently from the other use:
        min_max_scale(stacked_images, feature_range=(0, 1), axis=1)

    """

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range

        self.min_new = None
        self.max_new = None
        self.scale_new = None

        self.min_old = None
        self.max_old = None
        self.scale_old = None

    def fit(self, data, axis):
        shape = np.shape(data)
        sum_axis = tuple(set(range(len(shape))) - {axis})

        self.min_old = np.min(data, axis=sum_axis, keepdims=True)
        self.max_old = np.max(data, axis=sum_axis, keepdims=True)
        self.scale_old = self.max_old - self.min_old

        self.min_new = np.ones(self.min_old.shape) * self.feature_range[0]
        self.max_new = np.ones(self.max_old.shape) * self.feature_range[1]
        self.scale_new = self.max_new - self.min_new

    def transform(self, data):
        return safe_divide(self.scale_new * (data - self.min_old), self.scale_old) + self.min_new
        # if np.any(self.scale_old == 0):
        #     raise ValueError(f"self.scale_old is {self.scale_old}. Cannot divide by zero. Data shape -> {data.shape}")
        # return self.scale_new * (data - self.min_old) / self.scale_old + self.min_new

    def inverse_transform(self, data):
        return safe_divide(self.scale_old * (data - self.min_new), self.scale_new) + self.min_old
        # if np.any(self.scale_new == 0):
        #     raise ValueError(f"self.scale_new is {self.scale_new}. Cannot divide by zero")
        # return self.scale_old * (data - self.min_new) / self.scale_new + self.min_old

    def fit_transform(self, data, axis):
        self.fit(data, axis)
        return self.transform(data)


def scale_per_time_slot(data):
    data_scaled = min_max_scale(data, feature_range=(0, 1), axis=0)
    return data_scaled


import unittest


class TestMinMaxScaler(unittest.TestCase):

    def test_min_max_inverse(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        mock = np.arange(7 * 5 * 3).reshape(7, 5, 3) % 13
        mock_scaled = scaler.fit_transform(mock, axis=1)

        mock_scaled.max(1)

        mock_unscaled = scaler.inverse_transform(mock_scaled)
        self.assertTrue((mock_unscaled == mock).all())


@deprecated
class MeanStdScaler:
    """
    Used to scale and inverse scale features of data
    """

    def __init__(self):
        self.mean_ = None
        self.max_old = None
        self.std_ = None

    def fit(self, data, axis):
        shape = np.shape(data)
        sum_axis = tuple(set(range(len(shape))) - {axis})

        self.mean_ = np.mean(data, axis=sum_axis, keepdims=True)
        self.std_ = np.std(data, axis=sum_axis, keepdims=True)

    def transform(self, data):
        return (data - self.mean_) / self.std_

    def inverse_transform(self, data):
        return self.std_ * data + self.mean_

    def fit_transform(self, data, axis):
        self.fit(data, axis)
        return self.transform(data)


class TestMeanStdScaler(unittest.TestCase):

    def test_mean_std_inverse(self):
        scaler = MeanStdScaler()
        mock = np.arange(7 * 5 * 3).reshape(7, 5, 3) % 13
        mock_scaled = scaler.fit_transform(mock, axis=1)
        mock_unscaled = scaler.inverse_transform(mock_scaled)
        self.assertTrue((mock_unscaled == mock).all())
