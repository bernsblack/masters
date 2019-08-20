import numpy as np


# utils functions because we're having issues importing utils
def get_trans_mat(data, threshold=0):
    """
    :param data: array shaped (N, C, W, H)
    :param threshold: sum over all time should be above this threshold
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
    data_sum[data_sum > threshold] = 1
    data_sum = np.reshape(data_sum, (len(data_sum), 1))
    trans_mat = []
    for i in range(len(data_sum)):
        if data_sum[i] != 0:
            f = np.zeros(len(data_sum))
            f[i] = 1
            trans_mat.append(f)
    trans_mat = np.array(trans_mat).T
    return trans_mat


class Shaper:  # TODO MAKE CHANNEL SIZE INDEPENDENT
    def __init__(self, data):
        shape = list(np.shape(data))

        self.h, self.w = shape[-2:]
        self.l = self.h * self.w

        self.trans_mat = get_trans_mat(data)

    def squeeze(self, sparse_data):
        shape = list(np.shape(sparse_data))

        shape_new = shape[:-2]
        shape_new.append(int(np.product(shape[-2:])))

        reshaped_data = np.reshape(sparse_data, shape_new)
        dense_data = np.matmul(reshaped_data, self.trans_mat)
        return dense_data

    def unsqueeze(self, dense_data):
        shape = list(np.shape(dense_data))

        shape_old = shape[:-1]
        shape_old.extend([self.h, self.w])

        sparse_data = np.matmul(dense_data, self.trans_mat.T)
        reshaped_data = np.reshape(sparse_data, shape_old)
        return reshaped_data


def minmax_scale(data, feature_range=(0, 1), axis=0):
    """
    function can be used if we do not care about re-scaling data back into original values
    when we want to rescale or save min and max of a certain set then use MinMaxScaler class

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
    if np.any(scale_old == 0):
        raise ValueError(f"scale_old is {scale_old}. Division by zero is not allowed.")

    min_new = np.ones_like(min_old) * feature_range[0]
    max_new = np.ones_like(max_old) * feature_range[1]
    scale_new = max_new - min_new

    return scale_new * (data - min_old) / scale_old + min_new


class MinMaxScaler:
    """
    Used to scale and inverse scale features of data
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

        self.min_new = np.ones_like(self.min_old) * self.feature_range[0]
        self.max_new = np.ones_like(self.max_old) * self.feature_range[1]
        self.scale_new = self.max_new - self.min_new

    def transform(self, data):
        if np.any(self.scale_old == 0):
            raise ValueError(f"self.scale_old is {self.scale_old}. Cannot device by zero")
        return self.scale_new * (data - self.min_old) / self.scale_old + self.min_new

    def inverse_transform(self, data):
        if np.any(self.scale_new == 0):
            raise ValueError(f"self.scale_new is {self.scale_new}. Cannot device by zero")
        return self.scale_old * (data - self.min_new) / self.scale_new + self.min_old

    def fit_transform(self, data, axis):
        self.fit(data, axis)
        return self.transform(data)


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
