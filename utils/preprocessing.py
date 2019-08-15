import numpy as np


# utils functions because we're having issues importing utils
def get_trans_mat(data, threshold=0):
    """
    :param data: array shaped (N, C, W, H)
    :param threshold: sum over all time should be above this threshold
    :return trans_mat: transition matrix used to filter out cells where nothing occurs over time
    """
    if len(np.shape(data)) != 4:
        raise ValueError("data shape must be length 4: (N,C,H,W)")

    data = data[:, 0]  # select only the TOTAL chanel
    n, h, w = np.shape(data)
    new_shape = (n, h * w)

    data = np.reshape(data, new_shape)
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


class Shaper:
    def __init__(self, data):
        shape = list(np.shape(data))
        self.old_shape = shape
        self.new_shape = shape[:-2]
        self.new_shape.append(int(np.product(shape[-2:])))

        if len(np.shape(data)) > 3:
            self.trans_mat_d2s = get_trans_mat(data[:, 0])
        else:
            self.trans_mat_d2s = get_trans_mat(data)

    def squeeze(self, sparse_data):
        reshaped_data = np.reshape(sparse_data, self.new_shape)
        dense_data = np.matmul(reshaped_data, self.trans_mat_d2s)
        return dense_data

    def unsqueeze(self, dense_data):
        sparse_data = np.matmul(dense_data, self.trans_mat_d2s.T)
        reshaped_data = np.reshape(sparse_data, self.old_shape)
        return reshaped_data


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
        return self.scale_new * (data - self.min_old) / self.scale_old + self.min_new

    def inverse_transform(self, data):
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
