import numpy as np
import pandas as pd
from torch import nn
from utils.data_processing import get_period


class HistoricAverage:
    def __init__(self, step=1):
        self.step = step

    def fit(self, data):
        self.step = get_period(data)

    def __call__(self, a):
        r = np.empty(a.shape)
        r.fill(np.nan)
        for i in range(self.step + 1, len(r)):
            x = np.mean(a[i - self.step + 1:0:-self.step], axis=0)
            r[i] = x

        return r


class BaseMovingAverage:
    def __init__(self, window_len, weights):
        self.window_len = window_len
        self.weights = weights

        self.f_avg = lambda x, wgt: np.average(x, axis=0, weights=wgt)

    def fn(self, a):
        return np.average(a, axis=0, weights=self.weights)

    def __call__(self, a):
        w = self.window_len
        r = np.empty(a.shape)
        r.fill(np.nan)
        for i in range(w - 1, len(r)):
            x = a[(i - w + 1):i + 1]
            r[i] = self.fn(x)

        return r


class ExponentialMovingAverage(BaseMovingAverage):
    def __init__(self, alpha, window_len):
        self.alpha = alpha
        self.window_len = window_len
        self.weights = np.exp(-self.alpha / np.arange(1, self.window_len + 1))
        super(ExponentialMovingAverage, self).__init__(self.window_len, self.weights)


class TriangularMovingAverage(BaseMovingAverage):
    def __init__(self, window_len):
        self.window_len = window_len
        self.weights = np.arange(1, self.window_len + 1)
        super(TriangularMovingAverage, self).__init__(self.window_len, self.weights)


class UniformMovingAverage(BaseMovingAverage):
    def __init__(self, window_len):
        self.window_len = window_len
        self.weights = np.ones(self.window_len)
        super(UniformMovingAverage, self).__init__(self.window_len, self.weights)


class LinearRegressor(nn.Module):
    def __init__(self, in_features=10, out_features=2):
        super(LinearRegressor, self).__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.activation(self.linear(x))

        return out


# TODO CONVERT FUNCTIONS INTO A CLASS
# still needs to do it only on training data average no the total
def get_historic_average(a, step=24):
    """
    a: array (N,d) or (N,d,d)
    step: the historic step the average is taken over, e.g. 24 for every 24 hours
    """
    ha = []  # historic averages

    for i in range(step):
        ha.append(a[i::step].mean(0))

    b = -1 * np.ones_like(a)  # minus one there to check if anything went wrong
    for i in range(len(b)):
        b[i] = ha[i % step]

    return b


def rolling_apply(fun, a, w):  # this rolling average includes the current time step
    r = np.empty(a.shape)
    r.fill(np.nan)
    for i in range(w - 1, a.shape[0]):
        r[i] = fun(a[(i - w + 1):i + 1])
    return r


def rolling_avg(fun, a, weights):  # this rolling average includes the current time step
    w = len(weights)
    r = np.empty(a.shape)
    r.fill(np.nan)
    for i in range(w - 1, a.shape[0]):
        r[i] = fun(a[(i - w + 1):i + 1], weights)

    return r


if __name__ == "__main__":
    a = np.random.randn(100)

    folder = "./data/processed/"

    zip_file = np.load(folder + "generated_data.npz")

    grids = zip_file["crime_grids"]
    t_range = pd.read_pickle(folder + "t_range.pkl")

    grids = np.load(folder + 'crimes_grids.npy')
    f = lambda x: x.mean(0)
    avg = lambda x, weights: np.average(x, axis=0, weights=weights)  # x and weights should have same length
    # np.average()

    window = 4
    weights = np.exp(-10 / np.arange(window))

    lag = 1
    dT = '1D'
    if dT == '1D':
        folder = "./data/processed/T24H-X850M-Y880M/"
        historic_jump = 7
    elif dT == '4H':
        folder = "./data/processed/T4H-X850M-Y880M/"
        historic_jump = 24
    else:
        folder = "./data/processed/T24H-X850M-Y880M/"
        historic_jump = 1

    grids_lag = grids[:-1]
    grids = grids[1:]

    grids_ma = rolling_apply(f, grids_lag, window)  # use lag because we cannot include today's rate in the MA

    grids_avg = rolling_avg(avg, grids, weights)

    grids_ha = get_historic_average(grids[:len(grids)], historic_jump)  # use grids as we get average for that time
    grids_hama = grids_ha * grids_ma

    grids_ha = np.nan_to_num(grids_ha)
    grids_ma = np.nan_to_num(grids_ma)
    grids_hama = np.nan_to_num(grids_hama)

    # skip 10 for window size
    grids = grids[window - 1:]
    grids_lag = grids_lag[window - 1:]  # can cap too if we're only looking into binary classification

    grids_ma = grids_ma[window - 1:]
    grids_ha = grids_ha[window - 1:]
    grids_hama = grids_hama[window - 1:]
    grids_avg = grids_avg[window - 2:]
