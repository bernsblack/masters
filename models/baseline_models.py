import numpy as np
import pandas as pd
from torch import nn
import logging as log
from utils.data_processing import get_period


def get_max_steps(data, step):
    """
    data shape: (N,C,L)
    """
    if len(data.shape) != 3:
        raise ValueError("Data should be in format (N,C,L)")

    log.info("finding best max_steps parameter...")
    inputs = np.copy(data[:, 0])
    targets = np.copy(inputs)
    targets = targets[1:]
    inputs = inputs[:-1]

    tst_len = int(0.3 * len(inputs))

    results = {}

    for max_steps in range(1, 10):
        out = historic_average(data=inputs,
                               step=step,
                               max_steps=max_steps)

        mae = np.mean(np.abs(targets[tst_len:] - out[tst_len:]))
        # print(f"mae:{mae} max_steps:{max_steps}")
        results[mae] = max_steps
    best = results[min(list(results.keys()))]

    log.info(f"best max_steps -> {best}")
    return best


def historic_average(data, step, max_steps):
    """
    historic_average gets the historic average of the following time step without including that time step.
    it gets the the historic average of each cell and then moves it one time step forward.
    We do not want to use the input cell's historic average because it's less related to the target cell

    :param data: (N, ...) shape
    :param step: interval used to calculate the average
    :param max_steps: number of intervals to sample
    :return: historic average of the next time steps
    """
    r = np.zeros(data.shape)  # so that calculation can still be done on the first few cells
    # r = np.empty(data.shape)
    # r.fill(np.nan)
    for i in range(step + 1, len(r)):
        a_subset = data[i - step + 1:0:-step]  # +1 to take the historic average of the next time step
        if max_steps > 0:
            a_subset = a_subset[:max_steps]

        x = np.mean(a_subset, axis=0)
        r[i] = x

    return r


class HistoricAverage:
    def __init__(self, step=1, max_steps=0):
        self.step = step
        self.max_steps = max_steps
        self.fitted = False

    def fit(self, data):
        """
        determines the optimal
        """
        self.max_steps = get_max_steps(data, self.step)  # -1
        self.fitted = True
        print(f"fitted historic average: step ({self.step}) and max_steps ({self.max_steps})")

    def transform(self, data):
        if not self.fitted:
            raise RuntimeError(f"Model needs to be fitted to the data first: model.fitted = {self.fitted}")

        return historic_average(data, self.step, self.max_steps)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def __call__(self, data):
        return self.transform(data)


class BaseMovingAverage:
    def __init__(self, window_len, weights):
        self.window_len = window_len
        self.weights = weights

        self.f_avg = lambda x, wgt: np.average(x, axis=0, weights=wgt)

    def fn(self, data):
        return np.average(data, axis=0, weights=self.weights)

    def __call__(self, data):
        w = self.window_len
        r = np.empty(data.shape)
        r.fill(np.nan)
        # uncomment if we want to replace the nan values with smaller windowed averages
        # for i in range(w - 1):
        #     r[i] = np.average(data[:i + 1], axis=0, weights=self.weights[:i + 1])

        for i in range(w - 1, len(r)):
            x = data[(i - w + 1):i + 1]
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
        # self.activation = nn.ReLU()

    def forward(self, x):
        # out = self.activation(self.linear(x)) # linear regressor has no activation
        out = self.linear(x)
        return out


# TODO CONVERT FUNCTIONS INTO A CLASS
# still needs to do it only on training data average no the total
def get_historic_average(data, step=24):
    """
    a: array (N,d) or (N,d,d)
    step: the historic step the average is taken over, e.g. 24 for every 24 hours
    """
    ha = []  # historic averages

    for i in range(step):
        ha.append(data[i::step].mean(0))

    b = -1 * np.ones(data.shape)  # minus one there to check if anything went wrong
    for i in range(len(b)):
        b[i] = ha[i % step]

    return b


def rolling_apply(fun, data, w):  # this rolling average includes the current time step
    r = np.empty(data.shape)
    r.fill(np.nan)
    for i in range(w - 1, data.shape[0]):
        r[i] = fun(data[(i - w + 1):i + 1])
    return r


def rolling_avg(fun, data, weights):  # this rolling average includes the current time step
    w = len(weights)
    r = np.empty(data.shape)
    r.fill(np.nan)
    for i in range(w - 1, data.shape[0]):
        r[i] = fun(data[(i - w + 1):i + 1], weights)

    return r


if __name__ == "__main__":
    a = np.random.randn(100)

    folder = "./data/processed/"

    zip_file = np.load(folder + "generated_data.npz")

    grids = zip_file["crime_grids"]
    t_range = pd.read_pickle(folder + "t_range.pkl")

    grids = np.load(folder + 'crimes_grids.npy')
    f = lambda x: x.mean(0)
    avg = lambda x, w: np.average(x, axis=0, weights=w)  # x and weights should have same length
    # np.average()

    window = 4
    window_weights = np.exp(-10 / np.arange(window))

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

    grids_avg = rolling_avg(fun=avg, data=grids, weights=window_weights)

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
