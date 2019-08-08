import numpy as np
import pandas as pd


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
