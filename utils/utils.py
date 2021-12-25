import json
import logging
import os
from datetime import datetime
from time import time
from typing import List
from warnings import warn

import numpy as np
import pandas as pd
import torch
from torch import nn


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def to_snake(text):
    return text.lower().replace(' ', '_')


def topk_indices(data, k):
    """
    :param data: data array
    :param k: integer value of top values
    :return: top k indices in array
    """
    return np.argsort(-data)[:k]


def topk(data, k):
    """
    :param data: data array
    :param k: integer value of top values
    :return: top k values in array
    """
    return data[np.argsort(-data)[:k]]


def shift(xs, n, fill=np.nan):
    e = np.empty(xs.shape)
    e.fill(fill)
    if n >= 0:
        e[n:] = xs[:-n]
    else:
        e[:n] = xs[-n:]
    return e


def set_system_seed(seed=0):
    logging.info(f"Set system seed to {seed}")
    torch.manual_seed(seed)  # sets seed for cpu and gpu
    np.random.seed(seed)


def to_title(str_list):
    return list(map(lambda x: x.title(), str_list))


def load_total_counts(folder_name: str) -> pd.DataFrame:
    df = pd.read_pickle(f"./data/processed/{folder_name}/total_counts_by_type.pkl")
    # raise Exception("Reload all Total files and saved after changing: df.freq = df.freqstr")

    # todo: fix issue where reindex breaks runtime
    # newest change - breaks for some reason
    df['TOTAL'] = df.sum(axis=1)
    df = df.reindex(columns=np.roll(df.columns, 1))
    df.columns = to_title(df.columns)
    return df


def cmi_name(temporal_variables):
    cond_var_map = {
        'Hour': 'H_t',
        'Day of Week': 'DoW_t',
        'Time of Month': 'ToM_t',
        'Time of Year': 'ToY_t',
    }

    assert isinstance(temporal_variables, list) or isinstance(temporal_variables, tuple) == True, \
        f"temporal_variables must be list or tuple not {type(temporal_variables)}"

    return ",".join([cond_var_map[k] for k in temporal_variables])


def drop_nan(x):
    return x[~np.isnan(x)]


def cut(x, bins=10):
    """
    Shorthand cut function that bins the values of the array and returns the index of the bin the array value falls in.
    :param x: array
    :param bins: number of segments the array should be divided in x.min and x.max being the limits of the bins
    :return:
    """
    return pd.cut(x=x, bins=bins, labels=False)


def ffloor(value, delta):
    """
    floors value to the nearest multiple of delta where delta can be a float
    """
    return np.floor(value / delta) * delta


def fceil(value, delta):
    """
    ceils value to the nearest multiple of delta where delta can be a float
    """
    return np.ceil(value / delta) * delta


def get_data_resolutions():
    def parse_st_resolution(s):
        dt, s = s.split("T")[-1].split('H')
        dx, dy, s = s.split("X")[-1].split('M')
        _, dy = dy.split("Y")
        _, start_date, end_date = s.split("_")
        r = {"Hours": int(dt),
             "Metres (x-direction)": int(dx),
             "Metres (y-direction)": int(dy)}
        return r

    df = pd.DataFrame(list(map(parse_st_resolution, get_data_sub_paths())))
    df = df.sort_values(["Hours", "Metres (x-direction)"], ascending=[False, True])
    return df


def get_data_sub_paths() -> List[str]:
    data_sub_paths = os.listdir("./data/processed/")
    if '.DS_Store' in data_sub_paths:
        data_sub_paths.remove('.DS_Store')

    return data_sub_paths


def by_ref(ref: str) -> List[str]:
    """
    Get the data_sub_paths by the reference code
    :param ref: reference code: 3 letter code
    :return: list of sub-paths ending in ref
    """
    return [c for c in get_data_sub_paths() if c.endswith(ref)]


def pmax(*args):
    """
    prints the shapes of the arguments
    """
    for arg in args:
        print(np.max(arg))


def pmin(*args):
    """
    prints the shapes of the arguments
    """
    for arg in args:
        print(np.min(arg))


def pmean(*args):
    """
    prints the shapes of the arguments
    """
    for arg in args:
        print(np.mean(arg))


def pshape(*args):
    """
    prints the shapes of the arguments
    """
    for arg in args:
        print(np.shape(arg))


def if_none(a, b):  # common null checker
    return b if a is None else a


def timeit(func):
    """
    Profiling wrapper to time functions
    :param func: function that needs to be times
    :return: return value of the function
    """

    def wrapper(*args):
        start = time()
        a = func(*args)
        stop = time()
        print(f"{func.__name__}: {stop - start}")
        return a

    return wrapper


def deprecated(func):
    def wrapper():
        warn(f"{func.__name__} has been deprecated", DeprecationWarning)
        return func()

    return wrapper


class Timer:
    """
    Timer class to easily log the time codes take to execute
    """

    def __init__(self):
        """
        initialises the cache to now
        """
        self.cache = datetime.now()

    def check(self):
        """
        :return: duration since previous check or since last reset
        """
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        """
        :return: resets the cached time to now
        """
        self.cache = datetime.now()


def read_json(file_name: str):
    with open(file_name, "r") as fp:
        r = json.load(fp)
    return r


def write_json(data, file_name: str):
    """
    Saves dictionary data as json file
    :param data: dictionary
    :param file_name: file name or path to e saved
    :return:
    """
    with open(file_name, "w") as fp:
        json.dump(data, fp)


def write_txt(data: str, file_name: str):
    """
    Saves string data as text file
    :param data: str
    :param file_name: file name or path to e saved
    :return:
    """
    with open(file_name, "w") as fp:
        fp.write(data)


def read_txt(data, file_name):
    """
    Reads string data as text file
    :param data: str
    :param file_name: file name or path to e saved
    :return:
    """
    with open(file_name, "r") as fp:
        text = fp.read(data)
    return text


def describe_array(a):
    """

    :param a: array we need some common statistics on
    :return string format data
    """
    d = {
        "min": np.min(a),
        "max": np.max(a),
        "mean": np.mean(a),
        "std": np.std(a),
        #             "median":np.median(a),
        "shape": np.shape(a),
        #             "nunique":len(np.unique(a)),
        # dist becomes an issue with continuous variables
        #             "dist": dict(
        #                 zip(v,c))
    }
    return d
    # return f"____________________________________________________________\n{pformat(d)}\n" + "____________________________________________________________"


def is_all_integer(a):
    return np.sum(a - np.round(a)) == 0
