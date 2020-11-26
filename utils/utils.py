import json
from datetime import datetime
from time import time
from warnings import warn
import numpy as np
import os
import pandas as pd
from pprint import pformat
import pandas as pd


def cmi_name(temporal_variables):
    cond_var_map = {
        'Hour': 'H_t',
        'Day of Week': 'DoW_t',
        'Time of Month': 'ToM_t',
        'Time of Year': 'ToY_t',
    }

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


def get_data_sub_paths():
    data_sub_paths = os.listdir("./data/processed/")
    if '.DS_Store' in data_sub_paths:
        data_sub_paths.remove('.DS_Store')

    return data_sub_paths


def by_ref(ref):
    """
    Get the data_sub_paths by the reference code
    :param ref: reference code: 3 letter code
    :return: list of subpaths ending in ref
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


def read_json(file_name):
    with open(file_name, "r") as fp:
        r = json.load(fp)
    return r


def write_json(data, file_name):
    """
    Saves dictionary data as json file
    :param data: dictionary
    :param file_name: file name or path to e saved
    :return:
    """
    with open(file_name, "w") as fp:
        json.dump(data, fp)


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
