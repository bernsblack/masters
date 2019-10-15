import json
from datetime import datetime
from warnings import warn
from time import time, sleep


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
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()


def read_json(file_name):
    with open(file_name, "r") as fp:
        r = json.load(fp)
    return r


def write_json(data, file_name):
    with open(file_name, "w") as fp:
        json.dump(data, fp)
