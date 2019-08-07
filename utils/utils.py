import json
from datetime import datetime


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
