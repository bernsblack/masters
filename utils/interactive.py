import numpy as np
from utils import ffloor, fceil
from pprint import pformat
from pandas import Timedelta

class State:
    """
    State object to make interactive plots easier.
    Callbacks can be added to be triggered when state fields are updated.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v
        self.callbacks = []  # inner repr of the update callback
        self.__recursion_guard = 0

    def __repr__(self):
        display = {**self.__dict__}
        display.pop("_State__recursion_guard", None)
        return pformat(display)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value
        self.__update()

    def on_update(self, callback, append=False):
        """
        set the callback to be called when a values is set using state['a'] = 1 format
        callback: function that takes State class as input
        """
        if append:
            self.callbacks.append(callback)
        else:
            self.callbacks = [callback]

    def update(self, **kwargs):
        """
        set/update multiple values before calling the on_update callback
        """
        for k, v in kwargs.items():
            self.__dict__[k] = v
        self.__update()

    def __update(self):
        self.__recursion_guard += 1
        if self.__recursion_guard < 2:
            for fn in self.callbacks:
                fn(self)
        self.__recursion_guard = 0


def state_to_conf(state):
    conf = dict(
        start_date=state.date_min.strftime("%Y-%m-%d"),
        end_date=(state.date_max + Timedelta(state.freq)).strftime("%Y-%m-%d"),
        dT=state.freq,
        x_scale=int(state.dlon / 0.001),
        y_scale=int(state.dlat / 0.001),
        lat_max=np.round(state.lat_max, decimals=3),
        lat_min=np.round(state.lat_min, decimals=3),
        lon_max=np.round(state.lon_max, decimals=3),
        lon_min=np.round(state.lon_min, decimals=3),
        root="./"
    )

    return conf

# rename and move to utils
def new_int_bins(int_min, int_max):
    """
    Will create a array of bin values so that the integers from int_min, int_max fall exactly between the bins.
    Example:
    new_int_bins(int_min=0, int_max=4) => array([0, 1, 2, 3, 4, 5])
    with lower bin values inclusive and upper limits exclusive, i.e. [lower, upper)

    :param int_min: minimum value to be binned (inclusive)
    :param int_max: maximum value to be binned (inclusive)
    :return: array of bin edges to ensure all integers between int_min and int_max inclusive are binned separately.
    """
    return np.arange(int_min, int_max + 2)


def filter_frame(data_frame, state):
    """
    filter df given the values in state to get a subset of data
    """
    # spatial filters
    lon_mask = (data_frame.Longitude >= state.lon_min) & (data_frame.Longitude <= state.lon_max)
    lat_mask = (data_frame.Latitude >= state.lat_min) & (data_frame.Latitude <= state.lat_max)

    # time filters
    date_mask = (data_frame.Date >= state.date_min) & (data_frame.Date <= state.date_max)

    # crime type filters
    crime_types_mask = data_frame['Primary Type'].isin(state.crime_types)

    filter_mask = date_mask & crime_types_mask & lon_mask & lat_mask

    return data_frame[filter_mask]


def get_total_counts(data_frame, state, date_range):
    """
    data_frame: dataframe of crime incidents
    returns total_counts_y, total_counts_x
    """
    if len(data_frame) > 0:
        nt = int(np.ceil(data_frame.t.max()))
        tbins = new_int_bins(data_frame.t.min(), data_frame.t.max())

        #  total counts line/curve
        total_counts, edges = np.histogram(data_frame.t, bins=tbins)
        total_counts_y = total_counts
        total_counts_x = date_range[edges[:-2]]
    else:
        total_counts_y = np.zeros(len(date_range))
        i, j = state.date_indices
        total_counts_x = date_range[i, j]
    return total_counts_y, total_counts_x


def new_bins(data_frame, state):
    xbins = np.arange(
        start=ffloor(state.lon_min, state.dlon),
        stop=fceil(state.lon_max, state.dlon),
        step=state.dlon,
    )
    nx = len(xbins)

    ybins = np.arange(
        start=ffloor(state.lat_min, state.dlat),
        stop=fceil(state.lat_max, state.dlat),
        step=state.dlat,
    )
    ny = len(ybins)

    nt = int(np.ceil(data_frame.t.max()))
    tbins = new_int_bins(data_frame.t.min(), data_frame.t.max())

    nc = len(state.crime_types)
    cbins = np.arange(0, nc + 1, 1)

    return tbins, cbins, ybins, xbins


def bin_data_frame(data_frame, state):
    """
    will bin the data from into a N,C,H,W grid depending on the state
    """
    bins = new_bins(data_frame, state)

    binned_data, bins = np.histogramdd(
        sample=data_frame[['t', 'c', 'Latitude', 'Longitude']].values,
        bins=bins,
    )

    return binned_data, bins


def get_mean_map(data_frame, state):
    """
    will be a heat map of the means with the min-max lat and lon min and remaining constant
    """
    xbins = np.arange(
        start=ffloor(state.total_lon_min, state.dlon),
        stop=fceil(state.total_lon_max, state.dlon),
        step=state.dlon,
    )
    nx = len(xbins)

    ybins = np.arange(
        start=ffloor(state.total_lat_min, state.dlat),
        stop=fceil(state.total_lat_max, state.dlat),
        step=state.dlat,
    )
    ny = len(ybins)

    nt = int(np.ceil(data_frame.t.max()))
    tbins = new_int_bins(data_frame.t.min(), data_frame.t.max())

    nc = len(state.crime_types)
    cbins = np.arange(0, nc + 1, 1)

    bins = tbins, cbins, ybins, xbins

    binned_data, bins = np.histogramdd(
        sample=data_frame[['t', 'c', 'Latitude', 'Longitude']].values,
        bins=bins,
    )

    mean_map = binned_data.sum(1).mean(0)
    return mean_map, xbins, ybins
