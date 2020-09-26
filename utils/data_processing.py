from math import sin, cos, sqrt, atan2, radians

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp2d  # used for super resolution of grids
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import OneHotEncoder
from torch.autograd import Variable
from utils.utils import cut
from utils import deprecated


def sincos_vector(x):
    """
    Encode array x in sin cos vector representation
    x should be ndarray of integers
    given x is a cyclical series - we return a sin and cos vector of the cyclical series
    """
    x = x - x.min()
    x_max = x.max()

    sin_x = np.sin(2 * np.pi * (x % x_max) / x_max)
    cos_x = np.cos(2 * np.pi * (x % x_max) / x_max)
    return np.stack([sin_x, cos_x], axis=1)


def encode_sincos(X):
    n, d = X.shape
    vectors = []
    for i in range(d):
        vectors.append(sincos_vector(X[:, i]))

    return np.concatenate(vectors, axis=1)


def encode_time_vectors(t_range, month_divisions=10, year_divisions=10, kind='ohe'):
    """
    given t_range (datetime series)
    return E: H,D,DoW,isWeekend hot-encoded vector and sin(hour/24) cos(hour/24)
    as the columns features in numpy array
    kind: {'ohe' or 'sincos'} options - if sincos all vectors are represented and sin cos cycles, if ohe all vectors are
    one hot encoded and the hour of the day is sin cos vector representation.
    """
    time_frame = t_range.freqstr  # used to choose the external factors
    # External info
    is_weekend = t_range.dayofweek.to_numpy()
    is_weekend[is_weekend < 5] = 0
    is_weekend[is_weekend >= 5] = 1

    is_gte_24hours = "D" in time_frame or time_frame == "24H"

    df = pd.DataFrame({
        "hour": t_range.hour,
        "dayofweek": t_range.dayofweek,
        "is_weekend": is_weekend,
        "timeofmonth": cut(t_range.day / t_range.days_in_month, month_divisions),
        "timeofyear": cut(t_range.dayofyear / (366), year_divisions),
        #         "month": t_range.month,
    })

    if kind == "ohe":
        # OneHotEncoder for categorical data
        time_values = df[["dayofweek", "is_weekend", "timeofmonth", "timeofyear"]].values
        ohe = OneHotEncoder(categories="auto", sparse=False)  # It is assumed that input features take on values
        time_value_ohe = ohe.fit_transform(time_values)

        if not is_gte_24hours:  # only if we are working on a hourly time scale
            # Cyclical float values for hour of the day (so that 23:55 and 00:05 are more related to each other)
            hour_vec = sincos_vector(df.hour.values) / 2 + 0.5 # ensure all vectors are between 0 and 1
            time_vectors = np.hstack([time_value_ohe, hour_vec])
        else:
            time_vectors = time_value_ohe
    else: # option to encode all time information in a sin cos vector representation
        tv_sce = encode_sincos(df[["hour", "dayofweek", "timeofmonth", "timeofyear"]].values)/ 2 + 0.5 # ensure all vectors are between 0 and 1
        time_vectors = np.concatenate([tv_sce, is_weekend.reshape(-1,1)], axis=1)

    return time_vectors

def encode_category(series, categories):
    """
    encode_category maps categories to the index of the category in the list of categories for a series of values

    :param series: pandas.core.series.Series
    :param categories: list possible values category can take on
    :return: mapped series with category values going from 0 to len(categories) - 1

    If value is in series but not the 'category' index will be '-1'
    """
    index_map = {k: i for i, k in enumerate(categories)}

    return series.apply(lambda x: index_map.get(x, -1))

def freq_to_nano(freq: str):
    return pd.to_timedelta(freq).delta

def time_series_to_time_index(t_series: pd.Series, t_step: str = '1D', floor: bool = True):
    """
    Convert a datetime series into a more comparable number series (helps with counting and multi-d histograms)
    - starts at 0
    - 1 unit represents the t_step specified in the arguments

    Args:
    =====
    t_series: date time pandas series
    t_step: string describing time step
    floor: if the time index values should be floored to integers or kept as floats

    """

    dt_nano = freq_to_nano(t_step)

    result = t_series.astype('int64')
    t_min = result.min()

    result = (result - t_min) // dt_nano if floor else (result - t_min) / dt_nano
    return result


def get_period(a):
    n = len(a)
    corr_list = []
    for i in range(n):
        corr = np.correlate(a, np.roll(a, i))
        corr_list.append(corr)
    r = np.array(corr_list)[:, 0]
    r[r < 0] = 0
    period = np.argsort(r)[::-1][1]
    return period


def inv_weights(labels):
    """
    given 1D array of labels gives the inverse weights of the class labels
    :param labels:
    :return:
    """
    counts, _ = np.histogram(labels, bins=2)
    dist = 1 / counts
    dist = dist / np.sum(dist)
    return dist


def map_to_weights(labels):
    counts, _ = np.histogram(labels, bins=2)
    dist = 1 / counts
    dist = dist / np.sum(dist)
    hot_encoded = pd.get_dummies(pd.DataFrame(labels).loc[:, 0]).values
    weights = np.matmul(hot_encoded, dist)
    return weights

@deprecated
def encode_time_vectors_old(t_range):
    """
    given t_range (datetime series)
    return E: H,D,DoW,isWeekend hot-encoded vector and sin(hour/24) cos(hour/24)
    as the columns features in numpy array
    """
    time_frame = t_range.freqstr  # used to choose the external factors
    # External info
    is_weekend = t_range.dayofweek.to_series()
    is_weekend[is_weekend < 5] = 0
    is_weekend[is_weekend >= 5] = 1

    is_gte_24hours = "D" in time_frame or time_frame == "24H"

    df = pd.DataFrame({"datetime": t_range,
                       "hour": t_range.hour,
                       "dow": t_range.dayofweek,
                       "day": t_range.day,
                       "month": t_range.month,
                       "is_weekend": is_weekend})

    if is_gte_24hours:  # working on daily time slots
        time_values = df[["month", "day", "dow", "is_weekend"]].values  # swap hour for month
        time_values[:, 0] = time_values[:, 0] - 1  # minus one for month OHE expects range [0,n_values)
    else:  # working on hourly time slots
        # left out month because our hourly data isn"t more than a year
        time_values = df[["hour", "day", "dow", "is_weekend"]].values

    time_values[:, 1] = time_values[:, 1] - 1  # minus one OHE expects range [0,n_values)

    # OneHotEncoder for categorical data
    ohe = OneHotEncoder(categories="auto", sparse=False)  # It is assumed that input features take on values
    # in the range[0, n_values). Thus days minus 1
    time_value_ohe = ohe.fit_transform(time_values)

    if not is_gte_24hours:  # only if we are working on a hourly time scale
        # Cyclical float values for hour of the day (so that 23:55 and 00:05 are more related to each other)
        sin_hour = np.sin(2 * np.pi * (time_values[:, 2] % 24) / 24)
        sin_hour = np.reshape(sin_hour, (len(sin_hour), 1))
        cos_hour = np.cos(2 * np.pi * (time_values[:, 2] % 24) / 24)
        cos_hour = np.reshape(cos_hour, (len(cos_hour), 1))

        time_vectors = np.hstack([time_value_ohe, cos_hour, sin_hour])
    else:
        time_vectors = time_value_ohe

    return time_vectors


def set2d(x):
    return set(map(tuple, x))


def get_E(t_range):
    """
    given t_range (datetime series)
    return E: H,D,DoW,isWeekend hot encoded vector and sin(hour/24) cos(hour/24)
    as the columns features
    """
    time_frame = t_range.freqstr  # used to choose the external factors
    # External info
    is_weekend = t_range.dayofweek.to_series()
    is_weekend[is_weekend < 5] = 0
    is_weekend[is_weekend >= 5] = 1

    df = pd.DataFrame({"datetime": t_range,
                       "hour": t_range.hour,
                       "dow": t_range.dayofweek,
                       "day": t_range.day,
                       "month": t_range.month,
                       "is_weekend": is_weekend})

    if time_frame == "D":  # working on daily time slots
        A = df[["month", "day", "dow", "is_weekend"]].as_matrix()  # swap hourr for month
        A[:, 0] = A[:, 0] - 1  # minus one for month OHE expects range [0,n_values)
    else:  # working on hourly time slots
        A = df[["hour", "day", "dow",
                "is_weekend"]].as_matrix()  # left out month because our hourly data isn"t more than a year

    A[:, 1] = A[:, 1] - 1  # minus one OHE expects range [0,n_values)

    # OneHotEncoder for categorical data
    ohe = OneHotEncoder(
        sparse=False)  # It is assumed that input features take on values in the range[0, n_values). Thus days minus 1
    A_ohe = ohe.fit_transform(A)

    if time_frame != "D":  # only if we are working on a hourly time scale
        # Cyclical float values for hour of the day (so that 23:55 and 00:05 are more related to each other)
        sin_hour = np.sin(2 * np.pi * (A[:, 2] % 24) / 24)
        sin_hour = np.reshape(sin_hour, (len(sin_hour), 1))
        cos_hour = np.cos(2 * np.pi * (A[:, 2] % 24) / 24)
        cos_hour = np.reshape(cos_hour, (len(cos_hour), 1))

        E = np.hstack([A_ohe, cos_hour, sin_hour])
        E = Variable(torch.FloatTensor(E))
    else:
        E = Variable(torch.FloatTensor(A_ohe))

    return E


def get_S(data, do_superres=False, do_cumsum=False):
    if do_superres:
        N, W, H = data.shape
        x = np.zeros((N, 2 * W + 1, 2 * H + 1))
        for i in range(N):
            x[i] = upsample(data[i])
        S = Variable(torch.FloatTensor(x))
    else:
        S = Variable(torch.FloatTensor(data))

    if do_cumsum:
        S = periodic_cumsum(S)

    return S


def get_rev_S(S, do_superres, do_cumsum):
    """
    reverse S to get original data
    """
    rev = S.cpu()
    if do_cumsum:
        rev = reverse_cumsum(rev)

    if do_superres:
        rev = downsample(rev)

    return rev


@deprecated
def get_trans_mat_d2s(a, threshold=0):
    """
    Warning: has been deprecated
    a array shapped (N, W, H)
    sum over all time should be above this threshold
    """

    N, W, H = a.shape
    a = np.reshape(a, (N, W * H))
    dd = np.sum(a, 0)
    dd[dd > threshold] = 1
    dd = np.reshape(dd, (len(dd), 1))
    T = []
    for i in range(len(dd)):
        if dd[i] != 0:
            f = np.zeros(len(dd))
            f[i] = 1
            T.append(f)
    T = np.array(T).T
    return T


def cluster2coord(a, centers):
    """
    Arguments
    =========
    a: array with cluster number (N,n_clusters)
    centers: list with cluster coordinates (N,d)

    Returns
    =======
    poi: each number in array a translated to centre coordinates (N,d)
    i: indices of the
    """

    i = np.argwhere(a > 0).flatten()
    poi = centers[i]

    return poi, i


def get_times(a):
    """
    given array with true-false occurrences return indices of all true occurrences
    """
    t = []
    for i in range(len(a)):
        if a[i]:
            t.append(i)
    return t


def get_dead_cells(a):  # finding all the living cells
    """
    given matrix a (N,d,d)
    return coords (x,y) of dead cells, i.e. cells (d,d) that change over N
    """
    dead_cells = []
    for i in range(a.shape[-2]):
        for j in range(a.shape[-1]):
            if np.max(a[:, i, j], axis=0) == 0:
                dead_cells.append((i, j))

    print("Percentage of cells that are living: ", 100 * len(dead_cells) / (a.shape[-1] ** 2))

    return dead_cells


def get_living_cells(a):  # finding all the living cells
    """
    given matrix a (N,d,d)
    return coords (x,y) of living cells, i.e. cells (d,d) that change over N
    """
    living_cells = []
    for i in range(a.shape[-2]):
        for j in range(a.shape[-1]):
            if np.max(a[:, i, j], axis=0) != 0:
                living_cells.append((i, j))

    print("Percentage of cells that are living: ", 100 * len(living_cells) / (a.shape[-1] ** 2))

    return living_cells


def unique_for_live_cell(a):  # unique values catering for dead cells
    """
    given matrix a (N,d,d)
    return values, counts
    """

    counts = []
    for i in range(int(np.max(a)) + 1):
        counts.append(0)

    for i in range(a.shape[-2]):
        for j in range(a.shape[-1]):
            if np.max(a[:, i, j], axis=0) != 0:
                V, C = np.unique(a[:, i, j], return_counts=True)
                for v, c in zip(V, C):
                    counts[int(v)] = counts[int(v)] + c

    values = np.arange(len(counts))
    return values, counts


def get_moving_square_data(N=5000, n=33, s=1, gaus=True):
    """
    N: number of grids
    n: grid width/length
    """
    dx = 1
    dy = 1
    x = np.random.randint(n)
    y = np.random.randint(n)

    data = np.zeros((N, n, n))
    for i in range(N):
        m = np.zeros((n, n))

        try:
            m[int(x), int(y)] = 1
        except:
            print("e:", int(x), int(y), dx, dy)

        if (x + dx >= n) or (x + dx < 0):
            dx = -1 * dx
        #             dx = dx*(np.random.rand(1) + 0.5)
        if (y + dy >= n) or (y + dy < 0):
            dy = -1 * dy
            dy = dy * (np.random.rand(1) + 0.5)

        x += dx
        y += dy
        if gaus:
            data[i] = gaussian_filter(m, s)
        else:
            data[i] = m
    return data


# TODO: Put in separate preproc module
def norm_minmax(a, minmax=[0, 1]):  # should add way to reverse - maybe store in data loader
    a = a - a.min()
    a = a / a.max()
    a = a * (minmax[1] - minmax[0])
    a = a + minmax[0]

    return a


def norm_meanstd(a):
    a = a - a.mean()
    a = a / a.std()

    return a


@deprecated  # instead use pad2d
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get("padder", 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


@deprecated  # instead use pad2d
def pad(a, edge_size=1, pad_value=0):
    return np.pad(a, edge_size, pad_with, padder=pad_value)


def pad2d(a, value=0, size=1):
    h, w = list(np.shape(a))[-2:]
    r = np.ones((h + 2 * size, w + 2 * size)) * value

    r[size:-size, size:-size] = a

    return r


def pad4d(a, value=0, size=1):
    if size <= 0:
        return a

    n, c, h, w = np.shape(a)
    r = np.ones((n, c, h + 2 * size, w + 2 * size)) * value

    r[:, :, size:-size, size:-size] = a

    return r


def crop4d(a, size):
    """
    serves as inverse of pad4d
    """
    if size <= 0:
        return a

    return a[:, :, size:-size, size:-size]


def upsample(a):
    """
    upscale 2D array a: from size n to 2n + 1
    [xxxx] to [oxoxoxoxo]
    where the the edges are linearly interpolated between intput edge value and zero
    """
    a = pad(a)
    n, d = np.shape(a)

    nnew = 2 * n - 1
    xnew = np.arange(nnew)
    x = xnew[0::2]

    dnew = 2 * d - 1
    ynew = np.arange(dnew)
    y = ynew[0::2]

    f = interp2d(y, x, a)
    anew = f(ynew, xnew)

    return anew[1:-1, 1:-1]


def downsample(a):
    """
    Downsample:
    [oxoxoxoxo] to [xxxx]
    """
    if len(a.shape) > 2:
        return a[:, 1:-1:2, 1:-1:2]
    else:
        return a[1:-1:2, 1:-1:2]


def periodic_cumsum(s, T=24):
    """
    s is torch variable tensor representing upsampled cumsum data
    r is torch variable tensor representing inverse cumsum data
    T is cumsum period
    """
    r = torch.zeros_like(s)

    if len(s) % T == 0:
        ns = len(s) // T
    else:
        ns = len(s) // T + 1

    for n in range(ns):
        r[n * T:(n + 1) * T] = torch.cumsum(s[n * T:(n + 1) * T], 0)

    return r


def reverse_cumsum(s, T=24):
    """
    s is torch variable tensor representing upsampled cumsum data
    r is torch variable tensor representing inverse cumsum data
    T is cumsum period
    """
    s = s.data.numpy()
    r = np.zeros(s.shape)

    if len(s) % T == 0:
        ns = len(s) // T
    else:
        ns = len(s) // T + 1

    for n in range(ns):
        r[n * T] = s[n * T]
        r[n * T + 1:(n + 1) * T] = np.diff(s[n * T:(n + 1) * T], axis=0)  # because torch doesnt have np.diff
    return Variable(torch.FloatTensor(r))


def make_grid(A, t_size, x_size, y_size):
    """
    Note: np.histogramdd() can also be used - it"s a bit
    A: matrix with time, x and y coordinates
    returns grid matrix with each index filled where crimes occurred
    X and Y axis are swapped to make displaying easier
    """
    grids = np.zeros((t_size, y_size, x_size))
    for a in A:
        grids[a[0], y_size - 1 - a[2], a[1]] += 1

    return grids


def upsample_interpolate(a, scale=2, interpolate_kind="linear"):
    """
    might rather use cv2.pyrUp and pyrDown - seem to give less issues
    upsamlpe 2d grid array a
    scale: can be one value or tuple (nscale, dscale), default 2
    kind: "linear", "cubic", "quintic"
    """
    a = pad(a, 2)

    n, d = np.shape(a)

    z = a
    x = np.arange(0, 1, 1 / n) + 0.5 / n
    y = np.arange(0, 1, 1 / d) + 0.5 / d

    f = interp2d(x, y, z, kind=interpolate_kind)

    if type(scale) is tuple:
        xnew = np.linspace(0, 1, n * scale[0])
        ynew = np.linspace(0, 1, d * scale[1])

        xnew = np.arange(0, 1, 1 / (n * scale[0])) + 0.5 / (n * scale[0])
        ynew = np.arange(0, 1, 1 / (d * scale[1])) + 0.5 / (d * scale[1])
    else:
        xnew = np.arange(0, 1, 1 / (n * scale)) + 0.5 / (n * scale)
        ynew = np.arange(0, 1, 1 / (d * scale)) + 0.5 / (d * scale)
    znew = f(xnew, ynew)

    return znew[4:-4, 4:-4]


# calculate the distance between to coordinates #should actually be done with maps api to be more accurate
def get_distance_from_coords(lat1, lon1, lat2, lon2):
    """
    args in degrees
    returns distance in km
    """
    # approximate radius of the earth in km
    R = 6373.0

    # longitude: E to W; latitude: N to S
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance


# COUNT/BIN 2D ARRAYS FUNCTIONS

def get_interval_times(a):
    """
    given array with true false occurrences given number of indices between each true occurence
    """
    t = []
    prev_idx = 0
    cur_idx = 0
    for cur_idx in range(len(a)):
        if a[cur_idx]:
            t.append(cur_idx - prev_idx)
            prev_idx = cur_idx
    return t


def find_max_coord(grid):
    """
    find the max 2d coordinate in a grid
    """
    x_size, y_size = grid.shape
    m = np.max(grid)
    c = []
    for i in range(x_size):
        for j in range(y_size):
            if grid[i][j] == m:
                c = [i, j]
    return c


def sum_subset(a, step):
    """
    used to sum subset of the grids to change the time scale
    """
    r = []
    for i in range(0, len(a), step):
        r.append(np.sum(a[i:i + step], axis=0))
    return np.array(r)


def digitize2Dpoint(point, x_bins, y_bins):
    """
    return the cell, i.e. the x and y indices of the bin
    note: index 0 is below lowest bin value and index -1 is above highest bin values
    """

    n_x_bins = len(x_bins)
    n_y_bins = len(y_bins)

    px = point[0]
    py = point[1]

    cell = [0, 0]  # initialize cell

    for i in range(n_x_bins - 1):
        if x_bins[i] <= px < x_bins[i + 1]:
            cell[0] = i + 1

    for j in range(n_y_bins - 1):
        if y_bins[j] <= py < y_bins[j + 1]:
            cell[1] = j + 1

    # TODO: EDGE CASES NEED WORK
    if px < np.min(x_bins):
        cell[0] = 0
    elif px > np.max(x_bins):
        cell[0] = len(x_bins)

    if py < np.min(y_bins):
        cell[1] = 0
    elif py > np.max(y_bins):
        cell[1] = len(y_bins)

    return np.array(cell)


def digitize2D(points, x_bins, y_bins):
    """
    points is array of points: N X 2
    return the cell, i.e. the x and y indices of the bin
    note: index 0 is below lowest bin value and index -1 is above highest bin values
    """
    bin_indices = []

    for point in points:
        index = digitize2Dpoint(point, x_bins, y_bins)
        bin_indices.append(index)

    return np.array(bin_indices)


def count2Dvalues(points, x_bins, y_bins):
    """
    return 2D array with count of values in cells
    TODO: CHANGE THE COUNT +1 TO A VARIABLE NUMBER
    """
    cells = digitize2D(points, x_bins, y_bins)

    counts = np.zeros((len(x_bins) + 1, len(y_bins) + 1))

    for i in range(len(cells)):
        counts[cells[i][0], cells[i][1]] += 1

    # get correct orientation
    counts = np.fliplr(counts).T
    return counts


def batchify(a, batch_size):
    """
    return matrix (batch_size, len(a)//batch_size)
      note: sequencing should be handled in the training loop
    """
    row_len = (len(a) // batch_size)

    if len(a.shape) > 1:
        n_feats = a.shape[-1]
        r = np.reshape(a[:row_len * batch_size], (batch_size, row_len, n_feats))
    else:
        r = np.reshape(a[:row_len * batch_size], (batch_size, row_len))
    return r


def sequencify(a, seq_len):  # NB: SEQUENCES ARE OVERLAPPING
    """
    returns matrix with sequences that differ with one time step
    """
    r = []
    for i in range(len(a) - seq_len):
        r.append(a[i:i + seq_len])

    return np.array(r)


def threshold(a, thresh=1):
    """
    cap all vals above or equal to 1 to 1 and all below to zero
    """
    r = a >= thresh
    return r.astype("int")


def concentric_squares(a):
    """
    return a array with index 0 being sum of values of outer square
    """
    nrows, ncols = np.shape(a)
    if nrows != ncols:
        print("Error: should be square matrix.")
    elif nrows % 2 == 0:
        print("Error: should be odd number of rows")
    else:
        nsquares = (nrows // 2) + 1
        sums = np.zeros(nsquares)
        ret = np.zeros(nsquares)
        sums[0] = np.sum(a)
        for i in range(1, nsquares):
            sums[i] = np.sum(a[i:-i, i:-i])
        for i in range(nsquares - 1):
            ret[i] = sums[i] - sums[i + 1]
        ret[-1] = sums[-1]

        return ret
