#!python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from math import sin, cos, sqrt, atan2, radians
from scipy.interpolate import interp2d  # used for super resolution of grids
import torch
from torch.autograd import Variable
from scipy.ndimage import gaussian_filter
from time import time
import shutil
from mpl_toolkits.axes_grid1 import make_axes_locatable
import io
import base64
from IPython.display import HTML
import json
from datetime import datetime


class Timer:
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
    with open(file_name, 'r') as fp:
        r = json.load(fp)
    return r


def write_json(data, file_name):
    with open(file_name, 'w') as fp:
        json.dump(data, fp)


def encode_time_vectors(t_range):
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

    df = pd.DataFrame({'datetime': t_range, 'hour': t_range.hour,
                       'dow': t_range.dayofweek, 'day': t_range.day, 'month': t_range.month, 'is_weekend': is_weekend})

    if time_frame == 'D':  # working on daily time slots
        A = df[['month', 'day', 'dow', 'is_weekend']].values  # swap hourr for month
        A[:, 0] = A[:, 0] - 1  # minus one for month OHE expects range [0,n_values)
    else:  # working on hourly time slots
        A = df[['hour', 'day', 'dow',
                'is_weekend']].values  # left out month because our hourly data isn't more than a year

    A[:, 1] = A[:, 1] - 1  # minus one OHE expects range [0,n_values)

    # OneHotEncoder for categorical data
    ohe = OneHotEncoder(categories='auto', sparse=False)  # It is assumed that input features take on values
    # in the range[0, n_values). Thus days minus 1
    A_ohe = ohe.fit_transform(A)

    if time_frame != 'D':  # only if we are working on a hourly time scale
        # Cyclical float values for hour of the day (so that 23:55 and 00:05 are more related to each other)
        sin_hour = np.sin(2 * np.pi * (A[:, 2] % 24) / 24)
        sin_hour = np.reshape(sin_hour, (len(sin_hour), 1))
        cos_hour = np.cos(2 * np.pi * (A[:, 2] % 24) / 24)
        cos_hour = np.reshape(cos_hour, (len(cos_hour), 1))

        E = np.hstack([A_ohe, cos_hour, sin_hour])
    #     E = Variable(torch.FloatTensor(E))
    else:
        #     E = Variable(torch.FloatTensor(A_ohe))
        E = A_ohe

    return E


def set2d(x):
    return set(map(tuple, x))


# tood should rather use tensor.to_device(device), where device was previously set according to the machines.is_cuda_available
def to_cuda(obj):
    if torch.cuda.is_available():
        return obj.cuda()
    else:
        return obj


def imshow(a, ax, title=''):
    """
    a: 2D array (Image)
    ax: subplot axis
    """
    im = ax.imshow(a, cmap='viridis')
    ax.set_title(title)
    #  ax.set_xticks(np.arange(a.shape[-2]))
    #  ax.set_yticks(np.arange(a.shape[-1]))
    plt.grid('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, cmap='viridis')


def plot_compare(a, b, times, relative=True, a_title='A', b_title='B'):
    """
    plot to grids next to each other given a range of matching times
    """
    for i in range(len(a)):
        plt.figure(figsize=(18, 6))
        plt.suptitle(times[i])
        plt.subplot(1, 3, 1)
        plt.title(a_title)
        if relative:
            plt.imshow(a[i], cmap='viridis')
        else:
            plt.imshow(a[i], cmap='viridis', vmin=0, vmax=1)
        plt.xticks(np.arange(a.shape[-1]))
        plt.yticks(np.arange(a.shape[-1]))

        plt.subplot(1, 3, 2)
        plt.title(b_title)
        if relative:
            plt.imshow(b[i], cmap='viridis')
        else:
            plt.imshow(b[i], cmap='viridis', vmin=0, vmax=1)
        plt.xticks(np.arange(b.shape[-1]))
        plt.yticks(np.arange(b.shape[-1]))

        plt.subplot(1, 3, 3)
        plt.title("Error")
        if relative:
            plt.imshow(a[i] - b[i], cmap='viridis')
        else:
            plt.imshow(a[i] - b[i], cmap='viridis', vmin=0, vmax=1)
        plt.xticks(np.arange(b.shape[-1]))
        plt.yticks(np.arange(b.shape[-1]))

        plt.show()
        print(
            '============================================================================================================================')


def play_video(f):
    video = io.open(f, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data="""<video alt="test" controls>
                  <source src="data:video/mp4;base64,{0}" type="video/mp4" />
               </video>""".format(encoded.decode('ascii')))


def plot_targ_pred_over_time(trg, prd):
    """
    fig setup should be done before calling function
    function scatters the values of the target and predicted values
    and plots the intensity curve
    """
    indices = get_times(trg)
    plt.scatter(indices, prd[indices], s=20)
    plt.scatter(indices, trg[indices], s=20, c='r', marker='x')
    plt.plot(prd, alpha=0.7)


def get_E(t_range):
    """
    given t_range (datetime series)
    return E: H,D,DoW,isWeekend hotendcoded vector and sin(hour/24) cos(hour/24)
    as the columns features
    """
    time_frame = t_range.freqstr  # used to choose the external factors
    # External info
    is_weekend = t_range.dayofweek.to_series()
    is_weekend[is_weekend < 5] = 0
    is_weekend[is_weekend >= 5] = 1

    df = pd.DataFrame({'datetime': t_range, 'hour': t_range.hour,
                       'dow': t_range.dayofweek, 'day': t_range.day, 'month': t_range.month, 'is_weekend': is_weekend})

    if time_frame == 'D':  # working on daily time slots
        A = df[['month', 'day', 'dow', 'is_weekend']].as_matrix()  # swap hourr for month
        A[:, 0] = A[:, 0] - 1  # minus one for month OHE expects range [0,n_values)
    else:  # working on hourly time slots
        A = df[['hour', 'day', 'dow',
                'is_weekend']].as_matrix()  # left out month because our hourly data isn't more than a year

    A[:, 1] = A[:, 1] - 1  # minus one OHE expects range [0,n_values)

    # OneHotEncoder for categorical data
    ohe = OneHotEncoder(
        sparse=False)  # It is assumed that input features take on values in the range[0, n_values). Thus days minus 1
    A_ohe = ohe.fit_transform(A)

    if time_frame != 'D':  # only if we are working on a hourly time scale
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


def get_trans_mat_d2s(a, threshold=0):
    """
    a array shapped (N,d,d)
    sum over all time should be above this threshold
    """
    N, d, d = a.shape
    a = np.reshape(a, (N, d * d))
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


def plot_day(a, times, relative=True):
    """
    display array a (24,rows,cols)
    """
    plt.figure(figsize=(25, 15))
    for i in range(24):
        plt.subplot(4, 6, i + 1)
        plt.title(times[i])
        if relative:
            plt.imshow(a[i], cmap='viridis')
        else:
            plt.imshow(a[i], cmap='viridis', vmin=0, vmax=1)

        #     plt.xticks(np.arange(a.shape[-1]))
        #     plt.yticks(np.arange(a.shape[-1]))
        plt.grid('off')
    plt.show()


def plot_bar(v, c, title=''):
    plt.title(title)
    plt.bar(v, c)
    plt.ylim(0, 1)
    plt.yticks(np.linspace(0, 1, 11))
    plt.xticks(np.arange(0, v[-1], 2))


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

    print('Percentage of cells that are living: ', 100 * len(dead_cells) / (a.shape[-1] ** 2))

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

    print('Percentage of cells that are living: ', 100 * len(living_cells) / (a.shape[-1] ** 2))

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


def save_checkpoint(state, is_best, filename='models/checkpoint.pth.tar'):
    """
    state = {'model_arch': [{'n_channels': 6,
                    'n_layers': 12,  
                    'kernel_size': 3,
                    'lc':3,
                    'lp':3,
                    'lq':3,
                    'c':1,
                    'p':24,
                    'c':168}],
                'epoch':0,
                'batch_size':32,
                'best_loss':np.inf,
                'learn_rate':0.001,
                'model_state_dict':[],
                'optimizer_state_dict':[],
                'time':int(time())}
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'models/model_best.pth.tar')


def load_checkpoint(load_best=False):
    if load_best:
        filename = 'models/model_best.pth.tar'
    else:
        filename = 'models/checkpoint.pth.tar'

    return torch.load(filename)


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
            print('e:', int(x), int(y), dx, dy)

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


def accuracy_top_k(p, q, k):
    """
    Inputs
    ======
    p:    Predicted grid
    q:    Exact grid
    k:    Top k values
    """
    p_vals, p_args = torch.topk(p, k)
    q_vals, q_args = torch.topk(q, k)

    r = len(set(p_args) & set(q_args)) / len(set(q_args))

    return r


# top k accuracies for series of k and whole batch/set of data
def acc(pred, targ, k=1):
    """
    targ: target maps (N,d,d)
    pred: predicted maps (N,d,d)
    k: top k spots to be similar
    """
    accuracy_top_k
    accuracies = []
    for i in range(len(targ)):
        p = pred[i].view(-1)
        q = targ[i].view(-1)
        accuracies.append(accuracy_top_k(p, q, k))

    return accuracies


def get_n_params(model):
    """
    Returns the number of parameters the PyTorch model has
    """
    total = 0
    for i in list(model.parameters()):
        p = np.product(i.shape)
        total += p

    return total


class MySubplots2():
    def __init__(self, data, title_text="Plots"):  # init with targets
        """
        supply target data
        """
        self.fig = plt.figure(figsize=(16, 9))
        self.target_plots = []
        self.diff_plots = []
        self.pred_plots = []
        self.data_len = len(data)
        for i in range(self.data_len):
            plt.subplot(3, self.data_len, i + 1)
            self.target_plots.append(plt.imshow(data[i]))
            plt.axis('off')

            plt.subplot(3, self.data_len, self.data_len + i + 1)
            self.pred_plots.append(plt.imshow(data[i]))
            plt.axis('off')

            plt.subplot(3, self.data_len, 2 * self.data_len + i + 1)
            self.diff_plots.append(plt.imshow(data[i]))
            plt.axis('off')
            self.diff_plots[-1].set_cmap('Reds')

        self.title = plt.suptitle(title_text, fontsize=20)
        # plt.tight_layout()

    def update_subplots(self, data, title_text=None):  # update shows the difference and the
        """
        supply pred_data
        """
        for i in range(self.data_len):
            self.pred_plots[i].set_data(data[i])
            self.diff_plots[i].set_data(data[i] - self.target_plots[i].get_array().data)

        if title_text:
            self.title.set_text(title_text)


class MySubplots():
    def __init__(self, data, title_text="Plots"):
        """
        setup subplots 24: 3 by 8 for now
        """
        self.fig, self.axs = plt.subplots(nrows=3, ncols=8, figsize=(9.3, 6),
                                          subplot_kw={'xticks': [], 'yticks': []})

        self.fig.subplots_adjust(left=0.03, right=0.97, hspace=0.1, wspace=0.075)

        self.plots = []
        for i, (ax, grid) in enumerate(zip(self.axs.flat, data)):
            self.plots.append(ax.imshow(grid, vmin=0, vmax=2.))
            ax.set_title(str(i))

    def update_subplots(self, data, title_text=None):
        for i, plot in enumerate(self.plots):
            plot.set_data(data[i])

        if title_text:
            self.title.set_text(title_text)


def setup_subplots(a):
    n, m = getNearFactors(len(a))

    k = 0
    fig = plt.figure(figsize=(m, n))
    plots = []
    for i in range(n):
        for j in range(m):
            plt.subplot(n, m, k + 1, xticks=[], yticks=[])
            #             plt.title(str(k))
            plots.append(plt.imshow(a[k]))
            k += 1

    plt.tight_layout()
    return fig, plots


def update_subplots(a, plots):
    for i, plot in enumerate(plots):
        plot.set_data(a[i])


def im(a):
    plt.imshow(a)
    plt.colorbar()
    plt.show()


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


def pad(a, edge_size=1, pad_value=0):
    return np.pad(a, edge_size, pad_with, padder=pad_value)


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
    r = np.zeros_like(s)

    if len(s) % T == 0:
        ns = len(s) // T
    else:
        ns = len(s) // T + 1

    for n in range(ns):
        r[n * T] = s[n * T]
        r[n * T + 1:(n + 1) * T] = np.diff(s[n * T:(n + 1) * T], axis=0)  # because torch doesnt have np.diff
    return Variable(torch.FloatTensor(r))


def getNearFactors(C):
    """
    used in plot_convs to get ratio for the plot
    """
    c1 = int(C / np.sqrt(C) // 1)
    while C % c1 != 0:
        c1 -= 1
    c1 = int(c1)
    c2 = int(C / c1)
    return (c1, c2)


def plot_convs(a):
    """
    plots all the filter outputs of convs
    """
    if len(a.shape) < 4:
        plt.figure()
        plt.imshow(a[0].data)
        plt.show()
    else:
        N, C, H, W = a.shape
        c1, c2 = getNearFactors(C)
        plt.figure()
        for i in range(1, C + 1):
            plt.subplot(c1, c2, i)
            plt.axis('off')
            plt.imshow(a[0][i - 1].data, )
        plt.show()


def make_grid(A, t_size, x_size, y_size):
    """
    Note: np.histogramdd() can also be used - it's a bit 
    A: matrix with time, x and y coordinates
    returns grid matrix with each index filled where crimes occurred
    X and Y axis are swapped to make displaying easier
    """
    grids = np.zeros((t_size, y_size, x_size))
    for a in A:
        grids[a[0], y_size - 1 - a[2], a[1]] += 1

    return grids


def upsample_interpolate(a, scale=2, interpolate_kind='linear'):
    """
    mighht rather use cv2.pyrUp and pyerDown - seem to give less issues
    upsamlpe 2d grid array a
    scale: can be one value or tuple (nscale, dscale), default 2
    kind: 'linear', 'cubic', 'quintic'
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
    given array with true false occurences given number of indices between each true occurence
    """
    t = []
    prev_idx = 0
    cur_idx = 0
    for cur_idx in range(len(a)):
        if a[cur_idx]:
            t.append(cur_idx - prev_idx)
            prev_idx = cur_idx
    return t


def get_times(a):
    """
    given array with true-false occurences return indices of all true occurences
    """
    t = []
    for i in range(len(a)):
        if a[i]:
            t.append(i)
    return t


def plot(x):
    """
    simple quick plot
    """
    plt.figure()
    plt.plot(x)
    plt.show()


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
    return r.astype('int')


def get_recall(y_true, y_pred):
    """
    args: y_true, y_pred
    """
    all_real_pos = np.sum(y_true)
    correct_pos = np.sum(y_true & y_pred)
    return correct_pos / all_real_pos


def get_precision(y_true, y_pred):
    """
    args: y_true, y_pred
    """
    all_class_pos = np.sum(y_pred)
    correct_pos = np.sum(y_true & y_pred)
    return correct_pos / all_class_pos


def f1_score(y_true, y_pred):
    """
    args: y_true, y_pred
    returns number of true positive predictions
    """
    p = get_precision(y_true, y_pred)
    r = get_recall(y_true, y_pred)
    return 2 * p * r / (p + r)


def get_true_positive(y_true, y_pred):
    """
    returns number of true positive predictions
    """
    return np.sum(y_true & y_pred)


def get_false_positive(y_true, y_pred):
    """
    returns number of false positive predictions
    """
    xor = y_true != y_pred
    return np.sum(xor & y_pred)


def get_true_negative(y_true, y_pred):
    """
    args: y_true, y_pred
    returns number of true negative predictions
    """
    return np.sum((1 != y_true) & (1 != y_pred))


def get_false_negative(y_true, y_pred):
    """args: y_true, y_pred
    returns number of false negative predictions
    """
    xor = y_true != y_pred
    return np.sum(xor & (1 != y_pred))


def get_accuracy(y_true, y_pred):
    """
    args: y_true, y_pred
    returns accuracy between 0. and 1.

    """
    return np.sum(y_true == y_pred) / len(y_true)


def confusion(y_true, y_pred):
    """
    return confusion with true values on y axis and predicted values on x axis
        
           pred
          _ _ _ _
     t   |_|_|_|_|
     r   |_|_|_|_|
     u   |_|_|_|_|
     e   |_|_|_|_|
    
    
    """
    size = len(set(list(y_true) + list(y_true)))
    r = np.zeros((size, size))

    for i in range(len(y_true)):
        j = int(y_true[i])
        k = int(y_pred[i])
        r[j, k] += 1

    return r


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


def plot_dist(a):
    val, cnt = np.unique(a, return_counts=True)
    dst = cnt / np.sum(cnt) * 100
    plt.figure()
    plt.title("Distribution")
    if type(val[0]) == str:
        plt.xlabel("Percentage")
        plt.grid()
        plt.ylabel("Class")
        plt.barh(val, dst)
    else:
        plt.xlabel("Count")
        plt.grid()
        plt.ylabel("Percentage")
        plt.bar(val, dst)
    plt.show()


def plot3d(a):
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111, projection='3d')

    w, h = np.shape(a)
    xpos = []
    ypos = []

    for i in range(w):
        for j in range(h):
            xpos.append(i)
            ypos.append(j)

    zpos = w * h * [0]
    dx = np.ones(w * h)
    dy = np.ones(w * h)
    dz = a.flatten()
    # colors = np.arange(w * h)
    colors = []
    for i in dz:
        if i / dz.max() > 0.5:
            c = (1., 1 - 2. * (i / dz.max() - 0.5), 0., 1.)
        else:
            c = (2. * i / dz.max(), 1., 0., 1.)
        colors.append(c)

    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)
    ax1.set_xlabel("x pos")
    ax1.set_ylabel("y pos")
    ax1.set_zlabel("prob")
    plt.show()