import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import io
import base64
from IPython.display import HTML

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib import rcParams

"""
    THIS MODULE IS ONLY FOR GENERIC PLOT FUNCTIONS - MORE SPECIFIC PLOT FUNCTION RELATED TO METRICS
    CAN BE FOUND IN THE utils.metrics MODULE 
"""
from matplotlib import rcParams


class BasePlotter:
    """
    Class is used to setup and add plots to a figure and then save or show this figure
    Acts az the base-class where we set global style features like the rcParams
    """

    def __init__(self, title, xlabel, ylabel):
        rcParams["mathtext.fontset"] = "stix"
        rcParams["font.family"] = "STIXGeneral"
        rcParams["font.size"] = "18"

        self.title = title
        self.figsize = (20, 10)
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.grid_alpha = 0.2
        self.setup()

    @staticmethod
    def setup():
        raise NotImplemented

    def finalise(self):
        plt.title(self.title)
        plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
        plt.grid(alpha=self.grid_alpha)

    def show(self):
        self.finalise()
        plt.show()

    def savefig(self, file_location):
        self.finalise()
        plt.savefig(file_location)


class DistributionPlotter(BasePlotter):
    """
    Class is used to plot violin distributions of data

    Example:
        ```
        dist_plot = DistributionPlotter(title="Random Data Test", xlabel="Models",ylabel="Accuracy", is_box_plot=False)
        dist_plot.add_data(data=data, labels=labels)  # where data is a list [(N, 1),(N, 1), ...]
        dist_plot.show()
        ```
    """

    def __init__(self, title, xlabel="Features", ylabel="Values", is_box_plot=False):  # setup maybe add the size of the figure
        super(DistributionPlotter, self).__init__(title, xlabel, ylabel)
        self.data = []
        self.labels = []
        self.is_box_plot = is_box_plot

    @staticmethod
    def setup():
        return None

    def add_data(self, data, labels):
        self.data.extend(data)
        self.labels.extend(labels)

    def finalise(self):
        plt.figure(figsize=self.figsize)
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)

        if self.is_box_plot:
            plt.boxplot(x=self.data,labels=self.labels, flierprops={"marker": "_"})
        else:
            plt.violinplot(dataset=self.data, showmeans=True, showextrema=True, showmedians=False)
            plt.xticks(ticks=np.arange(1, len(self.labels) + 1), labels=self.labels)

        plt.grid(alpha=self.grid_alpha)


def visualize_weights(model):
    plt.figure(figsize=(15, 5))
    state_dict = model.state_dict()
    keys = []
    for i, k in enumerate(state_dict):
        values = state_dict[k].squeeze().data.numpy()
        plt.scatter(i * np.ones_like(values), values, alpha=0.5)
        keys.append(k)

    plt.xticks(np.arange(len(keys)), keys, rotation=45, fontsize=30)
    plt.show()


def plot(x):
    """
    simple quick plot
    """
    plt.figure()
    plt.plot(x)
    plt.show()


def imshow(a, ax, title=""):
    """
    a: 2D array (Image)
    ax: subplot axis
    """
    im = ax.imshow(a, cmap="viridis")
    ax.set_title(title)
    #  ax.set_xticks(np.arange(a.shape[-2]))
    #  ax.set_yticks(np.arange(a.shape[-1]))
    plt.grid("off")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, cmap="viridis")


def plot_compare(a, b, times, relative=True, a_title="A", b_title="B"):
    """
    plot to grids next to each other given a range of matching times
    """
    for i in range(len(a)):
        plt.figure(figsize=(18, 6))
        plt.suptitle(times[i])
        plt.subplot(1, 3, 1)
        plt.title(a_title)
        if relative:
            plt.imshow(a[i], cmap="viridis")
        else:
            plt.imshow(a[i], cmap="viridis", vmin=0, vmax=1)
        plt.xticks(np.arange(a.shape[-1]))
        plt.yticks(np.arange(a.shape[-1]))

        plt.subplot(1, 3, 2)
        plt.title(b_title)
        if relative:
            plt.imshow(b[i], cmap="viridis")
        else:
            plt.imshow(b[i], cmap="viridis", vmin=0, vmax=1)
        plt.xticks(np.arange(b.shape[-1]))
        plt.yticks(np.arange(b.shape[-1]))

        plt.subplot(1, 3, 3)
        plt.title("Error")
        if relative:
            plt.imshow(a[i] - b[i], cmap="viridis")
        else:
            plt.imshow(a[i] - b[i], cmap="viridis", vmin=0, vmax=1)
        plt.xticks(np.arange(b.shape[-1]))
        plt.yticks(np.arange(b.shape[-1]))

        plt.show()
        print(
            "============================================================================================================================")


def play_video(f):
    video = io.open(f, "r+b").read()
    encoded = base64.b64encode(video)
    return HTML(data="""<video alt="test" controls>
                  <source src="data:video/mp4;base64,{0}" type="video/mp4" />
               </video>""".format(encoded.decode("ascii")))


def get_times(a):
    """
    given array with true-false occurrences return indices of all true occurrences
    """
    t = []
    for i in range(len(a)):
        if a[i]:
            t.append(i)
    return t


def plot_targ_pred_over_time(trg, prd):
    """
    fig setup  should be done before calling function
    function scatters the values of the target and predicted values
    and plots the intensity curve
    """
    indices = get_times(trg)
    plt.scatter(indices, prd[indices], s=20)
    plt.scatter(indices, trg[indices], s=20, c="r", marker="x")
    plt.plot(prd, alpha=0.7)


def plot_day(a, times, relative=True):
    """
    display array a (24,rows,cols)
    """
    plt.figure(figsize=(25, 15))
    for i in range(24):
        plt.subplot(4, 6, i + 1)
        plt.title(times[i])
        if relative:
            plt.imshow(a[i], cmap="viridis")
        else:
            plt.imshow(a[i], cmap="viridis", vmin=0, vmax=1)

        #     plt.xticks(np.arange(a.shape[-1]))
        #     plt.yticks(np.arange(a.shape[-1]))
        plt.grid("off")
    plt.show()


def plot_bar(v, c, title=""):
    plt.title(title)
    plt.bar(v, c)
    plt.ylim(0, 1)
    plt.yticks(np.linspace(0, 1, 11))
    plt.xticks(np.arange(0, v[-1], 2))


class MySubplots2:
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
            plt.axis("off")

            plt.subplot(3, self.data_len, self.data_len + i + 1)
            self.pred_plots.append(plt.imshow(data[i]))
            plt.axis("off")

            plt.subplot(3, self.data_len, 2 * self.data_len + i + 1)
            self.diff_plots.append(plt.imshow(data[i]))
            plt.axis("off")
            self.diff_plots[-1].set_cmap("Reds")

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


class MySubplots:
    def __init__(self, data, title_text="Plots"):
        """
        setup subplots 24: 3 by 8 for now
        """
        self.fig, self.axs = plt.subplots(nrows=3, ncols=8, figsize=(9.3, 6),
                                          subplot_kw={"xticks": [], "yticks": []})

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
    plt.figure(figsize=(10, 8))
    plt.imshow(a)
    plt.colorbar()
    plt.show()


def getNearFactors(C):
    """
    used in plot_convs to get ratio for the plot
    """
    c1 = int(C / np.sqrt(C) // 1)
    while C % c1 != 0:
        c1 -= 1
    c1 = int(c1)
    c2 = int(C / c1)
    return c1, c2


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
            plt.axis("off")
            plt.imshow(a[0][i - 1].data, )
        plt.show()


def plot_dist(a):
    val, cnt = np.unique(a, return_counts=True)
    dst = cnt / np.sum(cnt) * 100
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


def plot_grid_sum_over_time(a):
    plt.plot(a.sum(1).sum(1))


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
    dz = a.squeeze()
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


def plot3D(a):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    h, w = np.shape(a)
    _x = [i for i in range(w)]
    _y = [i for i in range(h)]
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    _z = a.ravel()

    # There may be an easier way to do this, but I am not aware of it
    z = np.zeros(len(x))
    for i in range(1, len(x)):
        j = int((i * len(_z)) / len(x))
        z[i] = _z[j]

    bottom = np.zeros_like(z)
    width = depth = 1

    cmap = cm.get_cmap('viridis')  # Get desired colormap - you can change this!
    max_height = np.max(_z)  # get range of colorbars so we can normalize
    min_height = np.min(_z)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k - min_height) / max_height) for k in _z]

    ax.bar3d(x, y, bottom, width, depth, z, shade=True, color=rgba)

    plt.show()
