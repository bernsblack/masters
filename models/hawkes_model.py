from tick.plot import plot_hawkes_kernels
from tick.hawkes import SimuHawkesSumExpKernels, SimuHawkesMulti, \
    HawkesSumExpKern, HawkesEM
import numpy as np

class HawkesModelGeneral:
    """
    Using the HawkesEM learner from Tick library

    Using Tikc library from:
    - https://x-datainitiative.github.io/tick/modules/generated/tick.hawkes.HawkesSumExpKern.html
    """

    def __init__(self, kernel_size):
        self.name = "Hawkes"
        self.kernel_size = kernel_size
        self.kernel = None
        self.baseline = None
        self.em = None

    def fit(self, data):
        """
        data: (N,L)
        """
        N, L = data.shape

        # convert into format for the hawkes trainer
        realizations = []
        for i in range(L):
            data_counts = data[:, i]
            time_stamps = np.argwhere(data_counts).astype(np.float)
        realizations.append([time_stamps[:, 0]])

        # kernel_discretization if set explicitly it overrides kernel_support and kernel_size
        # todo: have kernel values be set by conf_dict
        self.em = HawkesEM(kernel_discretization=np.arange(self.kernel_size).astype(np.float),
                           n_threads=8,
                           verbose=False,
                           tol=1e-3,
                           max_iter=1000)
        self.em.fit(realizations)
        self.baseline = self.em.baseline.flatten()[0]
        self.kernel = self.em.kernel.flatten()

    def transform(self, data):
        # todo consider training saving a kernel AND baseline for each cell
        # kernels -> (N,L,kernel_size)
        # baselines -> (N,L,1)
        N, L = np.shape(data)
        result = np.empty(data.shape)
        for i in range(L):
            result[:, i] = self.baseline + np.convolve(data[:, i], self.kernel)[:N]
        return result

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def plot_kernel(self):
        plot_hawkes_kernels(self.em, hawkes=None, show=True)


class IndHawkesModel:
    """
    Indipendent Hawkes Modles where all cells are indipendent

    Using Tikc library from:
    - https://x-datainitiative.github.io/tick/modules/generated/tick.hawkes.HawkesSumExpKern.html
    """

    def __init__(self, kernel_size):
        self.name = "IndHawkes"
        self.kernel_size = kernel_size
        self.baselines = []
        self.kernels = []

    def fit(self, data):
        """
        data: (N,L)
        """
        N, L = data.shape
        self.baselines = []
        self.kernels = []

        # convert into format for the hawkes trainer
        for i in range(L):
            realizations = []
            data_counts = data[:, i]
            time_stamps = np.argwhere(data_counts).astype(np.float)
            realizations.append([time_stamps[:, 0]])

            # kernel_discretization if set explicitly it overrides kernel_support and kernel_size
            # todo: have kernel values be set by conf_dict
            em = HawkesEM(kernel_discretization=np.arange(self.kernel_size).astype(np.float),
                          n_threads=8,
                          verbose=False,
                          tol=1e-3,
                          max_iter=1000)
            em.fit(realizations)
            baseline = em.baseline.flatten()[0]
            kernel = em.kernel.flatten()
            self.baselines.append(baseline)
            self.kernels.append(kernel)

    def transform(self, data):
        # todo consider training saving a kernel AND baseline for each cell
        # kernels -> (N,L,kernel_size)
        # baselines -> (N,L,1)
        N, L = np.shape(data)
        result = np.empty(data.shape)
        for i in range(L):
            result[:, i] = self.baselines[i] + np.convolve(data[:, i], self.kernels[i])[:N]
        return result

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
