import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_absolute_error


class GaussianFilter1D:
    def __init__(self, sigma=1):
        self.sigma = sigma
        self.score_best = None

    def fit(self, a):
        sigma_best = None
        score_best = np.inf
        for i, sig in enumerate(np.linspace(1, 10, 50)):
            b = gaussian_filter1d(a, sigma=sig)
            score = mean_absolute_error(a[1:], b[:-1])
            if score < score_best:
                score_best = score
                sigma_best = sig

        self.sigma = sigma_best
        self.score_best = score_best
        print(f"fit: sigma => {sigma_best}")

    def transform(self, a):
        return gaussian_filter1d(a, sigma=self.sigma)

    def fit_transform(self, a):
        self.fit(a)
        return self.transform(a)
