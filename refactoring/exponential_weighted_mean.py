import numpy as np
import pandas as pd

from utils.forecasting import mean_absolute_error


class EWM:
    """
    Exponential Weighted Mean Model
    Parameter alpha (decay) is estimated in fit method.
    """

    def __init__(self, series=None):
        self.alpha: float = .9
        self.options: np.ndarray = np.arange(0.001, 0.999, 0.001)

        if series is not None:
            self.fit(series)

    def __call__(self, series: np.ndarray):
        return self.predict(series)

    def fit(self, series):
        """
        fit runs through entire training set and does a grid search over parameter options to find the minimum loss
        :param series: pandas series or numpy ndarray of format (L,C) with L as series length, and C number of features
        :return:
        """
        losses = []
        for alpha in self.options:
            self.alpha = alpha
            estimate = self.predict(series)
            loss = mean_absolute_error(series, estimate)
            losses.append(loss)

        self.alpha = self.options[np.argmin(losses)]

    def predict(self, series: np.ndarray):
        series = np.pad(series, (1, 0), 'edge')

        estimate = pd.Series(series).ewm(alpha=self.alpha).mean().values[:-1]
        return estimate
