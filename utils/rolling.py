import numpy as np
import unittest

from utils.data_processing import safe_divide, safe_divide_df, safe_divide_df_zero_on_zero


def periodic_rolling(func, data, window, period=1, center=False, fill=np.nan, axis=0, mode=None, offset=0):
    """
    Applies a rolling function

    :param mode: ('reflect' or None)
    :param func: function to apply across the moving windows
    :param data: ndarray of time series data
    :param window: window used to calculate running mean and std
    :param period: periodic offset to use when calculating rolling functions, default is 1
    :param center: if the rolling functions should centered around current value or have value at the right
    :param fill: values to fill array with where window is too large
    :param axis: axis to calculate the rolling function on
    :param offset: if the mean that is calculated is offset to further time so that short term trends are not wiped out
    :return:
    """
    # todo add offset to not wipe out short term trends from the data - e.g.

    if mode is not None:
        pad_width = window // 2 if center else window
        # pad_width = period * (window // 2) if center else period * window
        # pad_width = period * window
        data = np.pad(data, pad_width=pad_width, mode=mode)

    r = np.empty(data.shape)
    r.fill(fill)

    if center:
        hw = window // 2
        lo = period * hw
        hi = period * hw if window % 2 == 0 else period * hw + 1
        end = len(r) - hi + 1
        jump = 0
    else:
        lo = period * (window - 1)
        hi = 1
        end = len(r)
        jump = offset * period  # in counts of periods - same as window

    for i in range(lo + jump, end):
        r[i] = func(data[i - lo - jump:i + hi - jump:period], axis=axis)

    if mode is not None:
        pad_width = window // 2 if center else window
        r = r[pad_width:-pad_width]

    return r


def periodic_rolling_mean(data, window, period=1, center=False, fill=np.nan, axis=0, offset=0):
    """
    Applies a rolling mean

    :param data: ndarray of time series data
    :param window: window used to calculate running mean and std
    :param period: periodic offset to use when calculating rolling functions, default is 1
    :param center: if the rolling functions should centered around current value or have value at the right
    :param fill: values to fill array with where window is too large
    :param axis: axis to calculate the rolling function on
    :param offset: how many periods back should the window stop
    :return:
    """
    return periodic_rolling(func=np.mean,
                            data=data,
                            window=window,
                            period=period,
                            center=center,
                            fill=fill,
                            axis=axis,
                            offset=offset)


def periodic_rolling_std(data, window, period=1, center=False, fill=np.nan, axis=0, offset=0):
    """
    Applies a rolling std

    :param data: ndarray of time series data
    :param window: window used to calculate running mean and std
    :param period: periodic offset to use when calculating rolling functions, default is 1
    :param center: if the rolling functions should centered around current value or have value at the right
    :param fill: values to fill array with where window is too large
    :param axis: axis to calculate the rolling function on
    :param offset: how many periods back should the window stop
    :return:
    """
    return periodic_rolling(func=np.std,
                            data=data,
                            window=window,
                            period=period,
                            center=center,
                            fill=fill,
                            axis=axis,
                            offset=offset)


def rolling_norm(
        data,
        window,
        period=1,
        center=False,
        fill=np.nan,
        axis=0,
        mode=None,
        offset=0,
):
    """
    Applies a rolling normalisation by subtracting the rolling mean and dividing by the standard deviation

    :param mode:
    :param data: ndarray of time series data
    :param window: window used to calculate running mean and std
    :param period: periodic offset to use when calculating rolling functions, default is 1
    :param center: if the rolling functions should centered around current value or have value at the right
    :param fill: values to fill array with where window is too large
    :param axis: axis to calculate the rolling function on
    :param offset: how many periods back should the window stop
    :return:
    """
    d_mean = periodic_rolling(
        func=np.mean,
        data=data,
        window=window,
        period=period,
        center=center,
        fill=fill,
        axis=axis,
        mode=mode,
        offset=offset,
    )

    d_std = periodic_rolling(
        func=np.std,
        data=data,
        window=window,
        period=period,
        center=center,
        fill=fill,
        axis=axis,
        mode=mode,
        offset=offset,
    )

    d_normed = (data - d_mean) / d_std

    # use safe_divide to avoid dividing by zero
    # d_normed = safe_divide_df(data - d_mean, d_std)
    # d_normed = safe_divide_df_zero_on_zero(data - d_mean, d_std)

    # d_normed = data - d_mean
    # non_zeros = d_std > 1e-10  # tolerance check to avoid dividing by zero
    # d_normed[non_zeros] = d_normed[non_zeros] / d_std[non_zeros]

    return d_normed


class RollingNormScaler:
    def __init__(
            self,
            window,
            period=1,
            center=False,
            fill=np.nan,
            axis=0,
            mode=None,
            offset=0,
    ):
        """
        Applies a rolling normalisation by subtracting the rolling mean and dividing by the standard deviation

        :param mode:
        :param window: window used to calculate running mean and std
        :param period: periodic offset to use when calculating rolling functions, default is 1
        :param center: if the rolling functions should centered around current value or have value at the right
        :param fill: values to fill array with where window is too large
        :param axis: axis to calculate the rolling function on
        :param offset: how many periods back should the window stop
        :return:
        """
        self.window = window
        self.period = period
        self.center = center
        self.fill = fill
        self.axis = axis
        self.mode = mode
        self.d_mean = None
        self.d_std = None

    def fit(self, data):
        self.d_mean = periodic_rolling(
            func=np.mean,
            data=data,
            window=self.window,
            period=self.period,
            center=self.center,
            fill=self.fill,
            axis=self.axis,
            mode=self.mode,
        )

        self.d_std = periodic_rolling(
            func=np.std,
            data=data,
            window=self.window,
            period=self.period,
            center=self.center,
            fill=self.fill,
            axis=self.axis,
            mode=self.mode,
        )

    def transform(self, data):
        d_normed = (data - self.d_mean) / self.d_std
        return d_normed

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        return data * self.d_std + self.d_mean


def flag_anomalies(
        data,
        thresh,
        window,
        period=1,
        center=False,
        fill=np.nan,
        axis=0,
        mode=None,
):
    """
    Returns arguments of anomalous values outside the threshold

    :param mode:
    :param data: time series data
    :param thresh: normalised values outside threshold will be flagged as anomalies
    :param window: window used to calculate running mean and std
    :param period: periodic offset to use when calculating rolling functions, default is 1
    :param center: if the rolling functions should centered around current value or have value at the right
    :param fill: values to fill array with where window is too large
    :param axis: axis to calculate the rolling function on
    :return: indices of anomalous values
    """
    data_normed = rolling_norm(
        data=data,
        window=window,
        period=period,
        center=center,
        fill=fill,
        axis=axis,
        mode=mode,
    )
    indices = np.argwhere(np.abs(data_normed) > thresh).flatten()

    return indices


class TestPeriodRollingFunction(unittest.TestCase):
    def test_rolling_nan(self):
        z = np.zeros(50)
        z[20:25] = 1
        zn = rolling_norm(data=z, window=5)
        z_std = periodic_rolling_std(data=z, window=5)
        zmu = periodic_rolling_mean(data=z, window=5)

        self.assertTrue((z[24] == 1 and np.isnan(zn[24]) and zmu[24] == 1 and z_std[24] == 0))

    def test_period_center_and_no_center(self):
        # assertion test for the periodic rolling function
        period = 4
        a = np.zeros(45)
        a[::period] = 1
        a[[0, 16, 20, 44]] = 0
        w = 3

        center_ma = periodic_rolling(
            func=np.mean,
            data=a,
            window=w,
            period=period,
            center=True,
            fill=np.nan,
            axis=0,
        )

        normal_ma = periodic_rolling(
            func=np.mean,
            data=a,
            window=w,
            period=period,
            center=False,
            fill=np.nan,
            axis=0,
        )

        true_normal_ma, true_center_ma = (
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan,
                      np.nan, np.nan, np.nan, 0.66666667, 0.,
                      0., 0., 1., 0., 0.,
                      0., 0.66666667, 0., 0., 0.,
                      0.33333333, 0., 0., 0., 0.33333333,
                      0., 0., 0., 0.66666667, 0.,
                      0., 0., 1., 0., 0.,
                      0., 1., 0., 0., 0.,
                      1., 0., 0., 0., 0.66666667]),
            np.array([np.nan, np.nan, np.nan, np.nan, 0.66666667,
                      0., 0., 0., 1., 0.,
                      0., 0., 0.66666667, 0., 0.,
                      0., 0.33333333, 0., 0., 0.,
                      0.33333333, 0., 0., 0., 0.66666667,
                      0., 0., 0., 1., 0.,
                      0., 0., 1., 0., 0.,
                      0., 1., 0., 0., 0.,
                      0.66666667, np.nan, np.nan, np.nan, np.nan]))

        np.testing.assert_almost_equal(actual=true_center_ma, desired=center_ma)  # assertion test
        np.testing.assert_almost_equal(actual=true_normal_ma, desired=normal_ma)  # assertion test
