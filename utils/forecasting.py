import numpy as np

# possible forecasting metrics
from models.baseline_models import historic_average
from utils import deprecated


def mean_forecast_error(y_true, y_score):
    """
    mean_forecast_error also known as forecast bias
    returns the mean error
    """
    return np.mean(y_true - y_score)


def mean_absolute_error(y_true, y_score):
    return np.mean(np.abs(y_true - y_score))


def mean_squared_error(y_true, y_score):
    return np.mean(np.square(y_true - y_score))


def root_mean_squared_error(y_true, y_score):
    return np.sqrt(np.mean(np.square(y_true - y_score)))


def mean_average_percentage_error(y_true, y_score):
    return np.mean(np.abs((y_true - y_score) / y_true))


def mean_absolute_scaled_error(y_true, y_score, offset=1):
    """
    MASE: Calculates the mean absolute error between y_true[n] and y_score[n] scaled by the mean absolute error between y_true[n] and y_true[n-offset]   
    
    MASE > 1, means y_true[n-offset] is a better estimate than y_score[n]
    MASE < 1 means y_score is better
    MASE == 1 means y_score and y_true[n-offset] are identical

    offset can be set to the season jump, 7 for week or 24 for hours
    
    :param y_true: ndarray of ground truth (time as first axis)
    :param y_score: ndarray of estimated values (time as first axis)
    :param offset: distance used to calculate the scale, i.e. abs(y_true[n-offset] - y_true[n]) 
    :return: 
    """
    y_true_prev = y_true[:-offset]
    y_true = y_true[offset:]
    y_score = y_score[offset:]

    model_err = np.mean(np.abs(y_true - y_score))
    naive_err = np.mean(np.abs(y_true - y_true_prev))

    return model_err / naive_err


def forecast_metrics(y_true, y_score, offset=1):
    """
    Calculates various forecasting metrics:
    - MASE: mean absolute scaled error
    - MAE: mean absolute error (not scaled and can be confusing when comparing experiments on various scales)
    - RMSE: root mean square error (values squared before calculating mean - penalising larger errors more than in MAE)
    - MSE: mean square error

    :param y_true: ndarray time series of ground truth (first axis as time)
    :param y_score: ndarray time series of estimated values (first axis as time)
    :param offset: offset used in MASE calculation
    :return: returns a dictionary with MASE, MAE, MSE and RMSE
    """
    return {
        'MASE': mean_absolute_scaled_error(y_true, y_score, offset),
        #         'MFE': mean_forecast_error(y_true, y_score),
        'MAE': mean_absolute_error(y_true, y_score),
        'MSE': mean_squared_error(y_true, y_score),
        'RMSE': root_mean_squared_error(y_true, y_score),
        #         'MAPE': mean_average_percentage_error(y_true, y_score),
    }


from pandas import DataFrame
from utils.plots import plot_time_signals


@deprecated
def compare_time_series_metrics(y_true, y_score, t_range, feature_names, step=24, max_steps=29, alpha=0.5):
    raise 'compare_time_series_metrics is deprecated us the one in refactoring folder'
    kwargs = dict()

    offset = step * max_steps

    assert len(y_true.shape) == 2

    ll = {}
    for i, feat_name in enumerate(feature_names):
        kwargs[f"{feat_name}_y_score"] = y_score[offset:, i]
        kwargs[f"{feat_name}_y_true"] = y_true[offset:, i]
        # kwargs[f"{feat_name}_y_true_lag1"] = ([y_true[0, i],*y_true[:-1, i]])[offset:]
        kwargs[f"{feat_name}_y_ha"] = historic_average(y_true[:, i], step=step, max_steps=max_steps)[offset - 1:-1]

    for k, v in kwargs.items():
        feat_name = k.split("_")[0]

        assert isinstance(kwargs[f"{feat_name}_y_true"], np.ndarray) == True, f"{feat_name}_y_true not ndarray"
        assert isinstance(kwargs[k], np.ndarray) == True, f"{feat_name} not ndarray"

        ll[k] = forecast_metrics(y_true=kwargs[f"{feat_name}_y_true"], y_score=kwargs[k])

    plot_time_signals(t_range=t_range[offset:], alpha=alpha, **kwargs).show()

    metrics = DataFrame(ll).transpose()
    # metrics.sort_values('MASE')

    return metrics


from statsmodels.tsa.stattools import acf, pacf
import pandas as pd


def vector_auto_corr(dense_grid, nlags=40, partial=True):
    if partial:
        func = lambda x: pacf(x, nlags=nlags)
    else:
        func = lambda x: acf(x, nlags=nlags, fft=False)

    return pd.DataFrame(dense_grid).apply(func).iloc[1:]
