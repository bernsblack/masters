import numpy as np

# possible forecasting metrics
from models.baseline_models import historic_average


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
    values above one means prev step performs better,
    lower than one means y_score is better
    equal one is same than prev step

    offset can be set to the season jump, 7 for week or 24 for hours
    """
    y_true_prev = y_true[:-offset]
    y_true = y_true[offset:]
    y_score = y_score[offset:]

    model_err = np.mean(np.abs(y_true - y_score))
    naive_err = np.mean(np.abs(y_true - y_true_prev))

    return model_err / naive_err


def forecast_metrics(y_true, y_score, offset=1):
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


def compare_time_series_metrics(y_true, y_score, t_range, feature_names, step=24, max_steps=29, alpha=0.5):
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
        ll[k] = forecast_metrics(y_true=kwargs[f"{feat_name}_y_true"], y_score=kwargs[k])

    plot_time_signals(t_range=t_range[offset:], alpha=alpha, **kwargs).show()

    metrics = DataFrame(ll).transpose()
    # metrics.sort_values('MASE')

    return metrics
