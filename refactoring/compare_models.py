"""
Models compared are:
- GRUFNN
- EWM
- Historic Average
"""

# TODO: REFACTOR function: compare_time_series_metrics
# TODO: save trained baseline models and trained metrics to own folder...
# TODO: load baseline models...save evaluation metrics to own folder...
# TODO: all models have the same length of test set - the sequence needed should be incorporated into the loaders
# TODO: compare_time_series_metrics functions should actually just load metrics of each model and compare using latex and plots and save the result in a directory
# TODO: look at the grid metrics - we need meta information about the start adn stop times - can even have the results saved - look at grid result as an example
import logging
from typing import List

import pandas as pd
from numpy import ndarray
from pandas import DatetimeIndex

from refactoring.exponential_weighted_mean import EWM
from utils.configs import BaseConf
from utils.forecasting import forecast_metrics
from utils.plots import plot_time_signals
from utils.time_series import historic_average


def compare_time_series_metrics(
        conf: BaseConf,
        y_true: ndarray,
        y_score: ndarray,
        t_range: DatetimeIndex,
        feature_names: List[str],
        is_training_set: bool,
        step: int = 24,
        max_steps: int = 29,
        alpha: float = 0.5,
        range_slider_visible: bool = True,
):
    # raise Exception("REWRITE THIS CODE!! - ABSTRACT THE BASELINES INTO models.bases_line.sequence_models")
    kwargs = dict()

    offset = step * max_steps

    assert len(y_true.shape) == 2

    model_name_dict = {}
    feature_results = {}

    if is_training_set:
        training_str = "(Training Data)"
    else:
        training_str = "(Test Data)"

    for i, feat_name in enumerate(feature_names):
        kwargs[f"{feat_name}_y_score"] = y_score[offset:, i]
        kwargs[f"{feat_name}_y_true"] = y_true[offset:, i]
        kwargs[f"{feat_name}_y_ha"] = historic_average(
            data=y_true[:, i],
            step=step,
            max_steps=max_steps,
        )[offset - 1:-1]

        ewm = EWM(y_true[:, i])
        kwargs[f"{feat_name}_y_ewm"] = ewm(y_true[:, i])[offset:]

        feature_results[feat_name] = {
            "Ground Truth": y_true[offset:, i],
            "GRUFNN": y_score[offset:, i],
            f"HA({step},{max_steps})": historic_average(
                data=y_true[:, i],
                step=step,
                max_steps=max_steps
            )[offset - 1:-1],
            f"EWM({ewm.alpha:.3f})": ewm(y_true[:, i])[offset:],
        }

        model_name_dict[feat_name] = {
            "y_score": "GRUFNN",
            "y_true": "Ground Truth",
            "y_ha": f"HA({step},{max_steps})",
            "y_ewm": f"EWM({ewm.alpha:.3f})",
        }

        fig_predicted_normalised_city_counts = plot_time_signals(
            t_range=t_range[offset:],
            alpha=alpha,
            title=f'{conf.freq_title} {feat_name.title()} Predicted Normalised City Counts {training_str}',
            ylabel='Normalised Counts [0,1]',
            xlabel='Date',
            rangeslider_visible=range_slider_visible,
            **feature_results[feat_name]
        )
        file_name = f"{conf.plots_path}{conf.freq}_predictions_{feat_name}_" + \
                    f"{training_str.lower().replace(' ', '_').replace(')', '').replace('(', '')}.png"
        logging.info(f"Saving Predictions Plots in: {file_name}")
        fig_predicted_normalised_city_counts.write_image(file_name)
        fig_predicted_normalised_city_counts.show()

    ll = []  # {}
    for k, v in kwargs.items():
        i = k.find("_")
        feat_name = k[:i]
        model_type = k[i + 1:]
        if model_type == 'y_true':
            continue
        row = {}
        crime_type_name = feat_name.title()
        model_name = model_name_dict[feat_name][model_type]
        row['Crime Type'] = crime_type_name
        row['Model'] = model_name
        row_ = forecast_metrics(y_true=kwargs[f"{feat_name}_y_true"], y_score=kwargs[k])
        row = {**row, **row_}

        ll.append(row)

    #     plot_time_signals(t_range=t_range[offset:], alpha=alpha,
    #                       title=f'{conf.freq_title} Predicted Normalised City Counts',
    #                       yaxis_title='Normalised Counts [0,1]',**kwargs).show()

    metrics = pd.DataFrame(ll)
    metrics.sort_values(['Crime Type', 'MASE'], inplace=True)
    metrics.reset_index(inplace=True, drop=True)

    return metrics
