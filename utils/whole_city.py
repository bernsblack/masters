import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from utils.configs import BaseConf
from constants.crime_types import TOTAL
from utils.mutual_information_plots import subplot_mi_curves
from utils.plots import plot_autocorr, plot_df
from utils.rolling import flag_anomalies, periodic_rolling_mean, rolling_norm
from utils.testing import assert_no_nan, assert_valid_datetime_index


def plot_time_series_anomalies_wc(
        conf: BaseConf,
        df: pd.DataFrame,
        periodic_offset: int = 1,
        anomaly_threshold: float = 3,
):
    """
    Save a plot for each column in time series dataframe
    
    :param conf: Config file containing paths to save plots and information on data time step frequency
    :param df: time series data frame with date time index and columns of time series values
    :param periodic_offset: periodic offset to use when calculating rolling functions, default is 1
    :param anomaly_threshold: normalised values outside threshold will be flagged as anomalies
    :return: 
    """
    window = {
        "1H": 501,
        "24H": 51,
        "168H": 13,
    }.get(conf.freq)

    logging.warning("Plot outliers are done with symmetric windowing and are " +
                    "only used to flag outliers not predict them")

    for col in df.columns:
        a = df[col].values

        anomalies = flag_anomalies(data=a, thresh=anomaly_threshold, window=window, period=periodic_offset, center=True,
                                   mode='reflect')

        ma = periodic_rolling_mean(data=a, window=window, period=periodic_offset, center=True)

        fig = go.Figure(
            data=[
                go.Scatter(x=df.index, y=a, opacity=.5, name=f'Counts'),
                go.Scatter(x=df.index[anomalies], y=a[anomalies], mode='markers', opacity=.5, name='Outliers'),
                go.Scatter(x=df.index, y=ma, opacity=.5, name=f'MA'),
            ],
            layout=dict(
                title_text=col,
                title_x=0.5,
                font=dict(family="STIXGeneral"),
                yaxis_title="Counts",
                xaxis_title="Date Time",
            ),
        )
        fig.write_image(f"{conf.plots_path}{conf.freq}_outliers_{col}.png".replace(' ', '_'))
        fig.show()


def plot_mi_curves_wc(conf: BaseConf, df: pd.DataFrame):
    temporal_variables = {
        # "1H": ["Hour", "Day of Week", "Time of Month", "Time of Year"],
        "1H": ["Hour", "Day of Week"],
        "24H": ["Day of Week", "Time of Month", "Time of Year"],
        "168H": ["Time of Month", "Time of Year"],
    }.get(conf.freq, ["Time of Month", "Time of Year"])

    max_offset = {
        "1H": 168 * 2,
        "24H": 365,
        "168H": 54,
    }.get(conf.freq)

    for i, name in enumerate(df.columns):
        a = df[name].values

        mutual_info_bins = 16  # 16
        #         print(f"optimal bins: {get_optimal_bins(a)}")
        #         a = to_percentile(a)
        #         a = np.round(np.log(1+a)) # watch out for values between 1024 2048
        #         a = cut(np.log(1+a)) # watch out for values between 1024 2048

        fig = subplot_mi_curves(
            a=a,
            t_range=df.index,
            max_offset=max_offset,
            norm=True,
            log_norm=False,
            bins=mutual_info_bins,
            month_divisions=4,
            year_divisions=12,
            temporal_variables=temporal_variables,
            title=f'{conf.freq_title} {name} Mutual and Conditional Mutual Information',
            a_title=f'{conf.freq_title} {name} City Wide Counts',
        )
        fig.write_image(f"{conf.plots_path}{conf.freq}_mi_plots_{name}.png".replace(' ', '_'))
        fig.show()


def normalize_periodically_wc(conf: BaseConf, df: pd.DataFrame, norm_offset=0, corr_subplots=False):
    """
    Normalised the time series of dataframe df.

    :param conf:
    :param df:
    :param norm_offset:
    :param corr_subplots:
    :return:
    """
    logging.warning("Using rolling norm means values at the " +
                    "start within the window will be set to NaN and dropped")

    max_offset = {
        "1H": 168 * 2 + 24,
        "24H": 365,
        "168H": 54,
    }.get(conf.freq)

    fig = plot_autocorr(**df,
                        title="Autocorrelation by Crime Type before any Rolling Normalisation",
                        partial=False,
                        max_offset=max_offset,
                        subplots=corr_subplots,
                        freq=conf.freq)
    fig.write_image(f"{conf.plots_path}{conf.freq}_auto_corr_normed_none.png".replace(' ', '_'))
    fig.show()

    window, period, period_string = {
        "24H": (53, 7, "Weekly"),  # jumps in weeks
        "1H": (366, 24, "Daily"),  # jumps in days
        "168H": (52, 1, "Weekly"),  # jumps in weeks
        #         "168H": (10,52, "Yearly"),  # jumps in years
    }.get(conf.freq, (10, 1))
    logging.info(f"Using rolling norm window: {window} and period: {period}")

    #     normed_df = rolling_norm(data=df,window=window,period=period).dropna()
    normed_df = rolling_norm(data=df, window=window, period=period, offset=norm_offset)
    # ensure only rows where all values are nan are trimmed - replace other nan values with 0
    valid_rows = ~normed_df.isna().all(1)
    normed_df[valid_rows] = normed_df[valid_rows].fillna(0)
    normed_df = normed_df.dropna()
    assert_no_nan(normed_df)
    assert_valid_datetime_index(normed_df)

    fig = plot_autocorr(**normed_df,
                        title=f"Autocorrelation by Crime Type after {period_string} Rolling Normalisation",
                        max_offset=max_offset, subplots=corr_subplots, freq=conf.freq)
    fig.write_image(f"{conf.plots_path}{conf.freq}_auto_corr_normed_{period_string.lower()}.png".replace(' ', '_'))
    fig.show()

    fig = plot_df(df[TOTAL], xlabel="Date", ylabel="Count",
                  title=f"Total Crimes before any Rolling Normalisation")
    fig.write_image(f"{conf.plots_path}{conf.freq}_total_crimes_normed_none.png".replace(' ', '_'))
    fig.show()

    fig = plot_df(normed_df[TOTAL], xlabel="Date", ylabel="Scaled Count",
                  title=f"Total Crimes after {period_string} Rolling Normalisation")
    fig.write_image(f"{conf.plots_path}{conf.freq}_total_crimes_normed_{period_string.lower()}.png".replace(' ', '_'))
    fig.show()

    double_rolling_norm = True
    if double_rolling_norm:
        window2, period2, period_string2 = {
            "24H": (5, 365, "Yearly"),  # jumps in years
            "1H": (53, 168, "Weekly"),  # jumps in weeks
            "168H": (10, 52, "Yearly"),  # jumps in years
        }.get(conf.freq, (None, None, None))
        logging.info(f"Using second rolling norm window: {window2} and period: {period2}")

        if period_string2:
            # double norming takes out other periodic signals as well
            #             normed_df = rolling_norm(data=normed_df,window=window2,period=period2).dropna()
            normed_df = rolling_norm(data=normed_df, window=window2, period=period2, offset=norm_offset)
            # ensure only rows where all values are nan are trimmed - replace other nan values with 0
            valid_rows = ~normed_df.isna().all(1)
            normed_df[valid_rows] = normed_df[valid_rows].fillna(0)
            normed_df = normed_df.dropna()
            assert_no_nan(normed_df)
            assert_valid_datetime_index(normed_df)

            fig = plot_autocorr(**normed_df,
                                title=f"Autocorrelation by Crime Type after {period_string} and" +
                                      f" {period_string2} Rolling Normalisation",
                                max_offset=max_offset, subplots=corr_subplots, freq=conf.freq)
            fig.write_image(f"{conf.plots_path}{conf.freq}_auto_corr_normed_{period_string.lower()}_" +
                            f"{period_string2.lower()}.png".replace(' ', '_'))
            fig.show()

            fig = plot_df(
                df=normed_df[TOTAL],
                xlabel="Date", ylabel="Scaled Count",
                title=f"Total Crimes after {period_string} and {period_string2} Rolling Normalisation",
            )
            #             fig = subplots_df(
            #                 df=normed_df[NOT_TOTAL],
            #                 xlabel="Date",ylabel="Scaled Count",
            #                 title=f"Total Crimes after {period_string} and {period_string2} Rolling Normalisation",
            #             )
            fig.write_image(
                f"{conf.plots_path}{conf.freq}_total_crimes_normed_{period_string.lower()}_" +
                f"{period_string2.lower()}.png".replace(' ', '_'))
            fig.show()
    assert len(np.unique(np.diff(normed_df.index.astype(int)))), \
        "Normed values are not contiguous, dropna method might have dropped values"

    return normed_df


def plot_time_vectors(conf: BaseConf, t_range: pd.Series, time_vectors: pd.DataFrame):
    k = int(conf.time_steps_per_day * 365)
    tv, tr = time_vectors[:k], t_range[:k]

    t_vec_names = {
        "1H": ['$H_{sin}$', '$H_{cos}$', '$DoW_{sin}$', '$DoW_{cos}$',
               '$ToM_{sin}$', '$ToM_{cos}$', '$ToY_{sin}$', '$ToY_{cos}$', '$Wkd$'],
        "24H": ['$DoW_{sin}$', '$DoW_{cos}$', '$ToM_{sin}$',
                '$ToM_{cos}$', '$ToY_{sin}$', '$ToY_{cos}$', '$Wkd$'],
        "168H": ['$ToM_{sin}$', '$ToM_{cos}$', '$ToY_{sin}$', '$ToY_{cos}$'],
    }.get(conf.freq)

    pio.templates.default = "plotly"
    fig = go.Figure(
        data=go.Heatmap(
            z=tv.T,
            y=t_vec_names,
            x=tr,
        ),
        layout=dict(
            title=f"Encoded Time Vectors on {conf.freq_title} Level",
            title_x=0.5,
            xaxis_title="Date",
            yaxis_title="Encoded Vector Values",
            font=dict(family="STIXGeneral"),
        ),
    )
    fig.write_image(f"{conf.plots_path}{conf.freq}_time_vector_encoding.png")
    fig.show()
    pio.templates.default = "plotly_white"
