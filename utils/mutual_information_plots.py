import plotly.graph_objects as go
from plotly.subplots import make_subplots

from constants.date_time import TemporalVariables as tv
from sparse_discrete_table import (conditional_mutual_info_over_time, mutual_info_over_time,
                                   construct_temporal_information)
from utils import cmi_name


def plot_mi_curves(
        a,
        t_range,
        max_offset=50,
        norm=True,
        log_norm=False,
        bins=0,
        month_divisions=10,
        year_divisions=12,
        temporal_variables=(tv.DayOfWeek, tv.TimeOfMonth, tv.TimeOfYear),
        title="Mutual and Conditional Mutual Information",
        alpha=0.5,
):
    """

    :param alpha: the opacity of the two curves
    :param title: plot title
    :param a: array of values we want to calculate mi and cmi on
    :param t_range: the datetime range for 'a'
    :param max_offset: maximum lag offset used calculating mi and cmi
    :param norm: if the mi and cmi should be normed between 0 and 1
    :param log_norm: if the array 'a' should be scaled using log2(1 + x)
    :param bins: if the array 'a'
    :param month_divisions: how many blocks a month should be divided into when calculating cmi
    :param year_divisions: how many blocks a year should be divided into when calculating cmi
    :param temporal_variables: which time values should be used to construct the conditional variables, e.g. hour,
    day of week, time of month and time of year
    :return: a plotly figure with the mi and cmi curves, with y axis the normalized score and x axis the lag
    """
    conds = construct_temporal_information(
        date_range=t_range,
        temporal_variables=temporal_variables,
        month_divisions=month_divisions,
        year_divisions=year_divisions,
    ).values

    mi_score, offset = mutual_info_over_time(a=a,
                                             max_offset=max_offset,
                                             log_norm=log_norm,
                                             norm=norm,
                                             bins=bins)

    cmi_score, offset = conditional_mutual_info_over_time(a=a,
                                                          max_offset=max_offset,
                                                          conds=conds,
                                                          log_norm=log_norm,
                                                          norm=norm,
                                                          bins=bins)

    cmi_plot_label = cmi_name(temporal_variables=temporal_variables)

    ylabel = "Normalised Score [0,1]" if norm else "Score"

    return go.Figure(
        data=[
            go.Scatter(y=mi_score, x=offset, name='$I(C_{t},C_{t-k})$', opacity=alpha),
            go.Scatter(y=cmi_score, x=offset, name=f'$I(C_{{t}},C_{{t-k}}|{cmi_plot_label})$', opacity=alpha),
        ],
        layout=dict(
            title_text=title,
            title_x=0.5,
            xaxis_title="Time Step Offset (k)",
            yaxis_title=ylabel,
            legend_title="Curves",
            font=dict(family="STIXGeneral"),
        ),
    )


def subplot_mi_curves(
        a,
        t_range,
        max_offset=50,
        norm=True,
        log_norm=False,
        bins=0,
        month_divisions=10,
        year_divisions=12,
        temporal_variables=(tv.DayOfWeek, tv.TimeOfMonth, tv.TimeOfYear),
        title="Mutual and Conditional Mutual Information",
        a_title="Counts",
        alpha=0.5,
        shared_xaxes=False,
):
    """

    :param a_title:
    :param shared_xaxes:
    :param alpha: the opacity of the two curves
    :param title: plot title
    :param a: array of values we want to calculate mi and cmi on
    :param t_range: the datetime range for 'a'
    :param max_offset: maximum lag offset used calculating mi and cmi
    :param norm: if the mi and cmi should be normed between 0 and 1
    :param log_norm: if the array 'a' should be scaled using log2(1 + x)
    :param bins: if the array 'a'
    :param month_divisions: how many blocks a month should be divided into when calculating cmi
    :param year_divisions: how many blocks a year should be divided into when calculating cmi
    :param temporal_variables: which time values should be used to construct the conditional variables, e.g. hour,
    day of week, time of month and time of year
    :return: a plotly figure with the mi and cmi curves, with y axis the normalized score and x axis the lag
    """

    cmi_plot_label = cmi_name(temporal_variables=temporal_variables)

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[
            a_title,
            'Mutual Information',
            f'Conditional Mutual Information {temporal_variables}'],
        shared_xaxes=shared_xaxes,
    )

    conds = construct_temporal_information(
        date_range=t_range,
        temporal_variables=temporal_variables,
        month_divisions=month_divisions,
        year_divisions=year_divisions,
    ).values

    mi_score, offset = mutual_info_over_time(a=a,
                                             max_offset=max_offset,
                                             log_norm=log_norm,
                                             norm=norm,
                                             bins=bins)

    cmi_score, offset = conditional_mutual_info_over_time(a=a,
                                                          max_offset=max_offset,
                                                          conds=conds,
                                                          log_norm=log_norm,
                                                          norm=norm,
                                                          bins=bins)

    ylabel = "Normalised Score [0,1]" if norm else "Score"

    fig.add_trace(
        go.Scatter(y=a, x=t_range, name='$C_{t}$', opacity=alpha),
        row=1,
        col=1,
    )
    fig['layout'][f'xaxis']['title'] = "Date (t)"
    fig['layout'][f'yaxis']['title'] = "Counts"

    fig.add_trace(
        go.Scatter(y=mi_score, x=offset, name='$I(C_{t},C_{t-k})$', opacity=alpha),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(y=cmi_score, x=offset, name=f'$I(C_{{t}},C_{{t-k}}|{cmi_plot_label})$', opacity=alpha),
        row=3,
        col=1,
    )

    for i in range(1, 3):
        fig['layout'][f'xaxis{i + 1}']['title'] = "Time Step Offset (k)"
        fig['layout'][f'yaxis{i + 1}']['title'] = ylabel

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        #         xaxis_title="Time Step Offset (k)",
        #         yaxis_title=ylabel,
        legend_title="Curves",
        font=dict(family="STIXGeneral"),
        height=800,
        width=900,
    )

    return fig


def plot_mi(
        max_offset=50,
        norm=True,
        log_norm=False,
        bins=0,
        title="Auto Mutual Information",
        alpha=0.5,
        **kwargs,
):
    """
    :param alpha: the opacity of the two curves
    :param title: plot title
    :param max_offset: maximum lag offset used calculating mi and cmi
    :param norm: if the mi and cmi should be normed between 0 and 1
    :param log_norm: if the array 'a' should be scaled using log2(1 + x)
    :param bins: if the array 'a'
    day of week, time of month and time of year
    :return: a plotly figure with the mi and cmi curves, with y axis the normalized score and x axis the lag
    """
    ylabel = "Normalised Score [0,1]" if norm else "Score"

    plot_list = []
    for name, data in kwargs.items():
        mi_score, offset = mutual_info_over_time(a=data,
                                                 max_offset=max_offset,
                                                 log_norm=log_norm,
                                                 norm=norm,
                                                 bins=bins)

        plot_list.append(
            go.Scatter(y=mi_score, x=offset, name=name, opacity=alpha),
        )

    return go.Figure(
        data=plot_list,
        layout=dict(
            title_text=title,
            title_x=0.5,
            xaxis_title="Time Step Offset (k)",
            yaxis_title=ylabel,
            # legend_title="Curves",
            font=dict(family="STIXGeneral"),
        ),
    )
