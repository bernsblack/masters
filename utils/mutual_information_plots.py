import plotly.graph_objects as go
from sparse_discrete_table import conditional_mutual_info_over_time, mutual_info_over_time, \
    construct_temporal_information
from utils import cmi_name


def plot_mi_curves(
        a,
        t_range,
        max_offset=35,
        norm=True,
        log_norm=False,
        month_divisions=10,
        year_divisions=10,
        temporal_variables=("Day of Week", "Time of Month", "Time of Year"),
):
    conds = construct_temporal_information(
        date_range=t_range,
        temporal_variables=temporal_variables,
        month_divisions=month_divisions,
        year_divisions=year_divisions,
    ).values

    mi_score, offset = mutual_info_over_time(a=a, max_offset=max_offset,
                                             log_norm=log_norm, norm=norm)

    cmi_score, offset = conditional_mutual_info_over_time(a=a, max_offset=max_offset, conds=conds,
                                                          log_norm=log_norm, norm=norm)

    cmi_plot_label = cmi_name(temporal_variables=temporal_variables)

    return go.Figure(
        # data=[
        #     go.Scatter(y=mi_score, x=offset, name="MI"),
        #     go.Scatter(y=cmi_score, x=offset, name="CMI (DoW)"),
        # ],
        data=[
            go.Scatter(y=mi_score, x=offset, name='$I(C_{t},C_{t-k})$'),
            go.Scatter(y=cmi_score, x=offset, name=f'$I(C_{{t}},C_{{t-k}}|{cmi_plot_label})$'),
        ],
        layout=dict(
            title_text="Mutual and Conditional Mutual Information",
            title_x=0.5,
            xaxis_title="Offset in days (k)",
            yaxis_title="Normalised Score [0,1]",
            legend_title="Curves",
            font=dict(family="STIXGeneral"),
        ),
    )
