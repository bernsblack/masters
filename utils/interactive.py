import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

from sparse_discrete_table import conditional_mutual_info_over_time, mutual_info_over_time, \
    construct_temporal_information
from utils.data_processing import encode_category
from models.model_result import get_models_metrics
from utils import ffloor, fceil
from pprint import pformat
from pandas import Timedelta
from geopy import distance
from ipywidgets import Layout, widgets
import plotly.graph_objects as go
from utils.metrics import safe_f1_score
from pandas.core.indexes.datetimes import DatetimeIndex


class State:
    """
    State object to make interactive plots easier.
    Callbacks can be added to be triggered when state fields are updated.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v
        self.callbacks = []  # inner repr of the update callback
        self.__recursion_guard = 0

    def __repr__(self):
        display = {**self.__dict__}
        display.pop("_State__recursion_guard", None)
        return pformat(display)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value
        self.__update()

    def on_update(self, callback, append=False):
        """
        set the callback to be called when a values is set using state['a'] = 1 format
        callback: function that takes State class as input
        """
        if append:
            self.callbacks.append(callback)
        else:
            self.callbacks = [callback]

    def update(self, **kwargs):
        """
        set/update multiple values before calling the on_update callback
        """
        for k, v in kwargs.items():
            self.__dict__[k] = v
        self.__update()

    def __update(self):
        self.__recursion_guard += 1
        if self.__recursion_guard < 2:
            for fn in self.callbacks:
                fn(self)
        self.__recursion_guard = 0


def state_to_conf(state):
    conf = dict(
        start_date=state.date_min.strftime("%Y-%m-%d"),
        end_date=(state.date_max + Timedelta(state.freq)).strftime("%Y-%m-%d"),
        dT=state.freq,
        x_scale=int(state.dlon / 0.001),
        y_scale=int(state.dlat / 0.001),
        lat_max=np.round(state.lat_max, decimals=3),
        lat_min=np.round(state.lat_min, decimals=3),
        lon_max=np.round(state.lon_max, decimals=3),
        lon_min=np.round(state.lon_min, decimals=3),
        root="./",
        crime_types=state.crime_types,
    )

    return conf


# rename and move to utils
def new_int_bins(int_min, int_max):
    """
    Will create a array of bin values so that the integers from int_min, int_max fall exactly between the bins.
    Example:
    new_int_bins(int_min=0, int_max=4) => array([0, 1, 2, 3, 4, 5])
    with lower bin values inclusive and upper limits exclusive, i.e. [lower, upper)

    :param int_min: minimum value to be binned (inclusive)
    :param int_max: maximum value to be binned (inclusive)
    :return: array of bin edges to ensure all integers between int_min and int_max inclusive are binned separately.
    """
    return np.arange(int_min, int_max + 2)


def filter_frame(data_frame, state):
    """
    filter df given the values in state to get a subset of data
    """
    # spatial filters
    lon_mask = (data_frame.Longitude >= state.lon_min) & (data_frame.Longitude <= state.lon_max)
    lat_mask = (data_frame.Latitude >= state.lat_min) & (data_frame.Latitude <= state.lat_max)

    # time filters
    date_mask = (data_frame.Date >= state.date_min) & (data_frame.Date <= state.date_max)

    # crime type filters
    crime_types_mask = data_frame['Primary Type'].isin(state.crime_types)

    filter_mask = date_mask & crime_types_mask & lon_mask & lat_mask

    return data_frame[filter_mask]


def get_total_counts(data_frame, state, date_range):
    """
    data_frame: dataframe of crime incidents
    returns total_counts_y, total_counts_x
    """
    if len(data_frame) > 0:
        nt = int(np.ceil(data_frame.t.max()))
        tbins = new_int_bins(data_frame.t.min(), data_frame.t.max())

        #  total counts line/curve
        total_counts, edges = np.histogram(data_frame.t, bins=tbins)
        total_counts_y = total_counts
        total_counts_x = date_range[edges[:-2]]
    else:
        total_counts_y = np.zeros(len(date_range))
        i, j = state.date_indices
        total_counts_x = date_range[i, j]
    return total_counts_y, total_counts_x


def new_bins(data_frame, state):
    xbins = np.arange(
        start=ffloor(state.lon_min, state.dlon),
        stop=fceil(state.lon_max, state.dlon),
        step=state.dlon,
    )
    nx = len(xbins)

    ybins = np.arange(
        start=ffloor(state.lat_min, state.dlat),
        stop=fceil(state.lat_max, state.dlat),
        step=state.dlat,
    )
    ny = len(ybins)

    nt = int(np.ceil(data_frame.t.max()))
    tbins = new_int_bins(data_frame.t.min(), data_frame.t.max())

    nc = len(state.crime_types)
    cbins = np.arange(0, nc + 1, 1)

    return tbins, cbins, ybins, xbins


def bin_data_frame(data_frame, state):
    """
    will bin the data from into a N,C,H,W grid depending on the state
    """
    bins = new_bins(data_frame, state)

    data_frame['c'] = encode_category(series=data_frame['Primary Type'], categories=state.crime_types)

    binned_data, bins = np.histogramdd(
        sample=data_frame[['t', 'c', 'Latitude', 'Longitude']].values,
        bins=bins,
    )

    return binned_data, bins


def get_mean_map(data_frame, state):
    """
    will be a heat map of the means with the min-max lat and lon min and remaining constant
    """
    binned_data, bins = bin_data_frame(data_frame, state)
    _, _, ybins, xbins = bins

    mean_map = binned_data.sum(1).mean(0)
    return mean_map, xbins, ybins


def get_ratio_xy(data_frame):
    """
    Given data_frame with fields Latitude and Longitude return the ratio between the min and max points distances
    returns d(lon_max-lon_min)/d(lat_max-lat_min)
    this ratio helps with plotting grids in a realistic manner
    """
    # get meter per degree estimate
    coord_series = data_frame[['Latitude', 'Longitude']]

    lat_min, lon_min = coord_series.min()
    lat_max, lon_max = coord_series.max()

    # lat_mean, lon_mean = coord_series.mean()

    dy = distance.distance((lat_min, lon_min), (lat_max, lon_min)).m
    dx = distance.distance((lat_min, lon_min), (lat_min, lon_max)).m

    # lat_per_metre = (lat_max - lat_min)/dy
    # lon_per_metre = (lon_max - lon_min)/dx

    ratio_xy = dx / dy
    return ratio_xy


def new_interactive_heatmap(z, name=None, height=500):
    h, w = z.shape
    # height = 600  # int(30*h)
    width = height * w / h  # 300

    return go.FigureWidget(
        go.Heatmap(z=z),
        layout=dict(
            height=height,
            width=width,
            title_x=0.5,
            title=name,
            margin=dict(l=20, r=20, t=50, b=20),
            #             paper_bgcolor="LightSteelBlue",
            #             colorscale='Viridis', # one of plotly colorscales
            #             showscale=True,
        ),
    )


class InteractiveHeatmaps:

    def __init__(self, date_range, col_wrap=3, height=500, **kwargs):
        """
        InteractiveHeatmaps creates in interactive widget to scroll through and investigate grids that vary over time

        :param date_range: pandas date range array
        :param col_wrap: plots per row before wrapping
        :param height: height in pixels of a single images
        :param kwargs: key word arguments for name of plot as key and data (N,H,W format) as value
        """

        def get_widget_value(change):
            self.change = change
            if isinstance(change, dict) and change.get('name') == 'value':
                self.valid_change = change
                return change.get('new')
            return None

        self.heatmaps = dict()
        self.figures = []
        self.grids = dict()

        for name, grid in kwargs.items():
            self.grids[name] = grid
            hm_fig = new_interactive_heatmap(z=grid[0], name=name, height=height)
            self.figures.append(hm_fig)
            self.heatmaps[name] = hm_fig.data[0]

        self.date_range = date_range

        # time index date display label
        self.current_date_label = widgets.Label(f'Date: {self.date_range[0].strftime("%c")}')

        # time index selector
        self.time_index_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.date_range) - 1,
            step=1,
            description='Time Index:',
            continuous_update=False,
            layout=Layout(width='80%'),
        )

        def on_change_time_index(change):
            time_index = get_widget_value(change)
            if time_index is not None:
                self.time_index = time_index
                self.current_date_label.value = f'Date: {self.date_range[self.time_index].strftime("%c")}'

                for name_, heatmap in self.heatmaps.items():
                    heatmap.z = self.grids[name_][self.time_index]

        self.time_index_slider.observe(on_change_time_index)

        self.play_button = widgets.Play(
            value=0,
            min=self.time_index_slider.min,
            max=self.time_index_slider.max,
            step=1,
            interval=1000,
            description="Press play",
            disabled=False
        )
        widgets.jslink((self.play_button, 'value'), (self.time_index_slider, 'value'))

        wrapped_figs = []
        row_figs = []
        for i, fig in enumerate(self.figures):
            if i % col_wrap == 0:
                wrapped_figs.append(widgets.HBox(row_figs))
                row_figs = []

            row_figs.append(fig)
        wrapped_figs.append(widgets.HBox(row_figs))

        self.app = widgets.VBox([
            self.current_date_label,
            widgets.HBox([
                self.play_button,
                self.time_index_slider,
            ]),
            widgets.VBox(wrapped_figs),
        ])


class InteractiveHeatmapsWithLines:

    def __init__(self, date_range, col_wrap=3, height=500, thresh=0, **kwargs):
        """
        InteractiveHeatmaps creates in interactive widget to scroll through and investigate grids that vary over time

        :param date_range: pandas date range array
        :param col_wrap: plots per row before wrapping
        :param height: height in pixels of a single images
        :param kwargs: key word arguments for name of plot as key and data (N,H,W format) as value
        """

        def get_widget_value(change):
            self.change = change
            if isinstance(change, dict) and change.get('name') == 'value':
                self.valid_change = change
                return change.get('new')
            return None

        self.heatmaps = dict()
        self.figures = list()
        self.grids = dict()
        self.lines = dict()

        lines_data = []
        for name, grid in kwargs.items():
            self.grids[name] = grid
            hm_fig = new_interactive_heatmap(z=grid[0], name=name, height=height)
            self.figures.append(hm_fig)
            self.heatmaps[name] = hm_fig.data[0]

            lines_data.append(go.Scatter(name=name, opacity=.5))

        if thresh > 0:
            thresh_line = go.Scatter(x=date_range, y=thresh * np.ones(len(grid)), name="Threshold", opacity=.3)
            lines_data = [thresh_line, *lines_data]

        fw_lines = go.FigureWidget(data=lines_data)  # links the pointer to the dict
        for line in fw_lines.data:
            self.lines[line.name] = line

        state = State()

        def draw():
            x = state['x']
            y = state['y']

            with fw_lines.batch_update():
                fw_lines.update_layout(title={"text": f"Selected cell: y,x = {y, x}"}, title_x=0.5)
                for name, grid in self.grids.items():
                    z = grid[:, y, x]

                    self.lines[name].y = z
                    self.lines[name].x = date_range

        def set_state(_trace, points, _selector):
            y = points.ys[0]
            x = points.xs[0]

            state["x"] = x
            state["y"] = y
            draw()

        for fw_grid in self.figures:
            fw_grid.data[0].on_click(set_state)

        self.date_range = date_range

        # time index date display label
        if isinstance(date_range, DatetimeIndex):
            label = f'Date: {self.date_range[0].strftime("%c")}'
        else:
            label = f'Offset: {self.date_range[0]}'

        self.current_date_label = widgets.Label(label)

        # time index selector
        self.time_index_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.date_range) - 1,
            step=1,
            description='Time Index:',
            continuous_update=False,
            layout=Layout(width='80%'),
        )

        def on_change_time_index(change):
            time_index = get_widget_value(change)
            if time_index is not None:
                self.time_index = time_index

                # time index date display label
                if isinstance(date_range, DatetimeIndex):
                    label_ = f'Date: {self.date_range[self.time_index].strftime("%c")}'
                else:
                    label_ = f'Offset: {self.date_range[self.time_index]}'

                self.current_date_label.value = label_

                for name_, heatmap in self.heatmaps.items():
                    heatmap.z = self.grids[name_][self.time_index]

        self.time_index_slider.observe(on_change_time_index)

        self.play_button = widgets.Play(
            value=0,
            min=self.time_index_slider.min,
            max=self.time_index_slider.max,
            step=1,
            interval=1000,
            description="Press play",
            disabled=False
        )
        widgets.jslink((self.play_button, 'value'), (self.time_index_slider, 'value'))

        wrapped_figs = []
        row_figs = []
        for i, fig in enumerate(self.figures):
            if i % col_wrap == 0:
                wrapped_figs.append(widgets.HBox(row_figs))
                row_figs = []

            row_figs.append(fig)

        # box_layout = Layout(display='flex',
        #                     flex_flow='column',
        #                     align_items='center',
        #                     width='100%')
        # wrapped_figs.append(widgets.Box(children=row_figs, layout=box_layout))

        wrapped_figs.append(widgets.HBox(row_figs))

        self.app = widgets.VBox([
            self.current_date_label,
            widgets.HBox([
                self.play_button,
                self.time_index_slider,
            ]),
            widgets.VBox(wrapped_figs),
            widgets.VBox([fw_lines]),
        ])


def plot_interactive_epoch_losses(trn_epoch_losses, val_epoch_losses):
    # trn_val_epoch_losses = np.array(trn_epoch_losses) + np.array(val_epoch_losses)

    return go.Figure(
        [
            # go.Scatter(y=trn_val_epoch_losses, name="Train Valid Losses", mode='lines+markers'),
            go.Scatter(y=trn_epoch_losses, name="Train Losses", mode='lines+markers'),
            go.Scatter(y=val_epoch_losses, name="Validation Losses", mode='lines+markers'),
            go.Scatter(y=[np.min(val_epoch_losses)],
                       x=[np.argmin(val_epoch_losses)], name="Best Validation Loss", mode='markers',
                       marker_symbol='x', marker_size=10),
            go.Scatter(y=[np.min(trn_epoch_losses)],
                       x=[np.argmin(trn_epoch_losses)], name="Best Train Loss", mode='markers',
                       marker_symbol='x', marker_size=10),
            # go.Scatter(y=[np.min(trn_val_epoch_losses)],
            #            x=[np.argmin(trn_val_epoch_losses)], name="Best Train Valid Loss", mode='markers',
            #            marker_symbol='x', marker_size=10),

        ],
        layout=dict(
            title="Train and Validation Losses per epoch",
            title_x=0.5,
        )
    )


def plot_interactive_roc(data_path):
    metrics = get_models_metrics(data_path)
    fig = go.Figure(
        layout=dict(
            title_text="Receiver Operating Characteristic Curve",
            title_x=0.5,
            height=650,
            #         width=650,
            #         yaxis=dict(scaleanchor="x", scaleratio=1),
            yaxis_title='True Positive Rate',
            xaxis_title='False Positive Rate',
            yaxis=dict(range=[-0.01, 1.01]),
            xaxis=dict(range=[-0.01, 1.01]),
        ),
    )

    for metric in metrics:
        fig.add_trace(
            go.Scatter(
                y=metric.roc_curve.tpr,
                x=metric.roc_curve.fpr,
                name=f"{metric.model_name} (AUC={metric.roc_curve.auc:.3f})"
            )
        )

    return fig


def plot_interactive_det(data_path):
    metrics = get_models_metrics(data_path)
    fig = go.Figure(
        layout=dict(
            title_text="Detection Error Tradeoff (DET) Curve",
            title_x=0.5,
            height=650,
            #         width=650,
            #         yaxis=dict(scaleanchor="x", scaleratio=1),
            yaxis_title='False Negative Rate',
            xaxis_title='False Positive Rate',
            yaxis=dict(range=[0, 100], type="log"),
            xaxis=dict(range=[0, 100], type="log"),
        ),
    )

    for metric in metrics:
        fnr_scaled = metric.det_curve.fnr
        fpr_scaled = metric.det_curve.fpr

        fig.add_trace(
            go.Scatter(
                y=fnr_scaled,
                x=fpr_scaled,
                name=f"{metric.model_name} (EER={metric.det_curve.eer:.3f})"
            )
        )

    return fig


def plot_interactive_pr(data_path, beta=1):
    metrics = get_models_metrics(data_path)
    fig = go.Figure(
        layout=dict(
            title_text="Precision-Recall Curve",
            title_x=0.5,
            height=650,
            #         width=650,
            #         yaxis=dict(scaleanchor="x", scaleratio=1),
            yaxis_title='Precision',
            xaxis_title='Recall',
            yaxis=dict(range=[-0.01, 1.01]),
            xaxis=dict(range=[-0.01, 1.01]),
        ),

    )

    f_scores = [.3, .4, .5, .6, .7, .8, .9]
    for i, f_score in enumerate(f_scores):
        x = np.linspace(0.001, 1.1, 1000)
        y = f_score * x / ((1 + (beta ** 2)) * x - f_score * (beta ** 2))

        f_mask = (y >= 0) & (y <= 1.3) & (x >= 0) & (x <= 1.1)
        fig.add_trace(
            go.Scatter(
                y=y[f_mask],
                x=x[f_mask],
                name=f'F{beta}={f_score:0.1f}',
                marker=dict(color='black'),
                opacity=0.1,
                mode='lines',
            )
        )

    for metric in metrics:
        text = list(
            map(lambda x: f"F{beta}={x:.4f}",
                map(safe_f1_score, zip(metric.pr_curve.precision, metric.pr_curve.recall))))

        fig.add_trace(
            go.Scatter(
                y=metric.pr_curve.precision,
                x=metric.pr_curve.recall,
                text=text,
                name=f"{metric.model_name} (AP={metric.pr_curve.ap:.3f})"
            )
        )

    return fig


def plot_interactive_roc_(y_true, y_score, model_name='model'):
    fig = go.Figure(
        layout=dict(
            title_text="Receiver Operating Characteristic Curve",
            title_x=0.5,
            height=650,
            #         width=650,
            #         yaxis=dict(scaleanchor="x", scaleratio=1),
            yaxis_title='True Positive Rate',
            xaxis_title='False Positive Rate',
            yaxis=dict(range=[-0.01, 1.01]),
            xaxis=dict(range=[-0.01, 1.01]),
        ),
    )

    fpr, tpr, thresh = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig.add_trace(
        go.Scatter(
            y=tpr,
            x=fpr,
            name=f"{model_name} (AUC={auc:.3f})"
        )
    )

    return fig


def plot_interactive_pr_(y_true, y_score, model_name='model'):
    fig = go.Figure(
        layout=dict(
            title_text="Precision Recall Curve",
            title_x=0.5,
            height=650,
            #         width=650,
            #         yaxis=dict(scaleanchor="x", scaleratio=1),
            yaxis_title='Precision',
            xaxis_title='Recall',
            yaxis=dict(range=[-0.01, 1.01]),
            xaxis=dict(range=[-0.01, 1.01]),
        ),
    )

    precision, recall, thresh = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig.add_trace(
        go.Scatter(
            y=precision,
            x=recall,
            name=f"{model_name} (AP={ap:.3f})"
        )
    )

    return fig


def interactive_crime_prediction_comparison(y_class: np.ndarray,
                                            y_score: np.ndarray,
                                            t_range: DatetimeIndex,
                                            height: int = 500,
                                            y_count=None):
    """
    Plots interactive widget with a grid of the mean over time that can be used to select cells and view their scores
    over time. The crime occurrences are also overlain on the curve.


    :param y_class: crime/no-crime grids (N,H,W)
    :param y_score: crime score grids (N,H,W)
    :param height: height of grid images
    :param t_range:
    :param y_count: actual crime counts to be able to compare with the predicted values - is an optional value.
    :return:
    """
    assert y_class.shape == y_score.shape
    assert len(y_class.shape) == 3

    y_score_mean = y_score.mean(0)
    h, w = y_score_mean.shape

    width = height * w / h  # 300

    fw_grid = go.FigureWidget(
        go.Heatmap(z=y_score_mean),
        layout=dict(
            height=height,
            width=width,
            title_x=0.5,
            title='Mean over Time',
            margin=dict(l=20, r=20, t=50, b=20),
        ),
    )

    fw_lines = go.FigureWidget(
        data=[go.Scatter(name='Predicted Score'),
              go.Scatter(name='Occurrence', mode='markers', )],
        layout=dict(
            width=900,
        )
    )

    y_score_line = fw_lines.data[0]
    y_class_line = fw_lines.data[1]

    if y_count:
        fw_lines.add_trace(go.Scatter(name='Crime Counts'))
        y_count_line = fw_lines.data[-1]

    state = State()

    def draw():
        x = state['x']
        y = state['y']

        occ = y_class[:, y, x]
        args = np.argwhere(occ > 0)[:, 0]

        with fw_lines.batch_update():
            fw_lines.update_layout(title={"text": f"Selected cell: y,x = {y, x}"}, title_x=0.5)
            y_score_line.y = y_score[:, y, x]
            y_score_line.x = t_range

            y_class_line.y = y_score[args, y, x]
            y_class_line.x = t_range[args]

    def set_state(_trace, points, _selector):
        y = points.ys[0]
        x = points.xs[0]

        state["x"] = x
        state["y"] = y
        draw()

    fw_grid.data[0].on_click(set_state)

    box_layout = Layout(display='flex',
                        flex_flow='column',
                        align_items='center',
                        width='100%')

    return widgets.Box(
        children=[fw_grid, fw_lines],
        layout=box_layout,
    )


def cmi_name(temporal_variables):
    cond_var_map = {
        'Hour': 'H_t',
        'Day of Week': 'DoW_t',
        'Time of Month': 'ToM_t',
        'Time of Year': 'ToY_t',
    }

    return ",".join([cond_var_map[k] for k in temporal_variables])


def interactive_grid_visualiser(grids: np.ndarray, t_range: DatetimeIndex, height: int = 500, **kwargs):
    """

    :param grids: (N: time dimension,H: vertical/height,W: horizontal/width dimension) format of ndarray
    :param t_range:
    :param height: height of grid images
    
    kwargs: 
        - 'mutual_info'=True adds a mutual information plot
        - 'max_offset' sets the conditional
    :return:
    """
    assert len(grids.shape) == 3

    grids_mean = grids.mean(0)
    h, w = grids_mean.shape

    width = height * w / h  # 300

    fw_grid = go.FigureWidget(
        go.Heatmap(z=grids_mean),
        layout=dict(
            height=height,
            width=width,
            title_x=0.5,
            title='Mean over Time',
            margin=dict(l=20, r=20, t=50, b=20),
        ),
    )

    fw_lines = go.FigureWidget(
        data=[go.Scatter(name='Cell Value Over Time')],
        layout=dict(
            width=900,
        )
    )
    grid_line = fw_lines.data[0]
    figs = [fw_grid, fw_lines]

    if kwargs.get("mutual_info"):
        temporal_variables = kwargs.get('temporal_variables')
        if temporal_variables is None:
            if t_range.freqstr == 'H':
                temporal_variables = ["Hour", "Day of Week"]
            else:
                temporal_variables = ["Day of Week", "Time of Month", "Time of Year"]

        conds = construct_temporal_information(
            date_range=t_range,
            temporal_variables=temporal_variables,
            month_divisions=kwargs.get('month_divisions', 10),
            year_divisions=kwargs.get('year_divisions', 10),
        ).values

        fw_mi = go.FigureWidget(
            data=[
                go.Scatter(name='$I(C_{t},C_{t-k})$'),
                go.Scatter(name=f'$I(C_{{t}},C_{{t-k}}|{cmi_name(temporal_variables)})$'),
            ],
            layout=dict(
                title='Mutual Information and Conditional Mutual Information with Time Lagged Signal',
                title_x=0.5,
                width=900,
            )
        )
        mi_line, cmi_line = fw_mi.data
        figs.append(fw_mi)

    state = State()

    def draw():
        x = state['x']
        y = state['y']

        with fw_lines.batch_update():
            z = grids[:, y, x]

            fw_lines.update_layout(title={"text": f"Selected cell: y,x = {y, x}"}, title_x=0.5)
            grid_line.y = z
            grid_line.x = t_range

            if kwargs.get("mutual_info"):
                my, mx = mutual_info_over_time(a=z,
                                               max_offset=kwargs.get('max_offset', 30),
                                               bins=kwargs.get('bins', 0))

                cy, cx = conditional_mutual_info_over_time(a=z,
                                                           max_offset=kwargs.get('max_offset', 30),
                                                           bins=kwargs.get('bins', 0),
                                                           conds=conds)
                mi_line.y, mi_line.x = my, mx
                cmi_line.y, cmi_line.x = cy, cx

    def set_state(_trace, points, _selector):
        y = points.ys[0]
        x = points.xs[0]

        state["x"] = x
        state["y"] = y
        draw()

    fw_grid.data[0].on_click(set_state)

    box_layout = Layout(display='flex',
                        flex_flow='column',
                        align_items='center',
                        width='100%')

    return widgets.Box(
        children=figs,
        layout=box_layout,
    )
