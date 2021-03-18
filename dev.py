# re-writing the sequence training and evaluation notebook (Development - Whole city crime count predictor (Daily))
# TODO: ensure that the code works for hourly, daily and weekly
# NOTE THAT THIS FILE LEAVES OUT DATA EXPLORATION AND EXPERIMENTATION AND MAINLY FOCUSES ON THE TRAINING AND EVALUATION
# OF THE DL AND BASE-LINE MODELS - OUR MAIN CONCERN IS THAT THE TEST SET SIZES ARE NOT THE SAME FOR ALL MODELS, AND
# THAT THE TEST SET BASELINE MODELS ARE TRAINED ON TEST SET DATA AND NOT


### IMPORTS --------------------------------------------------------------------------------------------------

from utils.plots import plot, displot
from seaborn import distplot
import matplotlib.pyplot as plt
from models.baseline_models import historic_average

import os
import logging as log
from time import strftime
from copy import deepcopy
from torch import nn, optim
import torch.nn.functional as F
from utils.data_processing import *
from logger.logger import setup_logging
from utils.configs import BaseConf
from utils.utils import write_json, Timer
from models.kangkang_fnn_models import KangFeedForwardNetwork, SimpleKangFNN, evaluate_fnn
from dataloaders.flat_loader import FlatDataLoaders, MockLoader, MockLoaders
from datasets.flat_dataset import FlatDataGroup
from utils.metrics import PRCurvePlotter, ROCCurvePlotter, LossPlotter, PerTimeStepPlotter
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from models.model_result import ModelResult, ModelMetrics, save_results, save_metrics, \
    compare_all_models, get_models_metrics
from utils.mock_data import mock_fnn_data_classification
from utils.plots import im
from utils.utils import pshape, get_data_sub_paths, by_ref
from trainers.generic_trainer import train_model

from utils.metrics import best_threshold, get_y_pred, get_y_pred_by_thresholds, best_thresholds
from time import time
from torch.optim import lr_scheduler

from pprint import pprint
import logging

from utils.constants import NOT_TOTAL, TOTAL

from matplotlib import rcParams
from utils.forecasting import compare_time_series_metrics

from utils.utils import load_total_counts, to_title, set_system_seed
from utils.data_processing import normalize_df_mean_std, normalize_df_min_max
from utils.plots import plot, plot_df, plot_time_signals, plot_autocorr, subplots_df

from utils.forecasting import forecast_metrics

from models.sequence_models import train_epoch_for_sequence_model, evaluate_sequence_model
import plotly.graph_objects as go

import pingouin as pg  # used to allow partial correlation of dataframes

from trainers.generic_trainer import train_model_final

from utils.testing import assert_no_nan, assert_valid_datetime_index

os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())
rcParams["font.family"] = "STIXGeneral"

import plotly.io as pio

# ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
pio.templates.default = "plotly_white"  # "none"

from datasets.sequence_dataset import SequenceDataLoaders
from models.rnn_models import GRUFNN

### START CODE --------------------------------------------------------------------------------------------------

data_sub_paths = [i for i in get_data_sub_paths() if i.startswith('Totals')]
print(data_sub_paths)

#  if train/test sets should overlap by the sequence length of i put vector
#  only of concern if the loss functions in the training loop use the whole sequence outputs to calculate the loss - which in turn leads to quicker train times.
overlap_sequences = False  # True

data_sub_path = data_sub_paths[0]
save_folder = f"./data/processed/{data_sub_path}/plots/"
os.makedirs(save_folder, exist_ok=True)
FREQ = data_sub_path.lstrip('Totals_T').split('_')[0]
time_steps_per_day = 24 / int(FREQ[:-1])

freq_title = {
    "24H": "Daily",
    "1H": "Hourly",
    "168H": "Weekly",
}.get(FREQ, "Hourly")
print(f"Using: {freq_title} ({FREQ})")

df = load_total_counts(folder_name=data_sub_path)

if FREQ == "168H":
    # data is split into weeks starting on monays - we need to trim first
    # and last weeks because they are technically half weeks because they don't start on mondays
    df = df.iloc[1:-1]

from IPython.core.display import display

display(df)
# plot_df(df).show()
subplots_df(df[NOT_TOTAL], title='Crime Counts Over Time', xlabel='Date', ylabel='Count', ncols=2).show()

if torch.cuda.is_available():
    print(torch.cuda.memory_summary())
else:
    raise Exception("CUDA is not available")

## CONFIG AND LOGGING SETUP ----------------------------------------------------------------------------------------
# manually set
conf = BaseConf()
conf.seed = int(time())  # 3
set_system_seed(conf.seed)
conf.model_name = f"{freq_title} City Count"  # "SimpleKangFNN" # "KangFNN"  # needs to be created
conf.data_path = f"./data/processed/{data_sub_path}/"

if not os.path.exists(conf.data_path):
    raise Exception(f"Directory ({conf.data_path}) needs to exist.")

conf.model_path = f"{conf.data_path}models/{conf.model_name}/"
os.makedirs(conf.data_path, exist_ok=True)
os.makedirs(conf.model_path, exist_ok=True)

# logging config is set globally thus we only need to call this in this file
# imported function logs will follow the configuration
setup_logging(save_dir=conf.model_path, log_config='./logger/standard_logger_config.json', default_level=log.INFO)
log.info("=====================================BEGIN=====================================")

info = deepcopy(conf.__dict__)
info["start_time"] = strftime("%Y-%m-%dT%H:%M:%S")

# DATA LOADER SETUP
use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")
log.info(f"Device: {device}")
info["device"] = device.type
conf.device = device

# SET THE HYPER PARAMETERS
conf.shaper_top_k = -1
conf.use_classification = False
conf.train_set_first = True
conf.use_crime_types = False

## ANOMALY DETECTION PLOTS ----------------------------------------------------------------------------------------

from utils.mutual_information_plots import subplot_mi_curves, plot_mi_curves
from utils.data_processing import to_percentile
from utils.rolling import rolling_norm, flag_anomalies, periodic_rolling_mean

plot_anomalies = False
if plot_anomalies:
    window = {
        "1H": 501,
        "24H": 51,
        "168H": 13,
    }.get(FREQ)

    period = 1

    thresh = 3

    logging.warning("Plot outliers are done with symmetric windowing and are " +
                    "only used to flag outliers not predict them")

    for col in df.columns:
        a = df[col].values

        anoms = flag_anomalies(
            data=a, thresh=thresh, window=window, period=period, center=True, mode='reflect')

        ma = periodic_rolling_mean(data=a, window=window, period=period, center=True)
        normed = rolling_norm(data=a, window=window, period=period, center=True)

        fig = go.Figure(
            data=[
                go.Scatter(x=df.index, y=a, opacity=.5, name=f'Counts'),
                go.Scatter(x=df.index[anoms], y=a[anoms], mode='markers', opacity=.5, name='Outliers'),
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
        fig.write_image(f"{save_folder}{FREQ}_outliers_{col}.png".replace(' ', '_'))
        fig.show()

## MUTUAL INFORMATION PLOTS ------------------------------------------------------------------------------------------

# from utils.plots import subplots_df, plot_df
# from utils.mutual_information_plots import plot_mi_curves
plot_mi_curves = False
if plot_mi_curves:
    temporal_variables = {
        "1H": ["Hour", "Day of Week", "Time of Month", "Time of Year"],
        "1H": ["Hour", "Day of Week"],
        "24H": ["Day of Week", "Time of Month", "Time of Year"],
        "168H": ["Time of Month", "Time of Year"],
    }.get(FREQ, ["Time of Month", "Time of Year"])

    max_offset = {
        "1H": 168 * 2,
        "24H": 365,
        "168H": 54,
    }.get(FREQ)

    for i, name in enumerate(df.columns):
        a = df[name].values

        mutual_info_bins = 16  # 16
        #         print(f"optimal bins: {get_optimal_bins(a)}")
        #         a = to_percentile(a)
        #         a = np.round(np.log(1+a)) # whatch out for values between 1024 2048
        #         a = cut(np.log(1+a)) # whatch out for values between 1024 2048

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
            title=f'{freq_title} {name} Mutual and Conditional Mutual Information',
            a_title=f'{freq_title} {name} City Wide Counts',
        )
        fig.write_image(f"{save_folder}{FREQ}_mi_plots_{name}.png".replace(' ', '_'))
        fig.show()

"""
### Periodic Rolling Normalisation ---------------------------------------------------------------------------
By applying rolling normalisation we can control for the period nature of our signals. We subtract the signal by a 
periodic rolling mean and divide it by a period rolling standard deviation. We can also perform this on various 
periods e.g. 7 days and 365 days, which caters for weekly and yearly periodic cycles. By controlling for the signals 
periodic nature we essentially have a new signal show casing how many standard deviations a current time step is 
outside of the expected value for that day of the week and time of the year. Our potential models then just need 
to predict the residuals of our crime signals.
"""

norm_offset = 0
corr_subplots = False
normalize_periodically = True
if not normalize_periodically:
    normed_df = df.copy()
else:
    logging.warning("Using rolling norm means values at the " +
                    "start within the window will be set to NaN and dropped")

    max_offset = {
        "1H": 168 * 2 + 24,
        "24H": 365,
        "168H": 54,
    }.get(FREQ)

    fig = plot_autocorr(**df,
                        title="Autocorrelations by Crime Type before any Rolling Normalisation",
                        partial=False,
                        max_offset=max_offset,
                        subplots=corr_subplots)
    fig.write_image(f"{save_folder}{FREQ}_auto_corr_normed_none.png".replace(' ', '_'))
    fig.show()

    window, period, period_string = {
        "24H": (53, 7, "Weekly"),  # jumps in weeks
        "1H": (366, 24, "Daily"),  # jumps in days
        "168H": (52, 1, "Weekly"),  # jumps in weeks
        #         "168H": (10,52, "Yearly"),  # jumps in years
    }.get(FREQ, (10, 1))
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
                        title=f"Autocorrelations by Crime Type after {period_string} Rolling Normalisation",
                        max_offset=max_offset, subplots=corr_subplots)
    fig.write_image(f"{save_folder}{FREQ}_auto_corr_normed_{period_string.lower()}.png".replace(' ', '_'))
    fig.show()

    fig = plot_df(df[TOTAL], xlabel="Date", ylabel="Count",
                  title=f"Total Crimes before any Rolling Normalisation")
    fig.write_image(f"{save_folder}{FREQ}_total_crimes_normed_none.png".replace(' ', '_'))
    fig.show()

    fig = plot_df(normed_df[TOTAL], xlabel="Date", ylabel="Scaled Count",
                  title=f"Total Crimes after {period_string} Rolling Normalisation")
    fig.write_image(f"{save_folder}{FREQ}_total_crimes_normed_{period_string.lower()}.png".replace(' ', '_'))
    fig.show()

    double_rolling_norm = True
    if double_rolling_norm:
        window2, period2, period_string2 = {
            "24H": (5, 365, "Yearly"),  # jumps in years
            "1H": (53, 168, "Weekly"),  # jumps in weeks
            "168H": (10, 52, "Yearly"),  # jumps in years
        }.get(FREQ, (None, None, None))
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
                                title=f"Autocorrelations by Crime Type after {period_string} and" +
                                      f" {period_string2} Rolling Normalisation",
                                max_offset=max_offset, subplots=corr_subplots)
            fig.write_image(f"{save_folder}{FREQ}_auto_corr_normed_{period_string.lower()}_" +
                            f"{period_string2.lower()}.png".replace(' ', '_'))
            fig.show()

            fig = plot_df(
                df=pd.DataFrame(normed_df[TOTAL]),
                xlabel="Date", ylabel="Scaled Count",
                title=f"Total Crimes after {period_string} and {period_string2} Rolling Normalisation",
            )
            #             fig = subplots_df(
            #                 df=normed_df[NOT_TOTAL],
            #                 xlabel="Date",ylabel="Scaled Count",
            #                 title=f"Total Crimes after {period_string} and {period_string2} Rolling Normalisation",
            #             )
            fig.write_image(
                f"{save_folder}{FREQ}_total_crimes_normed_{period_string.lower()}_" +
                f"{period_string2.lower()}.png".replace(' ', '_'))
            fig.show()
    assert len(np.unique(np.diff(normed_df.index.astype(int)))), \
        "Normed values are not contiguous, dropna method might have dropped values"

# total_df = normalize_df_min_max(normed_df) # rescale between 0 and 1 again...is this realy needed?
# total_df = normalize_df_mean_std(normed_df) # rescale between 0 and 1 again...is this realy needed?
total_df = normed_df
# total_df = df
# total_df = normalize_df_min_max(df)
# total_df = normalize_df_mean_std(df)
total_df.plot(kind='kde', alpha=0.4, title='Normalised Crime Counts Distributions')
fig = plt.gcf()
fig.set_size_inches(9, 6)
plt.grid()
plt.savefig(f"{save_folder}{FREQ}_crime_counts_distribution_normed.png")
plt.show()
display(total_df)

from utils.plots import plot_corr

for temp, _normed, period_string in zip([df, total_df], ['', '_normed'], ['', ' after Normalisation']):
    corr = temp.loc[:, temp.columns != 'Total'].corr()
    plot_corr(corr, title=f'Pearson Correlation Coefficients Between {freq_title} Crime Types{period_string}\n')
    plt.savefig(f"{save_folder}{FREQ}_corr_matrix_pearson{_normed}.png")

    corr = temp.loc[:, temp.columns != 'Total'].pcorr()
    plot_corr(corr, title=f'Partial Pearson Correlation Coefficients ' +
                          f'Between {freq_title} Crime Types{period_string}\n')
    plt.savefig(f"{save_folder}{FREQ}_corr_matrix_pearson_partial{_normed}.png")

    corr = temp.loc[:, temp.columns != 'Total'].corr(method='spearman')
    plot_corr(corr, title=f'Spearman Correlation Coefficients Between {freq_title} Crime Types{period_string}\n')
    plt.savefig(f"{save_folder}{FREQ}_corr_matrix_spearman{_normed}.png")

t_range = total_df.index[1:]  # should be equal to the target times
total_crime_types = total_df.values

time_vectors = encode_time_vectors(t_range)

plot_time_vectors = True
if plot_time_vectors:
    k = int(time_steps_per_day * 365)
    tv, tr = time_vectors[:k], t_range[:k]

    t_vec_names = {
        "1H": ['$H_{sin}$', '$H_{cos}$', '$DoW_{sin}$', '$DoW_{cos}$',
               '$ToM_{sin}$', '$ToM_{cos}$', '$ToY_{sin}$', '$ToY_{cos}$', '$Wkd$'],
        "24H": ['$DoW_{sin}$', '$DoW_{cos}$', '$ToM_{sin}$',
                '$ToM_{cos}$', '$ToY_{sin}$', '$ToY_{cos}$', '$Wkd$'],
        "168H": ['$ToM_{sin}$', '$ToM_{cos}$', '$ToY_{sin}$', '$ToY_{cos}$'],
    }.get(FREQ)

    pio.templates.default = "plotly"
    fig = go.Figure(
        data=go.Heatmap(
            z=tv.T,
            y=t_vec_names,
            x=tr,
        ),
        layout=dict(
            title=f"Encoded Time Vectors on {freq_title} Level",
            title_x=0.5,
            xaxis_title="Date",
            yaxis_title="Encoded Vector Values",
            font=dict(family="STIXGeneral"),
        ),
    )
    fig.write_image(f"{save_folder}{FREQ}_time_vector_encoding.png")
    fig.show()
    pio.templates.default = "plotly_white"

## SETUP DATA LOADERS ------------------------


# conf.seq_len = 60 # int(14*time_steps_per_day) + 1
conf.seq_len = {
    "24H": 90,
    "1H": 168,
    "168H": 52,
}.get(FREQ, 60)

conf.batch_size = 128

logging.info(f"Using sequence length: {conf.seq_len}")

input_data = total_crime_types[:-1]

predict_only_total = True
if predict_only_total:
    target_data = total_crime_types[1:, 0:1]
else:
    target_data = total_crime_types[1:]

assert len(input_data) == len(time_vectors), \
    f"len(input_data) != len(time_vectors), {len(input_data)},{len(time_vectors)}"

use_time_vectors = True
if use_time_vectors:
    logging.info("using time vectors in input")
    input_data = np.concatenate([input_data, time_vectors], axis=1)
else:
    logging.info("NOT using time vectors in input")

input_size = input_data.shape[-1]
output_size = target_data.shape[-1]

assert len(input_data) == len(t_range)

test_size = {
    "24H": 365,
    "H": 8760,
    "1H": 8760,
    "168H": 52 * 2,
}.get(FREQ)

tst_ratio = test_size / len(input_data)

loaders = SequenceDataLoaders(  # setup data loader 1
    input_data=input_data,
    target_data=target_data,
    t_range=t_range,
    batch_size=conf.batch_size,
    seq_len=conf.seq_len,
    shuffle=conf.shuffle,
    num_workers=0,
    val_ratio=0.2,  # 0.5,
    tst_ratio=tst_ratio,
    overlap_sequences=overlap_sequences,
)

input_data.shape, target_data.shape

## HYPER PARAMETER OPTIMISATION ---------------------------------------------------------------

from pprint import pformat


def train_evaluate_hyper(hyper_parameters):
    logging.info(f"Running HyperOpt Trial with: {pformat(hyper_parameters)}")

    # hyper param setup
    hidden_size = int(hyper_parameters.get('hidden_size', 50))
    num_layers = int(hyper_parameters.get('num_layers', 5))
    conf.weight_decay = hyper_parameters.get('weight_decay', 1e-4)
    conf.lr = hyper_parameters.get('lr', 1e-3)

    default_seq_len = {
        "24H": 90,
        "1H": 168,
        "168H": 52,
    }.get(FREQ, 60)
    conf.seq_len = int(hyper_parameters.get('seq_len', default_seq_len))

    conf.early_stopping = True
    conf.patience = 30
    conf.min_epochs = 1
    conf.max_epochs = 10_000

    conf.seed = np.random.randint(10_000) * (int(time()) % hidden_size + num_layers)

    set_system_seed(conf.seed)  # should be reset with each model instatiation

    loaders = SequenceDataLoaders(  # setup data loader 2: hyper opt
        input_data=input_data,
        target_data=target_data,
        t_range=t_range,
        batch_size=conf.batch_size,
        seq_len=conf.seq_len,
        shuffle=conf.shuffle,
        num_workers=0,
        val_ratio=0.5,
        tst_ratio=tst_ratio,
        overlap_sequences=overlap_sequences,
    )

    # model setup
    model = GRUFNN(
        input_size=input_size,
        hidden_size0=hidden_size,
        hidden_size1=hidden_size // 2,
        output_size=output_size,
        num_layers=num_layers,
    ).to(conf.device)

    criterion = nn.MSELoss()
    optimiser = torch.optim.AdamW(params=model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)

    # training
    trn_epoch_losses, val_epoch_losses, stopped_early = train_model(
        model=model,
        optimiser=optimiser,
        loaders=loaders,
        train_epoch_fn=train_epoch_for_sequence_model,
        loss_fn=criterion,
        conf=conf,
        verbose=False,
    )

    # load the saved best validation model
    # Load latest or best validation model
    conf.checkpoint = "best_val"
    log.info(f"Loading model from checkpoint ({conf.checkpoint}) for evaluation")
    log.info(f"Loading model from {conf.model_path}")
    model_state_dict = torch.load(f"{conf.model_path}model_{conf.checkpoint}.pth",
                                  map_location=conf.device.type)
    model.load_state_dict(model_state_dict)

    # evaluation
    val_y_true, val_y_score = evaluate_sequence_model(model, loaders.validation_loader, conf)

    return forecast_metrics(y_true=val_y_true, y_score=val_y_score)['MASE']


total_hyper_opt_trials = 20  # 20
optimize_hyper_params = True
if optimize_hyper_params:
    # Hyper Opt Training

    from ax.plot.contour import plot_contour
    from ax.plot.trace import optimization_trace_single_method
    from ax.service.managed_loop import optimize
    from ax.utils.notebook.plotting import render, init_notebook_plotting
    from ax.utils.tutorials.cnn_utils import load_mnist, train, evaluate, CNN

    # PARAM_CLASSES = ["range", "choice", "fixed"]
    # PARAM_TYPES = {"int": int, "float": float, "bool": bool, "str": str}
    # potential parameters
    max_seq_len = {
        "24H": 366,
        "1H": 24 * 7 * 6,
        "168H": 64,
    }.get(FREQ, 60)

    hyper_params = [
        {"name": "lr", "type": "range", "bounds": [1e-4, 1e-2], "log_scale": True},
        {"name": "weight_decay", "type": "range", "bounds": [1e-8, 1e-2], "log_scale": True},
        #         {"name": "hidden_size", "type": "choice", "values": [8,16,32,64,128]},
        #         {"name": "num_layers", "type": "choice", "values": [1,2,4,8]},
        #         {"name": "seq_len", "type": "choice", "values": [7,24,64,128]},
        {"name": "hidden_size", "type": "range", "bounds": [8, 128]},
        {"name": "num_layers", "type": "range", "bounds": [1, 8]},
        {"name": "seq_len", "type": "range", "bounds": [8, max_seq_len]},
    ]

    best_parameters, values, experiment, hyper_model = optimize(
        parameters=hyper_params,
        evaluation_function=train_evaluate_hyper,
        objective_name='MASE',
        minimize=True,  # change when the objective is a loss
        total_trials=total_hyper_opt_trials,
        #     random_seed=1, # optional
    )

    experiment_data = experiment.fetch_data()
    display(experiment_data.df)
    display(pd.DataFrame([best_parameters]))

#     render(plot_contour(model=hyper_model, param_x='lr', param_y='weight_decay', metric_name='MASE'))
else:
    # default values to set when not using hyper-parameter optimization
    best_parameters = {
        'lr': 1e-3,
        'weight_decay': 1e-6,
        'hidden_size': 8,
        'num_layers': 1,
        'seq_len': 28,
    }

write_json(best_parameters, f"{save_folder}{FREQ}_best_hyper_params.json".replace(' ', '_'))

display(pd.DataFrame([best_parameters]).iloc[0])


## MULTI-RUN EVALUATION -------------------------------------------------------------------------
# TODO: EXTRACT INTO A CLASS WRAPPER LIKE DONE WITH TENSORFLOW AND KERAS

def run_trials(num_trials=10):
    """
    Run a full experiment on GRUFNN model multiple times with different seeds
    Data and hyper parameters must be set before hand.
    This function acts as a closure, i.e. some variables are created outside the scope of the function.

    Used to:
        1. Setup data loader with predefined input and target data
        2. Setup model
        3. Train model with a validation set once to determine the num_epochs
        4. Run trials loops:
            4.a. Reset seed
            4.b. Resetup model
            4.c. Train with train_val set for predetermined num_epochs
            4.d. Run evaluations on the model
        5. Return a dataframe with seed and forecast metrics for each trial run
    """

    trial_metrics_list = []

    conf.early_stopping = True
    conf.patience = 30
    conf.min_epochs = 1
    conf.max_epochs = 10_000

    conf.lr = best_parameters['lr']
    conf.weight_decay = best_parameters['weight_decay']
    hidden_size = best_parameters['hidden_size']
    num_layers = best_parameters['num_layers']
    conf.seq_len = best_parameters['seq_len']

    # ====================================== can be put into loop as well
    # model setup
    conf.seed = int(time())  # unique seed for each run
    set_system_seed(conf.seed)  # should be reset with each model instatiation

    loaders = SequenceDataLoaders(  # setup data loader 3: run trial
        input_data=input_data,
        target_data=target_data,
        t_range=t_range,
        batch_size=conf.batch_size,
        seq_len=conf.seq_len,
        shuffle=conf.shuffle,
        num_workers=0,
        val_ratio=0.5,
        tst_ratio=tst_ratio,
        overlap_sequences=overlap_sequences,
    )

    model = GRUFNN(
        input_size=input_size,
        hidden_size0=hidden_size,
        hidden_size1=hidden_size // 2,
        output_size=output_size,
        num_layers=num_layers,
    ).to(conf.device)

    criterion = nn.MSELoss()

    optimiser = torch.optim.AdamW(params=model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)

    trn_epoch_losses, val_epoch_losses, stopped_early = train_model(
        model=model,
        optimiser=optimiser,
        loaders=loaders,
        train_epoch_fn=train_epoch_for_sequence_model,
        loss_fn=criterion,
        conf=conf,
        verbose=False,
    )

    logging.info(f"best validation: {np.min(val_epoch_losses):.6f} @ epoch: {np.argmin(val_epoch_losses) + 1}")

    # full train-val dataset training
    conf.max_epochs = np.argmin(val_epoch_losses) + 1  # because of index starting at zero
    # ====================================== can be put into loop as well

    for i in range(num_trials):
        logging.info(f"Starting trial {i + 1} of {num_trials}.")

        conf.seed = int(time() + 10 * i)  # unique seed for each run
        set_system_seed(conf.seed)  # should be reset with each model instatiation
        model = GRUFNN(
            input_size=input_size,
            hidden_size0=hidden_size,
            hidden_size1=hidden_size // 2,
            output_size=output_size,
            num_layers=num_layers,
        ).to(conf.device)

        optimiser = torch.optim.AdamW(params=model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)

        trn_val_epoch_losses = train_model_final(
            model=model,
            optimiser=optimiser,
            loaders=loaders,
            train_epoch_fn=train_epoch_for_sequence_model,
            loss_fn=criterion,
            conf=conf,
        )

        tst_y_true, tst_y_score = evaluate_sequence_model(model, loaders.test_loader, conf)
        trial_metrics = forecast_metrics(y_true=tst_y_true, y_score=tst_y_score)
        trial_metrics['Seed'] = conf.seed
        trial_metrics_list.append(trial_metrics)

        logging.info(f"Completed trial {i + 1} of {num_trials}.")

    return trial_metrics_list


num_trials = 10
trial_results = run_trials(num_trials=num_trials)

display(pd.DataFrame(trial_results))

pd.DataFrame(trial_results)[['MASE']].plot(kind='box',
                                           title=f"{freq_title} Crime GRUFFN MASE Score for {num_trials} Seeds\n")
plt.savefig(f"{save_folder}{FREQ}_result_seed_consistency.png")
plt.show()

## SETUP MODELS WITH HYPER-PARAMETERS FROM PREVIOUS STEP -------------------------------------------

# set hyper params-> set seed -> set model -> set optimiser
conf.seed = int(time())  # 1607355910

conf.lr = best_parameters['lr']
conf.weight_decay = best_parameters['weight_decay']
hidden_size = best_parameters['hidden_size']
num_layers = best_parameters['num_layers']
conf.seq_len = best_parameters['seq_len']

set_system_seed(conf.seed)  # should be reset with each model instatiation
model = GRUFNN(
    input_size=input_size,
    hidden_size0=hidden_size,
    hidden_size1=hidden_size // 2,
    output_size=output_size,
    num_layers=num_layers,
).to(conf.device)

criterion = nn.MSELoss()

optimiser = torch.optim.AdamW(params=model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)

# train model with early stopping -> get max epochs for train validation set

conf.resume = False
conf.checkpoint = "final"  # "latest" # ["best_val", "best_trn", "best_trn_val"]
conf.early_stopping = True
conf.patience = 30
conf.min_epochs = 100
conf.max_epochs = 10_000

trn_epoch_losses, val_epoch_losses, stopped_early = train_model(
    model=model,
    optimiser=optimiser,
    loaders=loaders,
    train_epoch_fn=train_epoch_for_sequence_model,
    loss_fn=criterion,
    conf=conf,
)

logging.info(f"best validation: {np.min(val_epoch_losses):.6f} @ epoch: {np.argmin(val_epoch_losses) + 1}")

# plot train and validation sets

from utils.interactive import plot_interactive_epoch_losses

fig = plot_interactive_epoch_losses(trn_epoch_losses, val_epoch_losses)
fig.write_image(f"{save_folder}{FREQ}_loss_val_plot.png")
fig.show()

# set max epochs -> reset to same seed as before -> recreate model -> recreate optimiser -> train final model

conf.max_epochs = np.argmin(val_epoch_losses) + 1  # because of index starting at zero

set_system_seed(conf.seed)  # should be reset with each model instatiation
model = GRUFNN(
    input_size=input_size,
    hidden_size0=hidden_size,
    hidden_size1=hidden_size // 2,
    output_size=output_size,
    num_layers=num_layers,
).to(conf.device)

optimiser = torch.optim.AdamW(params=model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)

trn_val_epoch_losses = train_model_final(
    model=model,
    optimiser=optimiser,
    loaders=loaders,
    train_epoch_fn=train_epoch_for_sequence_model,
    loss_fn=criterion,
    conf=conf,
)

plot(trn_val_epoch_losses=trn_val_epoch_losses)

## TRAINING RESULTS ------------------------------------------------------------------------------

# Load latest or best validation or final model
# conf.checkpoint = "latest"
# conf.checkpoint = "best_val"
# conf.checkpoint = "best_trn"
conf.checkpoint = "final"  # train and validation set trained model

log.info(f"Loading model from checkpoint ({conf.checkpoint}) for evaluation")

# resume from previous check point or resume from best validaton score checkpoint
# load model state
log.info(f"loading model from {conf.model_path}")
model_state_dict = torch.load(f"{conf.model_path}model_{conf.checkpoint}.pth", map_location=conf.device.type)
model.load_state_dict(model_state_dict)

if FREQ == '1H':
    step = 24
    max_steps = 14
elif FREQ == '24H':
    step = 7
    max_steps = 6
elif FREQ == '168H':
    step = 4  # 52
    max_steps = 2

from sklearn.metrics import mean_absolute_error


class EWM:
    def __init__(self, series=None):
        self.alpha = .9
        self.options = np.arange(0.001, 0.999, 0.001)

        if series is not None:
            self.fit(series)

    def __call__(self, series):
        return self.predict(series)

    def fit(self, series):
        losses = []
        for alpha in self.options:
            self.alpha = alpha
            pred = self.predict(series)
            loss = mean_absolute_error(series, pred)
            losses.append(loss)

        self.alpha = self.options[np.argmin(losses)]

    #         print(f"alpha => {self.alpha}")

    def predict(self, series):
        series = np.pad(series, (1, 0), 'edge')

        pred = pd.Series(series).ewm(alpha=self.alpha).mean().values[:-1]
        return pred


# TODO: REFACTOR function: compare_time_series_metrics
# TODO: save trained baseline models and trained metrics to own folder...
# TODO: load baseline models...save evaluation metrics to own folder...
# TODO: all models  have the same length of test set,
# TODO: have separate function that actually compares the results.

from pandas import DataFrame


def compare_time_series_metrics(
        y_true,
        y_score,
        t_range,
        feature_names,
        is_training_set,
        step=24,
        max_steps=29,
        alpha=0.5,
        rangeslider_visible=True,
):
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
        kwargs[f"{feat_name}_y_ha"] = historic_average(y_true[:, i], step=step,
                                                       max_steps=max_steps)[offset - 1:-1]

        ewm = EWM(y_true[:, i])
        kwargs[f"{feat_name}_y_ewm"] = ewm(y_true[:, i])[offset:]

        feature_results[feat_name] = {
            "Ground Truth": y_true[offset:, i],
            "GRUFNN": y_score[offset:, i],
            f"HA({step},{max_steps})": historic_average(y_true[:, i],
                                                        step=step, max_steps=max_steps)[offset - 1:-1],
            f"EWM({ewm.alpha:.3f})": ewm(y_true[:, i])[offset:],
        }

        model_name_dict[feat_name] = {
            "y_score": "GRUFNN",
            "y_true": "Ground Truth",
            "y_ha": f"HA({step},{max_steps})",
            "y_ewm": f"EWM({ewm.alpha:.3f})",
        }

        fig = plot_time_signals(
            t_range=t_range[offset:],
            alpha=alpha,
            title=f'{freq_title} {feat_name.title()} Predicted Normalised City Counts {training_str}',
            ylabel='Normalised Counts [0,1]',
            xlabel='Date',
            rangeslider_visible=rangeslider_visible,
            **feature_results[feat_name]
        )
        file_name = f"{save_folder}{FREQ}_predictions_{feat_name}_" + \
                    f"{training_str.lower().replace(' ', '_').replace(')', '').replace('(', '')}.png"
        logging.info(f"Saving Prection Plots in: {file_name}")
        fig.write_image(file_name)
        fig.show()

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
    #                       title=f'{freq_title} Predicted Normalised City Counts',
    #                       yaxis_title='Normalised Counts [0,1]',**kwargs).show()

    metrics = DataFrame(ll)
    metrics.sort_values(['Crime Type', 'MASE'], inplace=True)
    metrics.reset_index(inplace=True, drop='index')

    return metrics


from utils.utils import write_txt

metrics_folder = f"./data/processed/{data_sub_path}/metrics/"
os.makedirs(metrics_folder, exist_ok=True)

## TRAINING EVALUATION -------------------------------------------------------------------------
trn_y_true, trn_y_score = evaluate_sequence_model(model, loaders.train_loader, conf)

# trn_y_score = scaler.inverse_transform(trn_y_score)
# trn_y_true = scaler.inverse_transform(trn_y_true)

trn_metrics = compare_time_series_metrics(
    y_true=trn_y_true,
    y_score=trn_y_score,
    t_range=loaders.train_loader.dataset.t_range[-len(trn_y_true):],
    feature_names=list(total_df.columns[:trn_y_true.shape[1]]),  # feature_names[:trn_y_true.shape[1]],
    step=step,
    max_steps=max_steps,
    is_training_set=True,
    rangeslider_visible=False,
)
display(trn_metrics)

trn_metrics_latex = trn_metrics.round(decimals=3).to_latex(
    index=False,
    caption=f"{freq_title} Crime Count Forecasting Metrics (Training Data)",
    label="tab:daily-crime-count-metrics",
)

pd.to_pickle(trn_metrics, f"{metrics_folder}{FREQ}_metrics_train.pkl")
write_txt(trn_metrics_latex, f"{metrics_folder}{FREQ}_metrics_train_latex.txt")

## TEST SET EVALUATION ---------------------------------------------------------------------------------------

tst_y_true, tst_y_score = evaluate_sequence_model(model, loaders.test_loader, conf)
# tst_y_true, tst_y_score = tst_y_true[:,0], tst_y_score[:,0]

tst_metrics = compare_time_series_metrics(
    y_true=tst_y_true,
    y_score=tst_y_score,
    t_range=loaders.test_loader.dataset.t_range[-len(tst_y_true):],
    feature_names=list(total_df.columns[:tst_y_true.shape[1]]),
    is_training_set=False,
    step=step,
    max_steps=max_steps,
    rangeslider_visible=False,
)

display(tst_metrics)

tst_metrics_latex = tst_metrics.round(decimals=3).to_late
x(
    index=False,
    caption=f"{freq_title} Crime Count Forecasting Metrics (Test Data)",
    label="tab:daily-crime-count-metrics",
)

pd.to_pickle(tst_metrics, f"{metrics_folder}{FREQ}_metrics_test.pkl")
write_txt(tst_metrics_latex, f"{metrics_folder}{FREQ}_metrics_test.txt")
