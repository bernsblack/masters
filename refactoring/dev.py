#!/home/bernard/anaconda3/envs/masters/bin/python

# re-writing the sequence training and evaluation notebook (Development - Whole city crime count predictor (Daily))
# NOTE THAT THIS FILE LEAVES OUT DATA EXPLORATION AND EXPERIMENTATION AND MAINLY FOCUSES ON THE TRAINING AND EVALUATION
# OF THE DL AND BASE-LINE MODELS - OUR MAIN ISSUE IS THAT THE TEST SET SIZES ARE NOT THE SAME FOR ALL MODELS,
# DUE TO DIFFERENT INPUT LENGTHS FOR THE VARIOUS MODELS. THE GOALS OF THE REWRITE IS TO GUARANTEE THAT THE TEST SET
# OUTPUTS ARE THE SAME LENGTHS REGARDLESS OF THE INPUTS LENGTHS. THIS MEANS ALL MODELS WILL BE COMPARED UNDER THE SAME
# CIRCUMSTANCES
# IMPORTS --------------------------------------------------------------------------------------------------

import logging
import logging as log
import os
from copy import deepcopy
from pprint import pformat
from time import strftime
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import torch
from IPython.core.display import display as display_inner
from ax.service.managed_loop import optimize as optimize_hyper_parameters
from matplotlib import rcParams
from torch import nn

from datasets.sequence_dataset import SequenceDataLoaders
from logger.logger import setup_logging
from models.rnn_models import GRUFNN
from models.sequence_models import train_epoch_for_sequence_model, evaluate_sequence_model
from refactoring.compare_models import compare_time_series_metrics
from refactoring.hyper_optimise_sequence import new_evaluate_hyper_parameters_fn
from refactoring.trials import run_trials_for_grufnn
from trainers.generic_trainer import train_model
from trainers.generic_trainer import train_model_final
from utils.configs import BaseConf
from constants.crime_types import NOT_TOTAL
from utils.data_processing import encode_time_vectors
from utils.interactive import plot_interactive_epoch_losses
from utils.plots import plot, subplots_df
from utils.plots import plot_corr
from utils.utils import (write_json, get_data_sub_paths,
                         load_total_counts, set_system_seed)
from utils.utils import write_txt
from utils.whole_city import (plot_time_series_anomalies_wc, plot_mi_curves_wc,
                              normalize_periodically_wc, plot_time_vectors)

pio.templates.default = "plotly_white"  # "none"

# todo split file into data part and model part...
if __name__ == '__main__':
    do_display = True
    if do_display:
        display = display_inner
    else:
        def return_none(x):
            return None


        display = return_none

    os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())
    rcParams["font.family"] = "STIXGeneral"
    # START CODE --------------------------------------------------------------------------------------------------

    conf = BaseConf()

    data_sub_paths = [i for i in get_data_sub_paths() if i.startswith('Totals')]
    logging.info(data_sub_paths)

    #  if train/test sets should overlap by the sequence length of i put vector
    #  only of concern if the loss functions in the training loop use the whole sequence outputs to calculate the loss
    #  - which in turn leads to quicker train times.
    overlap_sequences = True

    data_sub_path = data_sub_paths[0]
    conf.freq = data_sub_path.lstrip('Totals_T').split('_')[0]
    conf.time_steps_per_day = 24 / int(conf.freq[:-1])

    conf.freq_title = {
        "24H": "Daily",
        "1H": "Hourly",
        "168H": "Weekly",
    }.get(conf.freq, "Hourly")
    logging.info(f"Using: {conf.freq_title} ({conf.freq})")

    df = load_total_counts(folder_name=data_sub_path)

    if conf.freq == "168H":
        # data is split into weeks starting on mondays - we need to trim first
        # and last weeks because they are technically half weeks because they don't start on mondays
        df = df.iloc[1:-1]

    display(df)
    subplots_df(df[NOT_TOTAL], title='Crime Counts Over Time', xlabel='Date', ylabel='Count', ncols=2).show()

    if torch.cuda.is_available():
        logging.info(torch.cuda.memory_summary())
    else:
        raise Exception("CUDA is not available")

    # CONFIG AND LOGGING SETUP ----------------------------------------------------------------------------------------
    # manually set
    conf.seed = int(time())  # 3
    set_system_seed(conf.seed)
    conf.model_name = f"GRUFNN ({conf.freq_title})"
    conf.data_path = f"./data/processed/{data_sub_path}/"

    if not os.path.exists(conf.data_path):
        raise Exception(f"Directory ({conf.data_path}) needs to exist.")

    conf.model_path = f"{conf.data_path}models/{conf.model_name}/"
    conf.plots_path = f"./data/processed/{data_sub_path}/plots/"
    os.makedirs(conf.plots_path, exist_ok=True)
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

    # SET THE hyperparameterS
    conf.shaper_top_k = -1
    conf.use_classification = False
    conf.train_set_first = True
    conf.use_crime_types = False

    # conf.seq_len = 60 # int(14*conf.time_steps_per_day) + 1
    conf.seq_len = {
        "24H": 90,
        "1H": 168,
        "168H": 52,
    }.get(conf.freq, 60)

    conf.batch_size = 128

    # ANOMALY DETECTION PLOTS ----------------------------------------------------------------------------------------

    do_plot_anomalies = False
    if do_plot_anomalies:
        plot_time_series_anomalies_wc(conf=conf, df=df)

    # MUTUAL INFORMATION PLOTS -----------------------------------------------------------------------------------------

    do_plot_mi_curves = False
    if do_plot_mi_curves:
        plot_mi_curves_wc(conf=conf, df=df)

    """
    ## Periodic Rolling Normalisation ---------------------------------------------------------------------------
    By applying rolling normalisation we can control for the period nature of our signals. We subtract the signal by a 
    periodic rolling mean and divide it by a period rolling standard deviation. We can also perform this on various 
    periods e.g. 7 days and 365 days, which caters for weekly and yearly periodic cycles. By controlling for the signals 
    periodic nature we have a new signal showing how many standard deviations the current time step is from the expected
    value for that day/week/season. The models are then left to predict residuals of our crime signals.
    """

    normalize_periodically = True
    if not normalize_periodically:
        normed_df = df.copy()
    else:
        normed_df = normalize_periodically_wc(conf=conf, df=df, norm_offset=0, corr_subplots=False)

    # total_df = normalize_df_min_max(normed_df) # rescale between 0 and 1 again...is this really needed?
    # total_df = normalize_df_mean_std(normed_df) # rescale between 0 and 1 again...is this really needed?
    total_df = normed_df
    # total_df = df
    # total_df = normalize_df_min_max(df)
    # total_df = normalize_df_mean_std(df)

    # plot crime count distributions of z-score normed data
    do_plot_crime_count_distributions = False
    if do_plot_crime_count_distributions:
        total_df.plot(kind='kde', alpha=0.4, title='Normalised Crime Counts Distributions')
        fig = plt.gcf()
        fig.set_size_inches(9, 6)
        plt.grid()
        plt.savefig(f"{conf.plots_path}{conf.freq}_crime_counts_distribution_normed.png")
        plt.show()
        display(total_df)

    # plot normed crime correlations (pearson/spearman/partial)
    do_plot_correlation = False
    if do_plot_correlation:
        for temp, _normed, period_string in zip([df, total_df], ['', '_normed'], ['', ' after Normalisation']):
            corr = temp.loc[:, temp.columns != 'Total'].corr()
            plot_corr(corr=corr,
                      title=f'Pearson Correlation Coefficients Between {conf.freq_title} Crime Types{period_string}\n')
            plt.savefig(f"{conf.plots_path}{conf.freq}_corr_matrix_pearson{_normed}.png")
            # --
            corr = temp.loc[:, temp.columns != 'Total'].pcorr()
            plot_corr(corr=corr, title=f'Partial Pearson Correlation Coefficients ' +
                                       f'Between {conf.freq_title} Crime Types{period_string}\n')
            plt.savefig(f"{conf.plots_path}{conf.freq}_corr_matrix_pearson_partial{_normed}.png")
            # --
            corr = temp.loc[:, temp.columns != 'Total'].corr(method='spearman')
            plot_corr(corr=corr,
                      title=f'Spearman Correlation Coefficients Between {conf.freq_title} Crime Types{period_string}\n')
            plt.savefig(f"{conf.plots_path}{conf.freq}_corr_matrix_spearman{_normed}.png")

    # temporal variables setup
    t_range = total_df.index[1:]  # should be equal to the target times
    total_crime_types = total_df.values

    time_vectors = encode_time_vectors(t_range)
    # todo: (check notebooks first) calculate auto-correlation of errors signal of regressions between time vectors and
    #  crime count to see if we have information in the signal outside of the time factor of it

    # plot time vectors
    do_plot_time_vectors = False
    if do_plot_time_vectors:
        plot_time_vectors(conf=conf, t_range=t_range, time_vectors=time_vectors)
    # SETUP DATA LOADERS ------------------------

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
    }.get(conf.freq)

    # tst_ratio = test_size / len(input_data)

    loaders = SequenceDataLoaders(  # setup data loader 1
        input_data=input_data,
        target_data=target_data,
        t_range=t_range,
        batch_size=conf.batch_size,
        seq_len=conf.seq_len,
        shuffle=conf.shuffle,
        num_workers=0,
        val_ratio=0.3,
        tst_size=test_size,
        overlap_sequences=overlap_sequences,
    )

    logging.info(f"input_data.shape => {input_data.shape}")
    logging.info(f"target_data.shape => {target_data.shape}")

    # hyperparameter OPTIMISATION ---------------------------------------------------------------

    evaluate_hyper_parameters_fn = new_evaluate_hyper_parameters_fn(
        conf=conf,
        input_size=input_size,
        output_size=output_size,
        input_data=input_data,
        target_data=target_data,
        t_range=t_range,
        test_size=test_size,
        overlap_sequences=overlap_sequences,
    )

    total_hyper_opt_trials = 20  # 20
    do_optimize_hyper_params = True
    if do_optimize_hyper_params:
        # Hyper Opt Training
        # PARAM_CLASSES = ["range", "choice", "fixed"]
        # PARAM_TYPES = {"int": int, "float": float, "bool": bool, "str": str}
        # potential parameters
        max_seq_len = {
            "24H": 366,
            "1H": 24 * 7 * 6,
            "168H": 64,
        }.get(conf.freq, 60)

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

        logging.info(f"""hyperparameters and their bounds:
    {pformat(hyper_params)}""")

        best_parameters, values, experiment, hyper_model = optimize_hyper_parameters(
            parameters=hyper_params,
            evaluation_function=evaluate_hyper_parameters_fn,
            objective_name='MASE',
            minimize=True,  # change when the objective is a loss
            total_trials=total_hyper_opt_trials,
            #     random_seed=1, # optional
        )

        experiment_data = experiment.fetch_data()
        display((experiment_data.df,))
        display((pd.DataFrame([best_parameters]),))

    #     render(plot_contour(model=hyper_model, param_x='lr', param_y='weight_decay', metric_name='MASE'))
    else:
        # default values to set when not using hyperparameter optimization
        best_parameters = {
            'lr': 1e-3,
            'weight_decay': 1e-6,
            'hidden_size': 8,
            'num_layers': 1,
            'seq_len': 28,
        }

    write_json(best_parameters, f"{conf.plots_path}{conf.freq}_best_hyper_params.json".replace(' ', '_'))

    display(pd.DataFrame([best_parameters]).iloc[0])

    # MULTI-RUN EVALUATION -------------------------------------------------------------------------

    NUM_TRIALS = 10
    VAL_RATIO = 0.3

    # same model and loaders are used for multiple runs with different seeds
    # to test the influence of the seed on performance
    trial_results = run_trials_for_grufnn(
        conf=conf,
        hyper_parameters=best_parameters,
        model=GRUFNN(
            input_size=input_size,
            hidden_size0=conf.hidden_size,
            hidden_size1=conf.hidden_size // 2,
            output_size=output_size,
            num_layers=conf.num_layers,
        ).to(conf.device),
        loaders=SequenceDataLoaders(  # setup data loader 3: run trial
            input_data=input_data,
            target_data=target_data,
            t_range=t_range,
            batch_size=conf.batch_size,
            seq_len=conf.seq_len,
            shuffle=conf.shuffle,
            num_workers=0,
            val_ratio=VAL_RATIO,
            tst_size=test_size,
            overlap_sequences=overlap_sequences,
        ),
        num_trials=NUM_TRIALS,

    )

    display(pd.DataFrame(trial_results))

    pd.DataFrame(trial_results)[['MASE']].plot(
        kind='box',
        title=f"{conf.freq_title} Crime GRUFNN MASE Score for {NUM_TRIALS} Seeds\n",
    )
    plt.savefig(f"{conf.plots_path}{conf.freq}_result_seed_consistency.png")
    plt.show()

    # SETUP MODELS WITH hyperparameterS FROM PREVIOUS STEP -------------------------------------------

    # set hyper params-> set seed -> set model -> set optimiser
    conf.seed = int(time())  # 1607355910

    conf.lr = best_parameters['lr']
    conf.weight_decay = best_parameters['weight_decay']
    conf.hidden_size = best_parameters['hidden_size']
    conf.num_layers = best_parameters['num_layers']
    conf.seq_len = best_parameters['seq_len']

    set_system_seed(conf.seed)  # should be reset with each model instantiation
    model = GRUFNN(
        input_size=input_size,
        hidden_size0=conf.hidden_size,
        hidden_size1=conf.hidden_size // 2,
        output_size=output_size,
        num_layers=conf.num_layers,
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

    fig = plot_interactive_epoch_losses(trn_epoch_losses, val_epoch_losses)
    fig.write_image(f"{conf.plots_path}{conf.freq}_loss_val_plot.png")
    fig.show()

    # set max epochs -> reset to same seed as before -> recreate model -> recreate optimiser -> train final model

    conf.max_epochs = np.argmin(val_epoch_losses) + 1  # because of index starting at zero

    set_system_seed(conf.seed)  # should be reset with each model instantiation
    model = GRUFNN(
        input_size=input_size,
        hidden_size0=conf.hidden_size,
        hidden_size1=conf.hidden_size // 2,
        output_size=output_size,
        num_layers=conf.num_layers,
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

    # TRAINING RESULTS ------------------------------------------------------------------------------

    # Load latest or best validation or final model
    # conf.checkpoint = "latest"
    # conf.checkpoint = "best_val"
    # conf.checkpoint = "best_trn"
    conf.checkpoint = "final"  # train and validation set trained model

    log.info(f"Loading model from checkpoint ({conf.checkpoint}) for evaluation")

    # resume from previous check point or resume from best validation score checkpoint
    # load model state
    log.info(f"loading model from {conf.model_path}")
    model_state_dict = torch.load(f"{conf.model_path}model_{conf.checkpoint}.pth", map_location=conf.device.type)
    model.load_state_dict(model_state_dict)

    step, max_steps = {
        "1H": (24, 14),
        "24H": (7, 6),
        "168H": (4, 2),
    }.get(conf.freq, (24, 14))

    metrics_folder = f"./data/processed/{data_sub_path}/metrics/"
    os.makedirs(metrics_folder, exist_ok=True)

    # TRAINING EVALUATION -------------------------------------------------------------------------
    trn_y_true, trn_y_score = evaluate_sequence_model(model, loaders.train_loader, conf)

    # trn_y_score = scaler.inverse_transform(trn_y_score)
    # trn_y_true = scaler.inverse_transform(trn_y_true)

    trn_metrics = compare_time_series_metrics(
        conf=conf,
        y_true=trn_y_true,
        y_score=trn_y_score,
        t_range=loaders.train_loader.dataset.t_range[-len(trn_y_true):],
        feature_names=list(total_df.columns[:trn_y_true.shape[1]]),  # feature_names[:trn_y_true.shape[1]],
        step=step,
        max_steps=max_steps,
        is_training_set=True,
        range_slider_visible=False,
    )
    display((trn_metrics,))

    # save latex table code to txt file
    trn_metrics_latex = trn_metrics.round(decimals=3).to_latex(
        index=False,
        caption=f"{conf.freq_title} Crime Count Forecasting Metrics (Training Data)",
        label=f"tab:{conf.freq_title.lower()}-crime-count-metrics",
    )
    pd.to_pickle(trn_metrics, f"{metrics_folder}{conf.freq}_metrics_train.pkl")
    write_txt(trn_metrics_latex, f"{metrics_folder}{conf.freq}_metrics_train_latex.txt")

    # TEST SET EVALUATION ---------------------------------------------------------------------------------------

    tst_y_true, tst_y_score = evaluate_sequence_model(model, loaders.test_loader, conf)
    # tst_y_true, tst_y_score = tst_y_true[:,0], tst_y_score[:,0]

    tst_metrics = compare_time_series_metrics(
        conf=conf,
        y_true=tst_y_true,
        y_score=tst_y_score,
        t_range=loaders.test_loader.dataset.t_range[-len(tst_y_true):],
        feature_names=list(total_df.columns[:tst_y_true.shape[1]]),
        is_training_set=False,
        step=step,
        max_steps=max_steps,
        range_slider_visible=False,
    )

    display((tst_metrics,))

    tst_metrics_latex = tst_metrics.round(decimals=3).to_latex(
        index=False,
        caption=f"{conf.freq_title} Crime Count Forecasting Metrics (Test Data)",
        label=f"tab:{conf.freq_title.lower()}-crime-count-metrics",
    )

    pd.to_pickle(tst_metrics, f"{metrics_folder}{conf.freq}_metrics_test.pkl")
    write_txt(tst_metrics_latex, f"{metrics_folder}{conf.freq}_metrics_test.txt")
