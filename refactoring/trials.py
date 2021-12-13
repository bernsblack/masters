import logging
from typing import Dict

import numpy as np
import torch
from torch import nn

from datasets.sequence_dataset import SequenceDataLoaders
from models.rnn_models import GRUFNN
from models.sequence_models import train_epoch_for_sequence_model, evaluate_sequence_model
from trainers.generic_trainer import train_model_final, train_model
from utils import set_system_seed
from utils.configs import BaseConf
from time import time

from utils.forecasting import forecast_metrics

"""
trail.py is used to run functions multiple times with different seeds to get better estimates on model performances.
It allows us to estimate the expected value and gauge the variance of the results
"""


# # default values to set when not using hyper-parameter optimization
# hyper_parameters = {
#     'lr': 1e-3,
#     'weight_decay': 1e-6,
#     'hidden_size': 8,
#     'num_layers': 1,
#     'seq_len': 28,
# }


def run_trials_for_grufnn(
        conf: BaseConf,
        hyper_parameters: Dict,
        model: GRUFNN,
        loaders: SequenceDataLoaders,
        num_trials: int = 10,
):
    """
    Run a full experiment on GRUFNN model multiple times with different seeds, to determine if the variability is
    due to the hyper parameters or the initial seed that sets the parameters of the model.
    Data and hyper parameters must be set before hand.

    Used to:
        1. Train model with a validation set once to determine the num_epochs
        2. Run trials loops:
            2.a. Reset seed
            2.b. Re-setup model
            2.c. Train with train_val set for predetermined num_epochs
            2.d. Run evaluations on the model
        3. Return a dataframe with seed and forecast metrics for each trial run
    """

    trial_metrics_list = []

    conf.early_stopping = True
    conf.patience = 30
    conf.min_epochs = 1
    conf.max_epochs = 10_000

    conf.lr = hyper_parameters['lr']
    conf.weight_decay = hyper_parameters['weight_decay']
    conf.hidden_size = hyper_parameters['hidden_size']
    conf.num_layers = hyper_parameters['num_layers']
    conf.seq_len = hyper_parameters['seq_len']

    # ====================================== can be put into loop as well
    # model setup
    conf.seed = int(time())  # unique seed for each run
    set_system_seed(conf.seed)  # should be reset with each model instantiation

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

    logging.info(
        f"best validation: {np.min(val_epoch_losses):.6f} @ epoch: {np.argmin(val_epoch_losses) + 1}"
    )

    # full train-val dataset training
    conf.max_epochs = np.argmin(val_epoch_losses) + 1  # because of index starting at zero
    # ====================================== can be put into loop as well

    for i in range(num_trials):
        logging.info(f"Starting trial {i + 1} of {num_trials}.")

        conf.seed = int(time() + 10 * i)  # unique seed for each run
        set_system_seed(conf.seed)  # should be reset with each model instantiation
        model.reset_parameters()

        optimiser = torch.optim.AdamW(
            params=model.parameters(),
            lr=conf.lr,
            weight_decay=conf.weight_decay,
        )

        trn_val_epoch_losses = train_model_final(
            model=model,
            optimiser=optimiser,
            loaders=loaders,
            train_epoch_fn=train_epoch_for_sequence_model,
            loss_fn=criterion,
            conf=conf,
        )

        tst_y_true_trial, tst_y_score_trial = evaluate_sequence_model(
            model=model,
            batch_loader=loaders.test_loader,
            conf=conf,
        )
        trial_metrics = forecast_metrics(y_true=tst_y_true_trial, y_score=tst_y_score_trial)
        trial_metrics['Seed'] = conf.seed
        trial_metrics_list.append(trial_metrics)

        logging.info(f"Completed trial {i + 1} of {num_trials}.")

    return trial_metrics_list
