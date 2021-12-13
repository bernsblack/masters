import logging
from pprint import pformat
from time import time

import numpy as np
import torch
from torch import nn

from datasets.sequence_dataset import SequenceDataLoaders
from models.rnn_models import GRUFNN
from models.sequence_models import evaluate_sequence_model, train_epoch_for_sequence_model
from trainers.generic_trainer import train_model
from utils import set_system_seed
from utils.configs import BaseConf
from utils.forecasting import forecast_metrics
from utils.types import TParameterization


def new_evaluate_hyper_parameters_fn(
        conf: BaseConf,
        input_size,
        output_size,
        input_data,
        target_data,
        t_range,
        test_size,
        overlap_sequences,
):
    def evaluate_hyper_parameters_fn(hyper_parameters: TParameterization):  # todo better train loop eval loop
        """
        Trains GRUFNN model on training set and evaluate the hyper parameters on the validation set

        :param hyper_parameters:
        :return: Mean Absolute Scaled Error (MASE) on Validation Set given hyper_parameters
        """
        logging.info(f"Running HyperOpt Trial with: {pformat(hyper_parameters)}")

        # hyper param setup
        conf.lr = hyper_parameters.get('lr', 1e-3)
        conf.weight_decay = hyper_parameters.get('weight_decay', 1e-4)
        conf.hidden_size = int(hyper_parameters.get('hidden_size', 50))
        conf.num_layers = int(hyper_parameters.get('num_layers', 5))
        default_seq_len = {
            "24H": 90,
            "1H": 168,
            "168H": 52,
        }.get(conf.freq, 60)
        conf.seq_len = int(hyper_parameters.get('seq_len', default_seq_len))

        conf.early_stopping = True
        conf.patience = 30
        conf.min_epochs = 1
        conf.max_epochs = 10_000

        conf.seed = np.random.randint(10_000) * (int(time()) % conf.hidden_size + conf.num_layers)
        set_system_seed(conf.seed)  # should be reset with each run
        # loaders and model are dependent on hyper_parameters and must be created on every execution of this fn
        loaders = SequenceDataLoaders(  # setup data loader 2: hyper opt
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
        ),

        model = GRUFNN(
            input_size=input_size,
            hidden_size0=conf.hidden_size,
            hidden_size1=conf.hidden_size // 2,
            output_size=output_size,
            num_layers=conf.num_layers,
        ).to(conf.device)

        criterion = nn.MSELoss()
        optimiser = torch.optim.AdamW(params=model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)

        # training
        trn_epoch_losses, val_epoch_losses, stopped_early = train_model(
            model=model,
            optimiser=optimiser,
            loaders=loaders,  # trains on training data and stops when training and validation scores start splitting
            train_epoch_fn=train_epoch_for_sequence_model,
            loss_fn=criterion,
            conf=conf,
            verbose=False,
        )

        # Load best validation model
        conf.checkpoint = "best_val"
        logging.info(f"Loading model from checkpoint ({conf.checkpoint}) for evaluation")
        logging.info(f"Loading model from {conf.model_path}")
        model.load_state_dict(
            state_dict=torch.load(
                f=f"{conf.model_path}model_{conf.checkpoint}.pth",
                map_location=conf.device.type),
        )

        # Evaluation best validation model on validation set
        val_y_true, val_y_score = evaluate_sequence_model(
            model=model,
            batch_loader=loaders.validation_loader,
            conf=conf,
        )

        return forecast_metrics(
            y_true=val_y_true,
            y_score=val_y_score,
        )['MASE']

    return evaluate_hyper_parameters_fn
