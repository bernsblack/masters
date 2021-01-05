import torch
import logging as log
import numpy as np

from dataloaders.grid_loader import BaseDataLoaders
from utils.configs import BaseConf
from utils.metrics import LossPlotter, best_threshold, get_y_pred


def save_checkpoint(tag, model, optimiser, conf,
                    val_batch_losses,
                    val_epoch_losses,
                    trn_epoch_losses,
                    trn_batch_losses):
    # trn_val_epoch_losses):
    torch.save(model.state_dict(), f"{conf.model_path}model_{tag}.pth")
    torch.save(optimiser.state_dict(), f"{conf.model_path}optimiser_{tag}.pth")
    np.savez_compressed(file=f"{conf.model_path}losses_{tag}.npz",
                        trn_batch_losses=trn_batch_losses,
                        val_batch_losses=val_batch_losses,
                        trn_epoch_losses=trn_epoch_losses,
                        val_epoch_losses=val_epoch_losses)
    # trn_val_epoch_losses=trn_val_epoch_losses)


# generic training loop
def train_model(model, optimiser, loaders, train_epoch_fn, loss_fn, conf, scheduler=None, verbose=True):
    """
    train_model is a generic function used to train a variety of models

    :param model: torch model
    :param optimiser: torch optimizer
    :param loaders: batch loader class containing train, validation and test batch loaders
    :param train_epoch_fn: function used to train model for epoch
    :param loss_fn: loss function used in training process, can be CrossEntropy, MSE or MAE depending on model
    :param conf: configure object containing all preferences for model, optimisations and training loops
    :param scheduler: learning rate scheduler used during training loop
    :param verbose: if the losses should be logged when training
    :return: trn_epoch_losses, val_epoch_losses, stopped_early

    Notes
    -----
    train_model will log all information to model folder as well as produce training and validation loss curve plots
    """
    log.info(f"\n ====================== Training {conf.model_name} ====================== \n")
    log.info(f"\n ====================== Config Values ====================== \n{conf}" +
             "\n ====================== Config Values ====================== \n")

    stopped_early = False
    trn_batch_losses = []
    val_batch_losses = []

    trn_epoch_losses = []
    trn_epoch_losses_best = float("inf")
    val_epoch_losses = []
    val_epoch_losses_best = float("inf")
    # trn_val_epoch_losses = []
    # trn_val_epoch_losses_best = float("inf")

    if conf.resume:
        try:
            # load losses
            losses_zip = np.load(f"{conf.model_path}losses_{conf.checkpoint}.npz")

            val_batch_losses = losses_zip["val_batch_losses"].tolist()
            trn_batch_losses = losses_zip["trn_batch_losses"].tolist()

            trn_epoch_losses = losses_zip["trn_epoch_losses"].tolist()
            trn_epoch_losses_best = np.min(trn_epoch_losses)
            val_epoch_losses = losses_zip["val_epoch_losses"].tolist()
            val_epoch_losses_best = np.min(val_epoch_losses)
            # trn_val_epoch_losses = losses_zip["trn_val_epoch_losses"].tolist()
            # trn_val_epoch_losses_best = np.min(trn_val_epoch_losses)
        except Exception as e:
            log.error(f"Nothing to resume from, training from scratch \n\t-> {e}")

    log.info(f"Start Training {conf.model_name}")
    log.info(f"Using optimiser: \n{optimiser}\n\n")

    prev_best_val_step = 0
    for epoch in range(conf.max_epochs):
        if verbose:
            log.info(f"Epoch: {(1 + epoch):04d}/{conf.max_epochs:04d}")
        else:
            if epoch % 10 == 0:
                log.info(f"Epoch: {(1 + epoch):04d}/{conf.max_epochs:04d}")

        conf.timer.reset()
        # Training loop
        model.train()
        epoch_loss = train_epoch_fn(model=model,
                                    optimiser=optimiser,
                                    batch_loader=loaders.train_loader,
                                    loss_fn=loss_fn,
                                    total_losses=trn_batch_losses,
                                    conf=conf)

        trn_epoch_losses.append(epoch_loss)
        log.debug(f"Epoch {epoch} -> Training Loop Duration: {conf.timer.check()}")
        conf.timer.reset()

        # Validation loop
        tmp_val_epoch_losses = []
        with torch.set_grad_enabled(False):
            model.eval()
            epoch_loss = train_epoch_fn(model=model,
                                        optimiser=optimiser,
                                        batch_loader=loaders.validation_loader,
                                        loss_fn=loss_fn,
                                        total_losses=val_batch_losses,
                                        conf=conf)

            val_epoch_losses.append(epoch_loss)
            log.debug(f"Epoch {epoch} -> Validation Loop Duration: {conf.timer.check()}")

            if scheduler:
                scheduler.step()
                # scheduler.step(epoch)  # alternative option
                # scheduler.step(epoch_loss)  # alternative option

        # trn_val_epoch_losses.append(trn_epoch_losses[-1] + val_epoch_losses[-1])

        # save best validation model
        prev_best_val_step += 1
        if val_epoch_losses[-1] < val_epoch_losses_best:
            prev_best_val_step = 0
            val_epoch_losses_best = val_epoch_losses[-1]
            save_checkpoint('best_val', model, optimiser, conf, val_batch_losses, val_epoch_losses,
                            trn_epoch_losses, trn_batch_losses)  # , trn_val_epoch_losses)

        if trn_epoch_losses[-1] < trn_epoch_losses_best:
            trn_epoch_losses_best = trn_epoch_losses[-1]
            save_checkpoint('best_trn', model, optimiser, conf, val_batch_losses, val_epoch_losses,
                            trn_epoch_losses, trn_batch_losses)  # , trn_val_epoch_losses)

        # if trn_val_epoch_losses[-1] < trn_val_epoch_losses_best:
        #     trn_val_epoch_losses_best = trn_val_epoch_losses[-1]
        #     save_checkpoint('best_trn_val', model, optimiser, conf, val_batch_losses, val_epoch_losses,
        #                     trn_epoch_losses, trn_batch_losses, trn_val_epoch_losses)

        if verbose:
            log.info(f"\tLoss (Trn): \t\t{trn_epoch_losses[-1]:.8f}")
            log.info(f"\tLoss (Best Trn): \t{trn_epoch_losses_best:.8f}")
            log.info(f"\tLoss (Val): \t\t{val_epoch_losses[-1]:.8f}")
            log.info(f"\tLoss (Best Val): \t{val_epoch_losses_best:.8f}")
            log.info(f"\tLoss (Trn Val Dif): \t{np.abs(val_epoch_losses[-1] - trn_epoch_losses[-1]):.8f}\n")

        if conf.early_stopping and prev_best_val_step > conf.patience and epoch >= conf.min_epochs:
            log.warning(
                f"Early stopping: Over-fitting has taken place. Previous validation improvement is more that {conf.patience} epochs ago.")
            stopped_early = True
            break

        if conf.early_stopping and epoch > 5 and val_epoch_losses[-1] == val_epoch_losses[-2] and \
                val_epoch_losses[-3] == val_epoch_losses[-2] and epoch >= conf.min_epochs:
            log.warning(f"Converged: Past 3 validation losses are all the same,"
                        + f" local optima has been reached")
            stopped_early = True
            break

        # checkpoint - save latest models and loss values
        save_checkpoint('latest', model, optimiser, conf, val_batch_losses, val_epoch_losses,
                        trn_epoch_losses, trn_batch_losses)  # , trn_val_epoch_losses)

    # Save training and validation plots - add flag to actually save or display
    skip = 0

    if str(loss_fn) == 'CrossEntropyLoss()':
        loss_plot_title = "Cross Entropy Loss"
    elif str(loss_fn) == 'MSELoss()':
        loss_plot_title = "MSE Loss"
    else:
        loss_plot_title = "Loss"

    # plot the whole plot
    loss_plotter = LossPlotter(title=f"{loss_plot_title} - {conf.model_name}")
    loss_plotter.plot_losses(trn_epoch_losses, trn_batch_losses[skip:], val_epoch_losses, val_batch_losses[skip:])
    loss_plotter.savefig(f"{conf.model_path}plot_train_val_epoch_losses.png")

    return trn_epoch_losses, val_epoch_losses, stopped_early


# generic training loop
def train_model_final(model, optimiser, loaders, train_epoch_fn, loss_fn, conf):
    """
    train model using the train_validation_set up until the specified conf.max_epochs: no early stopping is used

    :param model: torch model
    :param optimiser: torch optimizer
    :param loaders: batch loader class containing train, validation and test batch loaders
    :param train_epoch_fn: function used to train model for epoch
    :param loss_fn: loss function used in training process, can be CrossEntropy, MSE or MAE depending on model
    :param conf: configure object containing all preferences for model, optimisations and training loops
    :return: trn_val_epoch_losses

    Notes
    -----
    train_model will log all information to model folder as well as produce training and validation loss curve plots
    """
    log.info(f"\n ====================== Final Training {conf.model_name} ====================== \n")
    log.info(f"\n ====================== Config Values ====================== \n{conf}" +
             "\n ====================== Config Values ====================== \n")

    trn_val_batch_losses = []
    trn_val_epoch_losses = []

    log.info(f"Start Training {conf.model_name}")
    log.info(f"Using optimiser: \n{optimiser}\n\n")

    for epoch in range(conf.max_epochs):
        log.info(f"Epoch: {(1 + epoch):04d}/{conf.max_epochs:04d}")
        conf.timer.reset()
        # Training loop
        model.train()
        epoch_loss = train_epoch_fn(model=model,
                                    optimiser=optimiser,
                                    batch_loader=loaders.train_validation_loader,
                                    loss_fn=loss_fn,
                                    total_losses=trn_val_batch_losses,
                                    conf=conf)

        trn_val_epoch_losses.append(epoch_loss)
        log.debug(f"Epoch {epoch} -> Training Loop Duration: {conf.timer.check()}")
        conf.timer.reset()

        log.info(f"\tLoss (Trn Val): \t\t{trn_val_epoch_losses[-1]:.8f}")

    tag = "final"

    torch.save(model.state_dict(), f"{conf.model_path}model_{tag}.pth")
    torch.save(optimiser.state_dict(), f"{conf.model_path}optimiser_{tag}.pth")
    np.savez_compressed(file=f"{conf.model_path}losses_{tag}.npz",
                        trn_val_batch_losses=trn_val_batch_losses,
                        trn_val_epoch_losses=trn_val_epoch_losses)

    return trn_val_epoch_losses


# generic training loop
def evaluate_model_loss(model, optimiser, loaders, train_epoch_fn, loss_fn, conf, scheduler=None):
    """
    Will
    """
    log.info(f"\n ====================== Training {conf.model_name} ====================== \n")
    log.info(f"\n ====================== Config Values ====================== \n{conf}" +
             "\n ====================== Config Values ====================== \n")

    stopped_early = False
    trn_batch_losses = []
    val_batch_losses = []

    trn_epoch_losses = []
    trn_epoch_losses_best = float("inf")
    val_epoch_losses = []
    val_epoch_losses_best = float("inf")
    # trn_val_epoch_losses = []
    # trn_val_epoch_losses_best = float("inf")

    if conf.resume:
        try:
            # load losses
            losses_zip = np.load(f"{conf.model_path}losses_{conf.checkpoint}.npz")

            val_batch_losses = losses_zip["val_batch_losses"].tolist()
            trn_batch_losses = losses_zip["trn_batch_losses"].tolist()

            trn_epoch_losses = losses_zip["trn_epoch_losses"].tolist()
            trn_epoch_losses_best = np.min(trn_epoch_losses)
            val_epoch_losses = losses_zip["val_epoch_losses"].tolist()
            val_epoch_losses_best = np.min(val_epoch_losses)

            # trn_val_epoch_losses = losses_zip["trn_val_epoch_losses"].tolist()
            # trn_val_epoch_losses_best = np.min(trn_val_epoch_losses)
        except Exception as e:
            log.error(f"Nothing to resume from, training from scratch \n\t-> {e}")

    log.info(f"Start Training {conf.model_name}")
    log.info(f"Using optimiser: \n{optimiser}\n\n")

    prev_best_val_step = 0
    for epoch in range(conf.max_epochs):
        log.info(f"Epoch: {(1 + epoch):04d}/{conf.max_epochs:04d}")
        conf.timer.reset()
        # Training loop
        model.train()
        epoch_loss = train_epoch_fn(model=model,
                                    optimiser=optimiser,
                                    batch_loader=loaders.train_loader,
                                    loss_fn=loss_fn,
                                    total_losses=trn_batch_losses,
                                    conf=conf)

        trn_epoch_losses.append(epoch_loss)
        log.debug(f"Epoch {epoch} -> Training Loop Duration: {conf.timer.check()}")
        conf.timer.reset()

        # Validation loop
        tmp_val_epoch_losses = []
        with torch.set_grad_enabled(False):
            model.eval()
            epoch_loss = train_epoch_fn(model=model,
                                        optimiser=optimiser,
                                        batch_loader=loaders.validation_loader,
                                        loss_fn=loss_fn,
                                        total_losses=val_batch_losses,
                                        conf=conf)

            val_epoch_losses.append(epoch_loss)
            log.debug(f"Epoch {epoch} -> Validation Loop Duration: {conf.timer.check()}")

            if scheduler:
                scheduler.step()
                # scheduler.step(epoch)  # alternative option
                # scheduler.step(epoch_loss)  # alternative option

        # trn_val_epoch_losses.append(trn_epoch_losses[-1] + val_epoch_losses[-1])

        # save best validation model
        prev_best_val_step += 1
        if val_epoch_losses[-1] < val_epoch_losses_best:
            prev_best_val_step = 0
            val_epoch_losses_best = val_epoch_losses[-1]
            save_checkpoint('best_val', model, optimiser, conf, val_batch_losses, val_epoch_losses,
                            trn_epoch_losses, trn_batch_losses)  # , trn_val_epoch_losses)

        if trn_epoch_losses[-1] < trn_epoch_losses_best:
            trn_epoch_losses_best = trn_epoch_losses[-1]
            save_checkpoint('best_trn', model, optimiser, conf, val_batch_losses, val_epoch_losses,
                            trn_epoch_losses, trn_batch_losses)  # , trn_val_epoch_losses)

        # if trn_val_epoch_losses[-1] < trn_val_epoch_losses_best:
        #     trn_val_epoch_losses_best = trn_val_epoch_losses[-1]
        #     save_checkpoint('best_trn_val', model, optimiser, conf, val_batch_losses, val_epoch_losses,
        #                     trn_epoch_losses, trn_batch_losses, trn_val_epoch_losses)

        log.info(f"\tLoss (Trn): \t\t{trn_epoch_losses[-1]:.8f}")
        log.info(f"\tLoss (Best Trn): \t{trn_epoch_losses_best:.8f}")
        log.info(f"\tLoss (Val): \t\t{val_epoch_losses[-1]:.8f}")
        log.info(f"\tLoss (Best Val): \t{val_epoch_losses_best:.8f}")
        # log.info(f"\tLoss (Trn Val): \t{trn_val_epoch_losses[-1]:.8f}")
        # log.info(f"\tLoss (Best Trn Val): \t{trn_val_epoch_losses_best:.8f}")
        log.info(f"\tLoss (Trn Val Dif): \t{np.abs(val_epoch_losses[-1] - trn_epoch_losses[-1]):.8f}\n")

        if conf.early_stopping and prev_best_val_step > conf.patience and epoch >= conf.min_epochs:
            log.warning(
                f"Early stopping: Over-fitting has taken place. Previous validation improvement is more that {conf.patience} epochs ago.")
            stopped_early = True
            break

        if conf.early_stopping and epoch > 5 and val_epoch_losses[-1] == val_epoch_losses[-2] and \
                val_epoch_losses[-3] == val_epoch_losses[-2] and epoch >= conf.min_epochs:
            log.warning(f"Converged: Past 3 validation losses are all the same,"
                        + f" local optima has been reached")
            stopped_early = True
            break

        # checkpoint - save latest models and loss values
        save_checkpoint('latest', model, optimiser, conf, val_batch_losses, val_epoch_losses,
                        trn_epoch_losses, trn_batch_losses)  # , trn_val_epoch_losses)

    # Save training and validation plots - add flag to actually save or display
    skip = 0

    if str(loss_fn) == 'CrossEntropyLoss()':
        loss_plot_title = "Cross Entropy Loss"
    elif str(loss_fn) == 'MSELoss()':
        loss_plot_title = "MSE Loss"
    else:
        loss_plot_title = "Loss"

    # plot the whole plot
    loss_plotter = LossPlotter(title=f"{loss_plot_title} - {conf.model_name}")
    loss_plotter.plot_losses(trn_epoch_losses, trn_batch_losses[skip:], val_epoch_losses, val_batch_losses[skip:])
    loss_plotter.savefig(f"{conf.model_path}plot_train_val_epoch_losses.png")

    return trn_epoch_losses, val_epoch_losses, stopped_early


from typing import Callable
from torch import nn
from models.model_result import ModelMetrics, ModelResult


def evaluate_model(model: nn.Module, evaluate_fn: Callable, loaders: BaseDataLoaders, conf: BaseConf):
    """
    returns the model result on the testing data

    :param model:
    :param evaluate_fn:
    :param loaders:
    :param conf:
    :return:
    """
    trn_y_count, trn_y_class, trn_y_score, trn_t_range = evaluate_fn(model=model,
                                                                     batch_loader=loaders.train_loader,
                                                                     conf=conf)

    tst_y_count, tst_y_class, tst_y_score, tst_t_range = evaluate_fn(model=model,
                                                                     batch_loader=loaders.test_loader,
                                                                     conf=conf)

    thresh = best_threshold(trn_y_class, trn_y_score)
    # trn_y_pred = get_y_pred(thresh, trn_y_score)
    tst_y_pred = get_y_pred(thresh, tst_y_score)

    tst_y_count = loaders.data_group.to_counts(tst_y_count)
    tst_y_count = loaders.data_group.shaper.squeeze(tst_y_count)
    tst_y_score = loaders.data_group.shaper.squeeze(tst_y_score)
    tst_y_pred = loaders.data_group.shaper.squeeze(tst_y_pred)

    result = ModelResult(
        model_name=model.name,
        y_count=tst_y_count,
        y_pred=tst_y_pred,
        y_score=tst_y_score,
        t_range=tst_t_range,
        shaper=loaders.data_group.shaper,
        averaged_over_time=False,
    )

    return result
