import torch
import logging as log
import numpy as np
from utils.metrics import LossPlotter


# generic training loop
def train_model(model, optimiser, loaders, train_epoch_fn, loss_fn, conf, scheduler=None, patience=10):
    """
    Generic training loop that handles:
    - early stopping
    - timing
    - logging
    - model checkpoints
    - saving epoch and batch losses
    - scheduler: is used to systematically update the learning rate if a plateau is reached
        if scheduler is None it will be ignored. Scheduler mode should be set to 'min'
    - patience: number of epochs to continue where the validation loss has not improved before early stopping

    :returns: best validation loss of all the epochs - used to tune the hyper-parameters of the models/optimiser
    """
    log.info(f"\n ====================== Training {conf.model_name} ====================== \n")
    log.info(f"\n ====================== Config Values ====================== \n{conf}" +
             "\n ====================== Config Values ====================== \n")

    stopped_early = False
    trn_batch_losses = []
    val_batch_losses = []

    trn_epoch_losses = []
    val_epoch_losses = []
    val_epoch_losses_best = float("inf")

    if conf.resume:
        try:
            # load losses
            losses_zip = np.load(f"{conf.model_path}losses_{conf.checkpoint}.npz")

            val_batch_losses = losses_zip["val_batch_losses"].tolist()
            trn_batch_losses = losses_zip["trn_batch_losses"].tolist()

            trn_epoch_losses = losses_zip["trn_epoch_losses"].tolist()
            val_epoch_losses = losses_zip["val_epoch_losses"].tolist()
            val_epoch_losses_best = float(losses_zip["val_epoch_losses_best"])
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
                scheduler.step(epoch_loss)

        # save best model
        if val_epoch_losses[-1] < val_epoch_losses_best:
            prev_best_val_step = 0
            val_epoch_losses_best = val_epoch_losses[-1]
            torch.save(model.state_dict(), f"{conf.model_path}model_best.pth")
            torch.save(optimiser.state_dict(), f"{conf.model_path}optimiser_best.pth")
            np.savez_compressed(file=f"{conf.model_path}losses_best.npz",
                                val_batch_losses=val_batch_losses,
                                val_epoch_losses=val_epoch_losses,
                                trn_epoch_losses=trn_epoch_losses,
                                trn_batch_losses=trn_batch_losses,
                                val_epoch_losses_best=val_epoch_losses_best)
        else:
            prev_best_val_step += 1

        log.info(f"\tLoss (Trn): \t\t{trn_epoch_losses[-1]:.8f}")
        log.info(f"\tLoss (Val): \t\t{val_epoch_losses[-1]:.8f}")
        log.info(f"\tLoss (Val Best): \t{val_epoch_losses_best:.8f}")
        log.info(f"\tLoss (Dif): \t\t{np.abs(val_epoch_losses[-1] - trn_epoch_losses[-1]):.8f}\n")

        if conf.early_stopping and prev_best_val_step > patience:
            log.warning(f"Early stopping: Over-fitting has taken place. Previous validation improvement is more that {patience} steps ago.")
            stopped_early = True
            break

        # # increasing moving average of val_epoch_losses
        # if conf.early_stopping and epoch > 10 and np.sum(np.diff(val_epoch_losses[-5:])) >= 0:
        #     log.warning("Early stopping: Over-fitting has taken place - sum-differences between last 5 steps are greate than 0")
        #     stopped_early = True
        #     break
        #
        # if conf.early_stopping and epoch > 10 and np.abs(val_epoch_losses[-1] - val_epoch_losses[-2]) < conf.tolerance:
        #     log.warning(f"Converged: Difference between the past two"
        #                 + f" validation losses is within tolerance of {conf.tolerance}")
        #     stopped_early = True
        #     break

        if epoch > 5 and val_epoch_losses[-1] == val_epoch_losses[-2] and val_epoch_losses[-3] == val_epoch_losses[-2]:
            log.warning(f"Converged: Past 3 validation losses are all the same,"
                        + f" local optima has been reached")
            stopped_early = True
            break

        # checkpoint - save models and loss values
        torch.save(model.state_dict(), f"{conf.model_path}model_latest.pth")
        torch.save(optimiser.state_dict(), f"{conf.model_path}optimiser_latest.pth")
        np.savez_compressed(file=f"{conf.model_path}losses_latest.npz",
                            val_batch_losses=val_batch_losses,
                            val_epoch_losses=val_epoch_losses,
                            trn_epoch_losses=trn_epoch_losses,
                            trn_batch_losses=trn_batch_losses,
                            val_epoch_losses_best=val_epoch_losses_best)

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

    # todo: plot only the last_n

    return trn_epoch_losses, val_epoch_losses, stopped_early
