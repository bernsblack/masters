import torch
import logging as log
import numpy as np
from utils.metrics import LossPlotter


# generic training loop
def train_model(model, optimiser, loaders, train_epoch_fn, loss_fn, conf):
    """
    Generic training loop that handles:
    - early stopping
    - timing
    - logging
    - model checkpoints
    - saving epoch and batch losses

    :returns: best validation loss of all the epochs - used to tune the hyper-parameters of the models/optimiser
    """
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

        log.info(f"\tLoss (Trn): \t{trn_epoch_losses[-1]:.5f}")
        log.info(f"\tLoss (Val): \t{val_epoch_losses[-1]:.5f}")
        log.info(f"\tLoss (Dif): \t{np.abs(val_epoch_losses[-1] - trn_epoch_losses[-1]):.5f}\n")

        # save best model
        if val_epoch_losses[-1] < val_epoch_losses_best:
            val_epoch_losses_best = val_epoch_losses[-1]
            torch.save(model.state_dict(), f"{conf.model_path}model_best.pth")
            torch.save(optimiser.state_dict(), f"{conf.model_path}optimiser_best.pth")
            np.savez_compressed(file=f"{conf.model_path}losses_best.npz",
                                val_batch_losses=val_batch_losses,
                                val_epoch_losses=val_epoch_losses,
                                trn_epoch_losses=trn_epoch_losses,
                                trn_batch_losses=trn_batch_losses,
                                val_epoch_losses_best=val_epoch_losses_best)

        # increasing moving average of val_epoch_losses
        if conf.early_stopping and epoch > 5 and np.sum(np.diff(val_epoch_losses[-5:])) > 0:
            log.warning("Early stopping: Over-fitting has taken place")
            stopped_early = True
            break

        if conf.early_stopping and epoch > 5 and np.abs(val_epoch_losses[-1] - val_epoch_losses[-2]) < conf.tolerance:
            log.warning(f"Converged: Difference between the past two"
                        + f" validation losses is within tolerance of {conf.tolerance}")
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
    loss_plotter = LossPlotter(title=f"Cross Entropy Loss ({conf.model_name})")
    loss_plotter.plot_losses(trn_epoch_losses, trn_batch_losses[skip:], val_epoch_losses, val_batch_losses[skip:])
    loss_plotter.savefig(f"{conf.model_path}plot_train_val_epoch_losses.png")

    return val_epoch_losses_best, stopped_early
