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
from models.baseline_models import LinearRegressor
from dataloaders.generic_loader import GenericDataLoaders
from utils.metrics import PRCurvePlotter, ROCCurvePlotter, LossPlotter
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

if __name__ == "__main__":

    data_dim_str = "MOCK-DATA"  # needs to exist
    model_name = "LINEAR-MOCK-MODEL"  # needs to be created

    data_path = f"./data/processed/{data_dim_str}/"
    model_path = data_path + f"models/{model_name}/"
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # logging config is set globally thus we only need to call this in this file
    # imported function logs will follow the configuration
    setup_logging(save_dir=model_path, log_config='./logger/standard_logger_config.json', default_level=log.INFO)
    log.info("=======================================BEGIN=======================================")

    timer = Timer()
    conf = BaseConf()

    info = deepcopy(conf.__dict__)
    info["start_time"] = strftime("%Y-%m-%dT%H:%M:%S")

    # DATA LOADER SETUP
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # GET DATA
    loaders = GenericDataLoaders()  # torch seed is set in the data loaders

    # TRAIN MODEL
    model = LinearRegressor(in_features=loaders.in_size, out_features=loaders.out_size)
    model.to(device)

    loss_function = nn.CrossEntropyLoss()

    trn_loss = []
    val_loss = []
    val_loss_best = float("inf")

    all_trn_loss = []
    all_val_loss = []

    optimiser = optim.Adam(params=model.parameters(), lr=0.1, weight_decay=0.001)
    if conf.resume:
        # load model and optimiser states
        model.load_state_dict(torch.load(model_path + "model.pth"))
        optimiser.load_state_dict(torch.load(model_path + "optimiser.pth"))
        # load losses
        losses_zip = np.load(model_path + "losses.npz")
        all_val_loss = losses_zip["all_val_loss"].tolist()
        val_loss = losses_zip["val_loss"].tolist()
        trn_loss = losses_zip["trn_loss"].tolist()
        all_trn_loss = losses_zip["all_trn_loss"].tolist()
        val_loss_best = float(losses_zip["val_loss_best"])

    for epoch in range(conf.max_epochs):
        timer.reset()
        # Training loop
        tmp_trn_loss = []
        for local_batch, local_labels in loaders.training_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            out = model(local_batch.float())

            loss = loss_function(out, local_labels)
            tmp_trn_loss.append(loss.item())
            all_trn_loss.append(tmp_trn_loss[-1])

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        trn_loss.append(np.mean(tmp_trn_loss))
        log.debug(f"Epoch {epoch} -> Training Loop Duration: {timer.check()}")
        timer.reset()

        # Validation loop
        tmp_val_loss = []
        with torch.set_grad_enabled(False):
            # Transfer to GPU
            for local_batch, local_labels in loaders.validation_generator:
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                out = model(local_batch.float())

                loss = loss_function(out, local_labels)
                tmp_val_loss.append(loss.item())
                all_val_loss.append(tmp_val_loss[-1])
        val_loss.append(np.mean(tmp_val_loss))
        log.debug(f"Epoch {epoch} -> Validation Loop Duration: {timer.check()}")

        log.info(f"Epoch {epoch} -> Training Loss: \t\t{trn_loss[-1]:.5f}")
        log.info(f"Epoch {epoch} -> Validation Loss: \t{val_loss[-1]:.5f}")
        log.info(f"Epoch {epoch} -> Difference: \t\t{(val_loss[-1]-trn_loss[-1]):.5f}\n")


        # save best model
        if min(val_loss) < val_loss_best:
            val_loss_best = min(val_loss)
            torch.save(model.state_dict(), model_path + "model_best.pth")
            torch.save(optimiser.state_dict(), model_path + "optimiser_best.pth")

        # model has been over-fitting stop maybe? # average of val_loss has increase - starting to over-fit
        if conf.early_stopping and epoch != 0 and val_loss[-1] > val_loss[-2]:
            log.warning("Over-fitting has taken place - stopping early")
            break

        # checkpoint - save models and loss values
        torch.save(model.state_dict(), model_path + "model.pth")
        torch.save(optimiser.state_dict(), model_path + "optimiser.pth")
        np.savez_compressed(model_path + "losses.npz",
                            all_val_loss=all_val_loss,
                            val_loss=val_loss,
                            trn_loss=trn_loss,
                            all_trn_loss=all_trn_loss,
                            val_loss_best=val_loss_best)

    # Save training and validation plots
    loss_plotter = LossPlotter(title="Cross Entropy Loss of Linear Regression Model")
    loss_plotter.plot_losses(trn_loss, all_trn_loss, val_loss, all_val_loss)
    loss_plotter.savefig(model_path + "plot_train_val_loss.png")

    # EVALUATE MODEL
    with torch.set_grad_enabled(False):
        # Transfer to GPU
        testing_losses = []
        y_true = []
        y_pred = []
        probas_pred = []
        for local_batch, local_labels in loaders.testing_generator:  # loop through is set does not fit in batch
            y_true.extend(local_labels.tolist())
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            out = model(local_batch.float())
            out = F.softmax(out, dim=1)
            out_label = torch.argmax(out, dim=1)
            y_pred.extend(out_label.tolist())
            out_proba = out[:, 1]  # likelihood of crime is more general form - when comparing to moving averages
            probas_pred.extend(out_proba.tolist())

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, probas_pred)
    ap = average_precision_score(y_true, probas_pred)
    log.info(f"Accuracy:\t\t {acc:.4f}")
    log.info(f"ROC AUC:\t\t {auc:.4f}")
    log.info(f"Average Precision:\t {ap:.4f}")

    np.savez_compressed(model_path + "evaluation_results.npz",
                        acc=acc,
                        auc=auc,
                        ap=ap,
                        y_true=y_true,
                        y_pred=y_pred,
                        probas_pred=probas_pred)

    pr_plotter = PRCurvePlotter()
    pr_plotter.add_curve(y_true, probas_pred, label_name="Model A")
    pr_plotter.add_curve(y_true, np.roll(probas_pred, 1), label_name="Model B")
    pr_plotter.savefig(model_path + "plot_pr_curve.png")

    roc_plotter = ROCCurvePlotter()
    roc_plotter.add_curve(y_true, probas_pred, label_name="Model A")
    roc_plotter.add_curve(y_true, np.roll(probas_pred, 1), label_name="Model B")
    roc_plotter.savefig(model_path + "plot_roc_curve.png")

    info["stop_time"] = strftime("%Y-%m-%dT%H:%M:%S")
    write_json(info, model_path + "info.json")

    log.info("=======================================END=======================================")
