import logging as log
import os
from copy import deepcopy
from time import strftime

import torch.nn.functional as F
from torch import nn, optim

from dataloaders.flat_loader import FlatDataLoaders
from logger.logger import setup_logging
from models.kangkang_fnn_models import KangFeedForwardNetwork
from models.model_result import ModelResult
from utils.configs import BaseConf
from utils.data_processing import *
from utils.metrics import PRCurvePlotter, ROCCurvePlotter, LossPlotter
from utils.utils import write_json, Timer

data_dim_str = "T24H-X850M-Y880M"  # "T1H-X1700M-Y1760M"  # needs to exist
model_name = "FNN-CRIME-MODEL"  # needs to be created

data_path = f"./data/processed/{data_dim_str}/"
model_path = data_path + f"models/{model_name}/"
os.makedirs(data_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# logging config is set globally thus we only need to call this in this file
# imported function logs will follow the configuration
setup_logging(save_dir=model_path, log_config='./logger/standard_logger_config.json', default_level=log.INFO)
log.info("=====================================BEGIN=====================================")

timer = Timer()
# manually set the config
conf_dict = {
    "seed": 3,
    "use_cuda": False,

    # data related hyper-params
    "val_ratio": 0.1,  # ratio of the total dataset
    "tst_ratio": 0.2,  # ratio of the total dataset
    "sub_sample_train_set": 1,
    "sub_sample_validation_set": 1,
    "sub_sample_test_set": 0,
    "flatten_grid": True,  # if the shaper should be used to squeeze the data
    "seq_len": 1,
    "shaper_top_k": -1,  # if less then 0, top_k will not be applied
    "shaper_threshold": 0,

    # training parameters
    "resume": False,
    "early_stopping": False,
    "lr": 1e-3,
    "weight_decay": 1e-8,
    "max_epochs": 1,
    "batch_size": 64,
    "dropout": 0,
    "shuffle": False,
    "num_workers": 6,  # only relevant for pure pytorch loaders
}

conf = BaseConf(conf_dict=conf_dict)

info = deepcopy(conf.__dict__)  # used as quick-reference when comparing models from terminal
info["start_time"] = strftime("%Y-%m-%dT%H:%M:%S")

np.random.seed(conf.seed)
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed(conf.seed)  # linter might complain if cuda PyTorch not installed on computer
else:
    torch.manual_seed(conf.seed)

device = torch.device("cuda:0" if use_cuda else "cpu")
log.info(f"Device: {device}")
info["device"] = device.type

# DATA LOADER SETUP ----------------------------------------------------------------------------------------------------
loaders = FlatDataLoaders(data_path=data_path, conf=conf)

# SET MODEL PARAMS -----------------------------------------------------------------------------------------------------
train_set = loaders.train_loader.dataset
spc_feats, tmp_feats, env_feats, target = train_set[train_set.min_index]
spc_size, tmp_size, env_size = spc_feats.shape[-1], tmp_feats.shape[-1], env_feats.shape[-1]

model = KangFeedForwardNetwork(spc_size=spc_size, tmp_size=tmp_size, env_size=env_size, dropout_p=conf.dropout)
model.to(device)

# def train(data_loader, model):


#  LOAD MODEL AND OPTIMIZER STATES AND LOSS CURVES ---------------------------------------------------------------------
loss_function = nn.CrossEntropyLoss()

trn_loss = []
val_loss = []
val_loss_best = float("inf")

all_trn_loss = []
all_val_loss = []

optimiser = optim.Adam(params=model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
if conf.resume:
    # load model and optimiser states
    model_state_dict = torch.load(model_path + "model_best.pth", map_location=device.type)
    model.load_state_dict(model_state_dict)
    optimiser_state_dict = torch.load(model_path + "optimiser_best.pth", map_location=device.type)
    optimiser.load_state_dict(optimiser_state_dict)
    # load losses
    losses_zip = np.load(model_path + "losses.npz")
    all_val_loss = losses_zip["all_val_loss"].tolist()
    val_loss = losses_zip["val_loss"].tolist()
    trn_loss = losses_zip["trn_loss"].tolist()
    all_trn_loss = losses_zip["all_trn_loss"].tolist()
    val_loss_best = float(losses_zip["val_loss_best"])
    # todo only load loss since last best_checkpoint

#  TRAINING LOOP ---------------------------------------------------------------------
for epoch in range(conf.max_epochs):
    log.info(f"Epoch: {(1 + epoch):04d}/{conf.max_epochs:04d}")
    timer.reset()
    # Training loop
    tmp_trn_loss = []
    num_batches = loaders.train_loader.num_batches
    for spc_feats, tmp_feats, env_feats, targets in loaders.train_loader:
        current_batch = loaders.train_loader.current_batch

        # Transfer to PyTorch Tensor and GPU
        spc_feats = torch.Tensor(spc_feats[-1]).to(device)  # only taking [-1] for fnn
        tmp_feats = torch.Tensor(tmp_feats[-1]).to(device)  # only taking [-1] for fnn
        env_feats = torch.Tensor(env_feats[-1]).to(device)  # only taking [-1] for fnn
        targets = torch.LongTensor(targets[-1, :, 0]).to(device)  # only taking [-1] for fnn
        out = model(spc_feats, tmp_feats, env_feats)
        loss = loss_function(input=out, target=targets)
        tmp_trn_loss.append(loss.item())
        all_trn_loss.append(tmp_trn_loss[-1])

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        log.debug(f"Batch: {current_batch:04d}/{num_batches:04d} \t Loss: {tmp_trn_loss[-1]:.4f}")
    trn_loss.append(np.mean(tmp_trn_loss))
    log.debug(f"Epoch {epoch} -> Training Loop Duration: {timer.check()}")
    timer.reset()

    # Validation loop
    tmp_val_loss = []
    with torch.set_grad_enabled(False):
        # Transfer to GPU
        for spc_feats, tmp_feats, env_feats, targets in loaders.validation_loader:
            # Transfer to GPU
            spc_feats = torch.Tensor(spc_feats[-1]).to(device)  # only taking [-1] for fnn
            tmp_feats = torch.Tensor(tmp_feats[-1]).to(device)  # only taking [-1] for fnn
            env_feats = torch.Tensor(env_feats[-1]).to(device)  # only taking [-1] for fnn
            targets = torch.LongTensor(targets[-1, :, 0]).to(device)  # only taking [-1] for fnn
            out = model(spc_feats, tmp_feats, env_feats)

            loss = loss_function(input=out, target=targets)
            tmp_val_loss.append(loss.item())
            all_val_loss.append(tmp_val_loss[-1])
    val_loss.append(np.mean(tmp_val_loss))
    log.debug(f"Epoch {epoch} -> Validation Loop Duration: {timer.check()}")

    # save best model
    if min(val_loss) < val_loss_best:
        val_loss_best = min(val_loss)
        torch.save(model.state_dict(), model_path + "model_best.pth")
        torch.save(optimiser.state_dict(), model_path + "optimiser_best.pth")

    log.info(f"\tLoss (Trn): \t{trn_loss[-1]:.5f}")
    log.info(f"\tLoss (Val): \t{val_loss[-1]:.5f}")
    log.info(f"\tLoss (Best Val): \t{val_loss_best:.5f}")
    log.info(f"\tLoss (Dif): \t{np.abs(val_loss[-1] - trn_loss[-1]):.5f}\n")

    # model has been over-fitting stop maybe? # average of val_loss has increase - starting to over-fit
    if conf.early_stopping and epoch != 0 and val_loss[-1] > val_loss[-2]:
        log.warning("Over-fitting has taken place - stopping early")
        break

    # checkpoint - save models and loss values every epoch
    torch.save(model.state_dict(), model_path + "model.pth")
    torch.save(optimiser.state_dict(), model_path + "optimiser.pth")
    np.savez_compressed(model_path + "losses.npz",
                        all_val_loss=all_val_loss,
                        val_loss=val_loss,
                        trn_loss=trn_loss,
                        all_trn_loss=all_trn_loss,
                        val_loss_best=val_loss_best)

# SAVE TRAINING AND VALIDATION PLOTS -----------------------------------------------------------------------------------
skip = 0
loss_plotter = LossPlotter(title="Cross Entropy Loss of Linear Regression Model")
loss_plotter.plot_losses(trn_loss, all_trn_loss[skip:], val_loss, all_val_loss[skip:])
loss_plotter.savefig(model_path + "plot_train_val_loss.png")

# EVALUATION LOOP ------------------------------------------------------------------------------------------------------
# todo create a model_result - take size from test_set's shape, create empty ndarray - save each result in the ...
# todo creare a model_result - ndarray using the indix (n,l) from the loader.
model.load_state_dict(torch.load(model_path + "model_best.pth"))
conf = BaseConf(conf_dict=conf_dict)
loaders = FlatDataLoaders(data_path=data_path, conf=conf)

# EVALUATE MODEL
with torch.set_grad_enabled(False):
    # Transfer to GPU
    testing_losses = []
    y_true = []
    y_pred = []
    probas_pred = []

    # loop through is set does not fit in batch
    for spc_feats, tmp_feats, env_feats, targets in loaders.test_loader:
        """
        IMPORTNANT NOTE: WHEN DOING LSTM - ONLY FEED THE TEMPORAL VECTORS IN THE LSTM
        FEED THE REST INTO THE NORMAL LINEAR NETWORKS
        """

        # Transfer to GPU
        spc_feats = torch.Tensor(spc_feats).to(device)
        tmp_feats = torch.Tensor(tmp_feats).to(device)
        env_feats = torch.Tensor(env_feats).to(device)
        targets = torch.LongTensor(targets).to(device)

        y_true.extend(targets.tolist())
        out = model(spc_feats, tmp_feats, env_feats)
        out = F.softmax(out, dim=1)
        out_label = torch.argmax(out, dim=1)
        y_pred.extend(out_label.tolist())
        out_proba = out[:, 1]  # likelihood of crime is more general form - when comparing to moving averages
        probas_pred.extend(out_proba.tolist())

model_result = ModelResult(model_name="FNN (Kang and Kang)",
                           y_true=y_true,
                           y_pred=y_pred,
                           probas_pred=probas_pred,
                           t_range=loaders.test_loader.dataset.t_range,
                           shape=None)  # todo change to be the shape (N,L) of the original prediction.

log.info(model_result)

np.savez_compressed(model_path + "evaluation_results.npz", model_result)

pr_plotter = PRCurvePlotter()
pr_plotter.add_curve(y_true, probas_pred, label_name="FNN (Kang and Kang)")
pr_plotter.savefig(model_path + "plot_pr_curve.png")

roc_plotter = ROCCurvePlotter()
roc_plotter.add_curve(y_true, probas_pred, label_name="FNN (Kang and Kang)")
roc_plotter.savefig(model_path + "plot_roc_curve.png")

info["stop_time"] = strftime("%Y-%m-%dT%H:%M:%S")
write_json(info, model_path + "info.json")

log.info("=====================================END=====================================")
