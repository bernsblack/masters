import logging as log
import os
from copy import deepcopy
from time import strftime

from torch import nn, optim

from dataloaders.flat_loader import FlatDataLoaders
from logger.logger import setup_logging
from models.kangkang_fnn_models import KangFeedForwardNetwork
from utils.configs import BaseConf
from utils.data_processing import *
from utils.utils import Timer

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
    "resume": False,
    "early_stopping": False,
    "sub_sample_train_set": True,
    "sub_sample_validation_set": True,
    "sub_sample_test_set": False,
    "use_cuda": False,
    "val_ratio": 0.1,
    "tst_ratio": 0.2,
    "flatten_grid": True,
    "lr": 5e-3,
    "weight_decay": 1e-8,
    "max_epochs": 1,
    "batch_size": 256,
    "dropout": 0.1,
    "shuffle": False,
    "seq_len": 1,
}
conf = BaseConf(conf_dict=conf_dict)

info = deepcopy(conf.__dict__)
info["start_time"] = strftime("%Y-%m-%dT%H:%M:%S")

# DATA LOADER SETUP
np.random.seed(conf.seed)
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed(conf.seed)
else:
    torch.manual_seed(conf.seed)

device = torch.device("cuda:0" if use_cuda else "cpu")
log.info(f"Device: {device}")
info["device"] = device.type

# GET DATA
loaders = FlatDataLoaders(data_path=data_path, conf=conf)

# SET MODEL PARAMS
train_set = loaders.train_loader.dataset
spc_feats, tmp_feats, env_feats, target = train_set[train_set.min_index]
spc_size, tmp_size, env_size = spc_feats.shape[-1], tmp_feats.shape[-1], env_feats.shape[-1]

model = KangFeedForwardNetwork(spc_size=spc_size, tmp_size=tmp_size, env_size=env_size, dropout_p=conf.dropout)
model.to(device)

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

for epoch in range(conf.max_epochs):
    log.info(f"Epoch: {(1 + epoch):04d}/{conf.max_epochs:04d}")
    timer.reset()
    # Training loop
    tmp_trn_loss = []
    num_batches = loaders.train_loader.num_batches
    for spc_feats, tmp_feats, env_feats, targets in loaders.train_loader:
        current_batch = loaders.train_loader.current_batch

        # Transfer to PyTorch Tensor and GPU
        spc_feats = torch.Tensor(spc_feats).to(device)
        tmp_feats = torch.Tensor(tmp_feats).to(device)
        env_feats = torch.Tensor(env_feats).to(device)
        targets = torch.LongTensor(targets).to(device)
        out = model(spc_feats, tmp_feats, env_feats)

        loss = loss_function(out, targets)
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
            spc_feats = torch.Tensor(spc_feats).to(device)
            tmp_feats = torch.Tensor(tmp_feats).to(device)
            env_feats = torch.Tensor(env_feats).to(device)
            targets = torch.LongTensor(targets).to(device)
            out = model(spc_feats, tmp_feats, env_feats)

            loss = loss_function(out, targets)
            tmp_val_loss.append(loss.item())
            all_val_loss.append(tmp_val_loss[-1])
    val_loss.append(np.mean(tmp_val_loss))
    log.debug(f"Epoch {epoch} -> Validation Loop Duration: {timer.check()}")

    log.info(f"\tLoss (Trn): \t{trn_loss[-1]:.5f}")
    log.info(f"\tLoss (Val): \t{val_loss[-1]:.5f}")
    log.info(f"\tLoss (Dif): \t{np.abs(val_loss[-1] - trn_loss[-1]):.5f}\n")

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