import logging as log
import os
from copy import deepcopy
from pprint import pprint
from time import strftime
from time import time
from torch import nn, optim

from dataloaders.flat_loader import FlatDataLoaders
from datasets.flat_dataset import FlatDataGroup
from logger.logger import setup_logging
from models.model_result import ModelResult, ModelMetrics, save_metrics, compare_all_models, get_models_metrics
from models.rnn_models import train_epoch_for_rfnn, evaluate_rfnn, \
    SimpleRecurrentFeedForwardNetwork, RecurrentFeedForwardNetwork
from trainers.generic_trainer import train_model
from utils.configs import BaseConf
from utils.data_processing import *
from utils.metrics import best_threshold, get_y_pred, get_y_pred_by_thresholds, best_thresholds
from utils.utils import pshape, get_data_sub_paths, by_ref


def main():
    data_sub_paths = get_data_sub_paths()
    pprint(sorted(data_sub_paths))

    data_sub_path = by_ref("7cd")[0]
    print(f"using: {data_sub_path}")

    # manually set
    conf = BaseConf()
    conf.seed = int(time())  # 3

    conf.model_name = "RFNN"  # needs to be created
    conf.data_path = f"./data/processed/{data_sub_path}/"

    # compare_all_models(data_path=conf.data_path)

    if not os.path.exists(conf.data_path):
        raise Exception(f"Directory ({conf.data_path}) needs to exist.")

    conf.model_path = f"{conf.data_path}models/{conf.model_name}/"
    os.makedirs(conf.data_path, exist_ok=True)
    os.makedirs(conf.model_path, exist_ok=True)

    # logging config is set globally thus we only need to call this in this file
    # imported function logs will follow the configuration
    setup_logging(save_dir=conf.model_path,
                  log_config='./logger/standard_logger_config.json',
                  default_level=log.INFO)
    log.info("=====================================BEGIN=====================================")

    info = deepcopy(conf.__dict__)
    info["start_time"] = strftime("%Y-%m-%dT%H:%M:%S")

    # DATA LOADER SETUP
    np.random.seed(conf.seed)
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(conf.seed)
    if use_cuda:
        torch.cuda.manual_seed(conf.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if use_cuda else "cpu")
    log.info(f"Device: {device}")
    info["device"] = device.type
    conf.device = device

    log.getLogger().setLevel("INFO")

    # SET THE HYPER PARAMETERS
    conf.resume = False
    conf.early_stopping = False
    conf.max_epochs = 2
    conf.dropout = 0  # 0.5 # if using dropout choose between 0.5 and 0.8 values
    conf.weight_decay = 0  # 1e-6
    conf.checkpoint = "best"  # ["best"|"latest"]
    conf.lr = 1e-3
    conf.batch_size = 64
    conf.seq_len = 30  # connect as hyper parameter to

    # CRIME DATA
    conf.sub_sample_test_set = 0
    conf.sub_sample_train_set = 1
    conf.sub_sample_validation_set = 1

    data_group = FlatDataGroup(data_path=conf.data_path, conf=conf)
    loaders = FlatDataLoaders(data_group=data_group, conf=conf)

    conf.freqstr = data_group.t_range.freqstr

    # SET LOSS FUNCTION
    # size averaged - so more epochs or larger lr for smaller batches
    if conf.use_classification:
        output_size = 2
        loss_function = nn.CrossEntropyLoss()
        log.info("model setup for binary classification")
        log.info("loss function: cross entropy loss")
    else:
        output_size = 1
        loss_function = nn.MSELoss()
        log.info("model setup for regression")
        log.info("loss function: mean square error loss")

    # SETUP MODEL
    train_set = loaders.train_loader.dataset
    indices, spc_feats, tmp_feats, env_feats, targets, labels = train_set[train_set.min_index]
    spc_size, tmp_size, env_size = spc_feats.shape[-1], tmp_feats.shape[-1], env_feats.shape[-1]

    model_arch = {
        "h_size0": 100,
        "h_size1": 100,
        "h_size2": 100,
    }
    model = SimpleRecurrentFeedForwardNetwork(spc_size=spc_size,
                                              tmp_size=tmp_size,
                                              env_size=env_size,
                                              output_size=output_size,
                                              dropout_p=conf.dropout,
                                              model_arch=model_arch)

    # model_arch = {
    #     "scp_net_h0": 64,
    #     "scp_net_h1": 32,
    #     "tmp_net_h0": 64,
    #     "tmp_net_h1": 32,
    #     "env_net_h0": 64,
    #     "env_net_h1": 32,
    #     "final_net_h1": 64,
    # }
    # model = RecurrentFeedForwardNetwork(spc_size=spc_size,
    #                                  tmp_size=tmp_size,
    #                                  env_size=env_size,
    #                                  dropout_p=conf.dropout,
    #                                  model_arch=model_arch)

    model.to(conf.device)

    # SETUP OPTIMISER
    parameters = model.parameters()

    # important note: using weight decay (l2 penalty) can prohibit long term memory in LSTM networks
    # - use gradient clipping instead
    optimiser = optim.Adam(params=parameters, lr=conf.lr, weight_decay=conf.weight_decay)

    ##### RESUME LOGIC
    if conf.resume:  # todo check if the files actually exist
        try:
            # resume from previous check point or resume from best validaton score checkpoint
            # load model state
            model_state_dict = torch.load(f"{conf.model_path}model_{conf.checkpoint}.pth",
                                          map_location=conf.device.type)
            model.load_state_dict(model_state_dict)

            # load optimiser state
            optimiser_state_dict = torch.load(f"{conf.model_path}optimiser_{conf.checkpoint}.pth",
                                              map_location=conf.device.type)
            optimiser.load_state_dict(optimiser_state_dict)

            # new optimiser hyper-parameters
            optimiser.param_groups[0]['lr'] = conf.lr
            optimiser.param_groups[0]['weight_decay'] = conf.weight_decay

            # new model hyper-parameters
            model.dropout.p = conf.dropout  # note that drop out is not part of the saved state dict

        except Exception as e:
            log.error(f"Nothing to resume from, training from scratch \n\t-> {e}")

    trn_epoch_losses, val_epoch_losses, stopped_early = train_model(model=model,
                                                                    optimiser=optimiser,
                                                                    loaders=loaders,
                                                                    train_epoch_fn=train_epoch_for_rfnn,
                                                                    loss_fn=loss_function,
                                                                    conf=conf)

    print(f"stopped_early: {stopped_early}")  # use the current epoch instead
    # if stopped_early -> continue with best_model - new hyper-parameters -> no n

    # Load latest or best validation model
    # conf.checkpoint = "latest_val"
    # conf.checkpoint = "latest_trn"
    # conf.checkpoint = "best_trn"
    conf.checkpoint = "best_val"

    log.info(f"Loading model from checkpoint ({conf.checkpoint}) for evaluation")

    # resume from previous check point or resume from best validation score checkpoint
    # load model state
    model_state_dict = torch.load(f"{conf.model_path}model_{conf.checkpoint}.pth",
                                  map_location=conf.device.type)

    model.load_state_dict(model_state_dict)

    conf.sub_sample_test_set = 0
    conf.sub_sample_train_set = 0
    conf.sub_sample_validation_set = 0

    loaders = FlatDataLoaders(data_group=data_group, conf=conf)

    # todo set the train_loader to eval so that it does not subsample
    trn_y_count, trn_y_class, trn_y_score, trn_t_range = evaluate_rfnn(model=model,
                                                                       batch_loader=loaders.train_loader,
                                                                       conf=conf)

    thresh = best_threshold(y_class=trn_y_class,
                            y_score=trn_y_score)

    tst_y_count, tst_y_class, tst_y_score, tst_t_range = evaluate_rfnn(model=model,
                                                                       batch_loader=loaders.test_loader,
                                                                       conf=conf)

    tst_y_pred = get_y_pred(thresh=thresh,
                            y_score=tst_y_score)

    tst_y_count = loaders.data_group.to_counts(dense_data=tst_y_count)

    save_metrics(y_count=tst_y_count,
                 y_pred=tst_y_pred,
                 y_score=tst_y_score,
                 t_range=tst_t_range,
                 shaper=data_group.shaper,
                 conf=conf)


if __name__ == '__main__':
    main()
