import logging as log
import os

from dataloaders.flat_loader import FlatDataLoaders
from datasets.flat_dataset import FlatDataGroup
from logger.logger import setup_logging
from models.baseline_models import ExponentialMovingAverage, UniformMovingAverage, \
    TriangularMovingAverage, HistoricAverage
from models.model_result import save_metrics
from utils.configs import BaseConf
from utils.data_processing import *
from utils.metrics import best_threshold, get_y_pred
from utils.plots import im
from utils.utils import by_ref

if __name__ == '__main__':
    data_sub_paths = by_ref("939")
    print(f"using: {data_sub_paths}")

    # run base line models for all data dimensions
    for data_sub_path in data_sub_paths:
        conf = BaseConf()

        conf.data_path = f"./data/processed/{data_sub_path}/"

        setup_logging(save_dir=conf.data_path,
                      log_config='./logger/standard_logger_config.json',
                      default_level=log.INFO)
        log.info("\n====================================================" + \
                 "===============================================================\n" + \
                 f"CALCULATING BASE LINE RESULTS FOR {data_sub_path}\n" + \
                 "==============================================================" + \
                 "===t==================================================")

        log.info("=====================================BEGIN=====================================")

        if not os.path.exists(conf.data_path):
            raise Exception(f"Directory ({conf.data_path}) needs to exist.")

        # DATA LOADER SETUP
        np.random.seed(conf.seed)

        # CRIME DATA
        data_group = FlatDataGroup(data_path=conf.data_path, conf=conf)
        loaders = FlatDataLoaders(data_group=data_group, conf=conf)

        # LOG CLASS DISTRIBUTION
        vals, counts = np.unique(data_group.targets, return_counts=True)
        counts = counts / np.sum(counts)
        dist = dict(zip(vals, counts))
        log.info(f"class distribution: {dist}")

        test_set_size = data_group.testing_set.target_shape[0]
        crimes = data_group.shaper.unsqueeze(data_group.crimes)
        im(crimes.mean(0)[0])
        crimes = data_group.crimes
        t_range = data_group.t_range

        # -----------HISTORIC AVERAGE
        # Create model folder and setup logging for model
        conf.model_name = "Historic Average"  # needs to be created
        conf.model_path = f"{conf.data_path}models/{conf.model_name}/"
        os.makedirs(conf.model_path, exist_ok=True)
        setup_logging(save_dir=conf.model_path,
                      log_config='./logger/standard_logger_config.json',
                      default_level=log.INFO)
        log.info("=====================================BEGIN=====================================")

        # time step in this context is used for
        freqstr = t_range.freqstr
        if freqstr == "D":
            freqstr = "24H"
        if freqstr == "H":
            freqstr = "1H"
        time_step = int(24 / int(freqstr[:freqstr.find("H")]))

        log.info(f"using time step: {time_step}")

        test_set_size = data_group.testing_set.target_shape[0]

        ha = HistoricAverage(step=time_step)
        all_crimes = data_group.crimes[:, 0]
        all_targets = data_group.targets
        all_labels = data_group.labels
        all_crimes_ha = ha.fit_transform(data_group.crimes[:, 0:1])[:, 0]

        ha.fit(data_group.crimes[:, 0:1])
        ha.max_steps = -1
        all_crimes_ha = ha.transform(data_group.crimes[:, 0:1])[:, 0]

        all_t_range = data_group.t_range

        tst_crimes_ha = all_crimes_ha[-test_set_size:]
        tst_targets = all_targets[-test_set_size:]
        tst_t_range = all_t_range[-test_set_size:]

        trn_y_score = all_crimes_ha[time_step + 1:-test_set_size]  # skip all the nan values
        trn_y_true = all_targets[time_step + 1:-test_set_size]
        trn_y_class = all_labels[time_step + 1:-test_set_size]

        thresh = best_threshold(y_class=trn_y_class,
                                y_score=trn_y_score)  # should only come from the train predictions

        tst_y_true = tst_targets
        tst_y_count = data_group.to_counts(tst_y_true)
        tst_y_score = tst_crimes_ha
        tst_y_pred = get_y_pred(thresh, tst_y_score)  # might want to do this for each cell either?

        tst_y_pred = np.expand_dims(tst_y_pred, axis=1)
        tst_y_score = np.expand_dims(tst_y_score, axis=1)

        log.info(f"======== {conf.model_path}  ========")
        save_metrics(y_count=tst_y_count,
                     y_pred=tst_y_pred,
                     y_score=tst_y_score,
                     t_range=tst_t_range,
                     shaper=data_group.shaper,
                     conf=conf)

        log.info("=====================================END=====================================\n")
        # -----------HISTORIC AVERAGE

        # -----------GLOBAL AVERAGE
        # Create model folder and setup logging for model
        conf.model_name = "Global Average"  # needs to be created
        conf.model_path = f"{conf.data_path}models/{conf.model_name}/"
        os.makedirs(conf.model_path, exist_ok=True)
        setup_logging(save_dir=conf.model_path,
                      log_config='./logger/standard_logger_config.json',
                      default_level=log.INFO)
        log.info("=====================================BEGIN=====================================")

        test_set_size = data_group.testing_set.target_shape[0]

        all_crimes = data_group.crimes[:, 0]
        all_targets = data_group.targets
        all_labels = data_group.labels

        trn_crimes = all_crimes[:-test_set_size]
        trn_targets = all_targets[:-test_set_size]
        trn_labels = all_labels[:-test_set_size]

        tst_crimes = all_crimes[-test_set_size:]
        tst_targets = all_targets[-test_set_size:]
        tst_t_range = all_t_range[-test_set_size:]

        trn_mean = np.mean(trn_crimes, axis=0, keepdims=True)  # keep dims used to make scalar product easy
        trn_ones = np.ones_like(trn_crimes, dtype=np.float)
        trn_y_score = trn_mean * trn_ones

        thresh = best_threshold(y_class=trn_labels,
                                y_score=trn_y_score)  # should only come from the train predictions

        # only use the training sets - mean
        tst_ones = np.ones_like(tst_crimes, dtype=np.float)
        tst_y_score = trn_mean * tst_ones

        tst_y_count = data_group.to_counts(tst_targets)
        tst_y_pred = get_y_pred(thresh, tst_y_score)  # might want to do this for each cell either?

        tst_y_pred = np.expand_dims(tst_y_pred, axis=1)
        tst_y_score = np.expand_dims(tst_y_score, axis=1)

        log.info(f"======== {conf.model_path}  ========")
        save_metrics(y_count=tst_y_count,
                     y_pred=tst_y_pred,
                     y_score=tst_y_score,
                     t_range=tst_t_range,
                     shaper=data_group.shaper,
                     conf=conf)

        log.info("=====================================END=====================================\n")
        # -----------GLOBAL AVERAGE

        # ------------PREVIOUS TIME STEP
        # Create model folder and setup logging for model
        conf.model_name = "Previous Time Step"  # needs to be created
        conf.model_path = f"{conf.data_path}models/{conf.model_name}/"

        os.makedirs(conf.model_path, exist_ok=True)
        setup_logging(save_dir=conf.model_path,
                      log_config='./logger/standard_logger_config.json',
                      default_level=log.INFO)

        log.info("=====================================BEGIN=====================================")

        test_set_size = data_group.testing_set.target_shape[0]

        all_y_score = data_group.crimes[:, 0]
        all_targets = data_group.targets
        all_labels = data_group.labels
        all_t_range = data_group.t_range

        tst_y_score = all_y_score[-test_set_size:]
        tst_targets = all_targets[-test_set_size:]
        tst_t_range = all_t_range[-test_set_size:]

        trn_y_score = all_y_score[:-test_set_size]  # skip all the nan values
        trn_y_true = all_targets[:-test_set_size]
        trn_y_class = all_labels[:-test_set_size]

        thresh = best_threshold(y_class=trn_y_class,
                                y_score=trn_y_score)  # should only come from the train predictions

        tst_y_count = data_group.to_counts(tst_y_true)
        tst_y_pred = get_y_pred(thresh, tst_y_score)  # might want to do this for each cell either?

        tst_y_pred = np.expand_dims(tst_y_pred, axis=1)
        tst_y_score = np.expand_dims(tst_y_score, axis=1)

        save_metrics(y_count=tst_y_count,
                     y_pred=tst_y_pred,
                     y_score=tst_y_score,
                     t_range=tst_t_range,
                     shaper=data_group.shaper,
                     conf=conf)

        log.info("=====================================END=====================================\n")
        # ------------PREVIOUS TIME STEP

        # #    --------------    --------------    --------------    --------------    --------------    --------------

        # ------------Uniform Moving Average
        # Create model folder and setup logging for model
        conf.model_name = "Uni. Mov. Avg."  # needs to be created
        conf.model_path = f"{conf.data_path}models/{conf.model_name}/"

        os.makedirs(conf.model_path, exist_ok=True)
        setup_logging(save_dir=conf.model_path,
                      log_config='./logger/standard_logger_config.json',
                      default_level=log.INFO)
        log.info("=====================================BEGIN=====================================")

        # time step in this context is used for
        freqstr = t_range.freqstr
        if freqstr == "D":
            freqstr = "24H"
        if freqstr == "H":
            freqstr = "1H"
        time_step = int(24 / int(freqstr[:freqstr.find("H")]))
        time_step = 30 * time_step

        log.info(f"using time step: {time_step}")

        test_set_size = data_group.testing_set.target_shape[0]

        window_len = time_step

        ma = UniformMovingAverage(window_len=window_len)

        all_crimes = data_group.crimes[:, 0]
        all_targets = data_group.targets
        all_labels = data_group.labels
        all_y_score = ma(all_crimes)
        all_t_range = data_group.t_range

        tst_y_score = all_y_score[-test_set_size:]
        tst_targets = all_targets[-test_set_size:]
        tst_t_range = all_t_range[-test_set_size:]

        trn_y_score = all_y_score[time_step + 1:-test_set_size]  # skip all the nan values
        trn_y_true = all_targets[time_step + 1:-test_set_size]
        trn_y_class = all_labels[time_step + 1:-test_set_size]

        thresh = best_threshold(y_class=trn_y_class,
                                y_score=trn_y_score)  # should only come from the train predictions

        tst_y_true = tst_targets
        tst_y_count = data_group.to_counts(tst_y_true)
        tst_y_score = tst_y_score
        tst_y_pred = get_y_pred(thresh, tst_y_score)  # might want to do this for each cell either?

        tst_y_pred = np.expand_dims(tst_y_pred, axis=1)
        tst_y_score = np.expand_dims(tst_y_score, axis=1)

        log.info(f"======== {conf.model_path}  ========")
        save_metrics(y_count=tst_y_count,
                     y_pred=tst_y_pred,
                     y_score=tst_y_score,
                     t_range=tst_t_range,
                     shaper=data_group.shaper,
                     conf=conf)

        log.info("=====================================END=====================================\n")
        # ------------Uniform Moving Average

        #############################################################################################################

        # ------------Exponential Moving Average
        # Create model folder and setup logging for model
        conf.model_name = "Exp. Mov. Avg."  # needs to be created
        conf.model_path = f"{conf.data_path}models/{conf.model_name}/"

        os.makedirs(conf.model_path, exist_ok=True)
        setup_logging(save_dir=conf.model_path,
                      log_config='./logger/standard_logger_config.json',
                      default_level=log.INFO)
        log.info("=====================================BEGIN=====================================")

        # time step in this context is used for
        freqstr = t_range.freqstr
        if freqstr == "D":
            freqstr = "24H"
        if freqstr == "H":
            freqstr = "1H"
        time_step = int(24 / int(freqstr[:freqstr.find("H")]))
        time_step = 30 * time_step

        log.info(f"using time step: {time_step}")

        test_set_size = data_group.testing_set.target_shape[0]

        alpha = 1e-2
        window_len = time_step

        ma = ExponentialMovingAverage(alpha=alpha, window_len=window_len)

        all_crimes = data_group.crimes[:, 0]
        all_targets = data_group.targets
        all_labels = data_group.labels
        all_y_score = ma(all_crimes)
        all_t_range = data_group.t_range

        tst_y_score = all_y_score[-test_set_size:]
        tst_targets = all_targets[-test_set_size:]
        tst_t_range = all_t_range[-test_set_size:]

        trn_y_score = all_y_score[time_step + 1:-test_set_size]  # skip all the nan values
        trn_y_true = all_targets[time_step + 1:-test_set_size]
        trn_y_class = all_labels[time_step + 1:-test_set_size]

        thresh = best_threshold(y_class=trn_y_class,
                                y_score=trn_y_score)  # should only come from the train predictions

        tst_y_true = tst_targets
        tst_y_count = data_group.to_counts(tst_y_true)
        tst_y_score = tst_y_score
        tst_y_pred = get_y_pred(thresh, tst_y_score)  # might want to do this for each cell either?

        tst_y_pred = np.expand_dims(tst_y_pred, axis=1)
        tst_y_score = np.expand_dims(tst_y_score, axis=1)

        log.info(f"======== {conf.model_path}  ========")
        save_metrics(y_count=tst_y_count,
                     y_pred=tst_y_pred,
                     y_score=tst_y_score,
                     t_range=tst_t_range,
                     shaper=data_group.shaper,
                     conf=conf)

        log.info("=====================================END=====================================\n")
        # ------------Exponential Moving Average

        # ------------Triangular Moving Average
        # Create model folder and setup logging for model
        conf.model_name = "Tri. Mov. Avg."  # needs to be created
        conf.model_path = f"{conf.data_path}models/{conf.model_name}/"

        os.makedirs(conf.model_path, exist_ok=True)
        setup_logging(save_dir=conf.model_path,
                      log_config='./logger/standard_logger_config.json',
                      default_level=log.INFO)
        log.info("=====================================BEGIN=====================================")

        # time step in this context is used for
        freqstr = t_range.freqstr
        if freqstr == "D":
            freqstr = "24H"
        if freqstr == "H":
            freqstr = "1H"
        time_step = int(24 / int(freqstr[:freqstr.find("H")]))
        time_step = 30 * time_step

        log.info(f"using time step: {time_step}")

        test_set_size = data_group.testing_set.target_shape[0]

        window_len = time_step

        ma = TriangularMovingAverage(window_len=window_len)

        all_crimes = data_group.crimes[:, 0]
        all_targets = data_group.targets
        all_labels = data_group.labels
        all_y_score = ma(all_crimes)
        all_t_range = data_group.t_range

        tst_y_score = all_y_score[-test_set_size:]
        tst_targets = all_targets[-test_set_size:]
        tst_t_range = all_t_range[-test_set_size:]

        trn_y_score = all_y_score[time_step + 1:-test_set_size]  # skip all the nan values
        trn_y_true = all_targets[time_step + 1:-test_set_size]
        trn_y_class = all_labels[time_step + 1:-test_set_size]

        thresh = best_threshold(y_class=trn_y_class,
                                y_score=trn_y_score)  # should only come from the train predictions

        tst_y_true = tst_targets
        tst_y_count = data_group.to_counts(tst_y_true)
        tst_y_score = tst_y_score
        tst_y_pred = get_y_pred(thresh, tst_y_score)  # might want to do this for each cell either?

        tst_y_pred = np.expand_dims(tst_y_pred, axis=1)
        tst_y_score = np.expand_dims(tst_y_score, axis=1)

        log.info(f"======== {conf.model_path}  ========")
        save_metrics(y_count=tst_y_count,
                     y_pred=tst_y_pred,
                     y_score=tst_y_score,
                     t_range=tst_t_range,
                     shaper=data_group.shaper,
                     conf=conf)

        log.info("=====================================END=====================================\n")
        # ------------Triangular Moving Average
