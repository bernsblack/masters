import logging as log
import numpy as np
import os
import pandas as pd
import pickle
from IPython.core.display import display
from numpy import ndarray
from pandas.core.indexes.datetimes import DatetimeIndex
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, matthews_corrcoef \
    , precision_recall_curve, roc_curve, recall_score, precision_score, mean_absolute_error, mean_squared_error
from typing import List, Optional

from utils import get_data_sub_paths
from utils.forecasting import mean_absolute_scaled_error, root_mean_squared_error
from utils.metrics import PRCurvePlotter, ROCCurvePlotter, roc_auc_score_per_time_slot, \
    average_precision_score_per_time_slot, accuracy_score_per_time_slot, precision_score_per_time_slot, \
    recall_score_per_time_slot, safe_f1_score, mae_per_time_slot, rmse_per_time_slot, \
    predictive_accuracy_index_per_time_slot, matthews_corrcoef_per_time_slot, predictive_accuracy_index, det_curve, \
    DETCurvePlotter, ndcg_per_time_slot
from utils.preprocessing import Shaper, scale_per_time_slot
from utils.utils import is_all_integer


class PRCurve:
    def __init__(self, y_class, y_score):
        self.precision, self.recall, self.thresholds = precision_recall_curve(y_true=y_class, probas_pred=y_score)
        self.ap = average_precision_score(y_true=y_class, y_score=y_score)


class ROCCurve:
    def __init__(self, y_class, y_score):
        self.fpr, self.tpr, self.thresholds = roc_curve(y_true=y_class, y_score=y_score, drop_intermediate=False)
        self.auc = roc_auc_score(y_true=y_class, y_score=y_score)


from utils.metrics import compute_eer


class DETCurve:
    def __init__(self, y_class, y_score):
        self.fpr, self.fnr, self.thresholds = det_curve(y_true=y_class, y_score=y_score)
        self.eer, self.eer_thresh = compute_eer(fpr=self.fpr, fnr=self.fnr, thresholds=self.thresholds)


def mean_std(a):
    a = drop_nan(a)
    return np.mean(a), np.std(a)


def mean_std_str(m, s):
    return f"{m:.4f} ± {s:.4f}"


from utils.utils import drop_nan


class RankingMetrics:
    def __init__(self, model_name, y_count, y_score, t_range):
        """
        RankingMetrics scales y_count, y_score to min=0, max=1 for each time slot. Then performs certain metrics on
        the scaled data.

        :param model_name: name of model used in plots
        :param y_count: true crime counts (N,1,L) [0,inf) # should be scaled before - use data_group.to_counts()s
        :param y_score: (y_score) probability of hot or not [0,1] for classification models and estimated count for regression models

        Note
        ----
        y_count is converted to y_class by setting all values above zero to 1. y_class is used in calculation of other
        metrics
        """
        self.model_name = model_name

        # y_true must be in format N,1,L to be able to correctly compare all the models
        if len(y_count.shape) != 3 or y_count.shape[1] != 1:
            raise Exception(f"y_count must be in (N,1,L) not {y_count.shape}.")

        if len(y_score.shape) != 3 or y_score.shape[1] != 1:
            raise Exception(f"y_score must be in (N,1,L) not {y_score.shape}.")

        self.t_range = t_range

        # by scaling we can determine the goodness of distribution for each time step
        y_count = scale_per_time_slot(y_count)  # scaled each time step to have max 1 and min 0
        y_score = scale_per_time_slot(y_score)  # scaled each time step to have max 1 and min 0

        ndcg_per_time = ndcg_per_time_slot(y_count=y_count, y_score=y_score)
        self.mean_ndcg_score = np.mean(drop_nan(ndcg_per_time))
        # self.ndcg_per_time = np.nan_to_num(ndcg_per_time)
        self.ndcg_per_time = ndcg_per_time

        mae_per_time = mae_per_time_slot(y_count=y_count, y_score=y_score)
        self.mean_mae = np.mean(drop_nan(mae_per_time))
        # self.mae_per_time = np.nan_to_num(mae_per_time)
        self.mae_per_time = mae_per_time

        y_class = np.copy(y_count)
        y_class[y_class > 0] = 1

        ap_per_time = average_precision_score_per_time_slot(y_class=y_class, y_score=y_score)
        self.mean_average_precision = np.mean(drop_nan(ap_per_time))
        # self.ap_per_time = np.nan_to_num(ap_per_time)
        self.ap_per_time = ap_per_time

        self.class_count_0 = len(y_class[y_class == 0].flatten())
        self.class_count_1 = len(y_class[y_class == 1].flatten())

    def __repr__(self):
        r = rf"""
        MODEL NORMALIZED RANKING METRICS
            Class Balance (Crime:No-Crime) - 1:{self.class_count_0 / self.class_count_1:.3f}
            Model Name: {self.model_name}
                NDCG:      {self.mean_ndcg_score:.6f}
                MAP:       {self.mean_average_precision:.6f}
                MAE:       {self.mean_mae:.6f}
        """
        return r


class RegressionsMetrics:  # short memory light way of comparing models - does not save the actually predictions
    def __init__(self, model_name, y_count, y_pred, y_score, t_range=None, averaged_over_time=False):
        pass


class ClassificationMetrics:  # short memory light way of comparing models - does not save the actually predictions
    def __init__(self, model_name, y_count, y_pred, y_score, t_range=None, averaged_over_time=False):
        pass


class ModelMetrics:  # short memory light way of comparing models - does not save the actually predictions
    def __init__(self, model_name, y_count, y_pred, y_score, t_range=None, averaged_over_time=False):
        """

        :param model_name: name of model used in plots
        :param y_count: true crime counts (N,1,L) [0,inf) # should be scaled before - use data_group.to_counts()s
        :param y_pred: predicted hot or not spot {0,1}
        :param y_score: (probas_pred or y_score) probability of hot or not [0,1] for classification models and estimated count for regression models
        :param averaged_over_time: if the metrics should be calculated per time step and then averaged of them or calculated globally

        Note
        ----
        y_true is converted to y_class by setting all values above zero to 1. y_class is used in calculation of other
        metrics
        """

        # y_true must be in format N,1,L to be able to correctly compare all the models
        if len(y_count.shape) != 3 or y_count.shape[1] != 1:
            raise Exception(f"y_count must be in (N,1,L) not {y_count.shape}.")

        if len(y_pred.shape) != 3 or y_pred.shape[1] != 1:
            raise Exception(f"y_pred must be in (N,1,L) not {y_pred.shape}.")

        if len(y_score.shape) != 3 or y_score.shape[1] != 1:
            raise Exception(f"y_score must be in (N,1,L) not {y_score.shape}.")

        if not is_all_integer(y_count):
            raise Exception(f"y_count must be all integers: representing the true counts")

        if np.max(y_count) <= 1:
            raise Exception(f"y_count is the true count of crimes: max value ({np.max(y_count)}) should be above 1")

        self.t_range = t_range

        self.averaged_over_time = averaged_over_time

        # by scaling we can determine the goodness of distribution for each time step
        y_count_scl = scale_per_time_slot(y_count)  # scaled each time step to have max 1 and min 0
        y_score_scl = scale_per_time_slot(y_score)  # scaled each time step to have max 1 and min 0

        ndcg_per_time = ndcg_per_time_slot(y_count=y_count_scl, y_score=y_score_scl)
        self.ndcg_per_time = np.nan_to_num(ndcg_per_time)
        self.ndcg = np.mean(self.ndcg_per_time)

        mae_per_time_scl = mae_per_time_slot(y_count=y_count_scl, y_score=y_score_scl)
        self.mae_scl = np.mean(mae_per_time_scl)

        mae_per_time = mae_per_time_slot(y_count=y_count, y_score=y_score)
        self.mae_per_time = mae_per_time

        # rmse is not intuitive and skews the scores when test samples are few
        rmse_per_time = rmse_per_time_slot(y_count=y_count, y_score=y_score)
        self.rmse_per_time = rmse_per_time

        pai_per_time = predictive_accuracy_index_per_time_slot(y_count=y_count, y_pred=y_pred)
        self.pai_per_time = pai_per_time

        y_class = np.copy(y_count)
        y_class[y_class > 0] = 1

        ap_per_time = average_precision_score_per_time_slot(y_class=y_class, y_score=y_score)
        self.ap_per_time = ap_per_time
        self.mean_ap = np.mean(drop_nan(ap_per_time))

        roc_per_time = roc_auc_score_per_time_slot(y_class=y_class, y_score=y_score)
        self.roc_per_time = roc_per_time

        acc_per_time = accuracy_score_per_time_slot(y_class=y_class, y_pred=y_pred)
        self.acc_per_time = acc_per_time

        p_per_time = precision_score_per_time_slot(y_class=y_class, y_pred=y_pred)
        self.p_per_time = p_per_time

        r_per_time = recall_score_per_time_slot(y_class=y_class, y_pred=y_pred)
        self.r_per_time = r_per_time

        mcc_per_time = matthews_corrcoef_per_time_slot(y_class=y_class, y_pred=y_pred)
        self.mcc_per_time = mcc_per_time

        # flatten array for the next functions
        y_class = y_class.flatten()
        y_count, y_pred, y_score = y_count.flatten(), y_pred.flatten(), y_score.flatten()

        self.model_name = model_name
        if self.averaged_over_time:  # false by default
            self.accuracy_score = mean_std(self.acc_per_time)
            self.roc_auc_score = mean_std(self.roc_per_time)
            self.recall_score = mean_std(r_per_time)
            self.precision_score = mean_std(p_per_time)
            self.average_precision_score = mean_std(ap_per_time)
            self.matthews_corrcoef = mean_std(mcc_per_time)
            self.predictive_accuracy_index = mean_std(pai_per_time)
            # treats all errors linearly
            self.mean_absolute_error = mean_std(mae_per_time)
            # penalises large variations in error - high weight to large errors - great for training, not intuitive, when comparing models especially when the number of test samples can become large.
            self.root_mean_squared_error = mean_std(np.sqrt(mae_per_time))
        else:
            self.accuracy_score = accuracy_score(y_true=y_class, y_pred=y_pred)
            self.roc_auc_score = roc_auc_score(y_true=y_class, y_score=y_score)
            self.recall_score = recall_score(y_true=y_class, y_pred=y_pred)
            self.precision_score = precision_score(y_true=y_class, y_pred=y_pred)
            self.average_precision_score = average_precision_score(y_true=y_class, y_score=y_score)
            self.matthews_corrcoef = matthews_corrcoef(y_true=y_class, y_pred=y_pred)
            self.predictive_accuracy_index = predictive_accuracy_index(y_count=y_count, y_pred=y_pred)
            # treats all errors linearly
            self.mean_absolute_error = mean_absolute_error(y_true=y_count, y_pred=y_score)
            # penalises large variations in error - high weight to large errors - great for training, not intuitive, when comparing models especially when the number of test samples can become large.
            self.root_mean_squared_error = np.sqrt(mean_squared_error(y_true=y_count, y_pred=y_score))

        """
        Extra info on MAE vs RMSE:
        MAE <= RMSE: if all errors are equal, e.g. only guessing between one and zero, error is max 1.
        RMSE <= (MAE * sqrt(n_samples)):  usually occurs when the n_samples is small.   
        """

        self.pr_curve = PRCurve(y_class=y_class, y_score=y_score)
        self.roc_curve = ROCCurve(y_class=y_class, y_score=y_score)
        self.det_curve = DETCurve(y_class=y_class, y_score=y_score)

        self.class_count_0 = len(y_class[y_class == 0].flatten())
        self.class_count_1 = len(y_class[y_class == 1].flatten())

    def __repr__(self):
        if self.averaged_over_time:
            r = rf"""
            MODEL METRICS (Averaged over Time Steps)
                Class Balance (Crime:No-Crime) - 1:{self.class_count_0 / self.class_count_1:.3f}
                Model Name: {self.model_name}
                    NDCG:               {self.ndcg:.5f}                      
                    ROC AUC:            {mean_std_str(*self.roc_auc_score)}                
                    Average Precision:  {mean_std_str(*self.average_precision_score)}
                    Precision:          {mean_std_str(*self.precision_score)}
                    Recall:             {mean_std_str(*self.recall_score)}
                    Accuracy:           {mean_std_str(*self.accuracy_score)}
                    PAI:                {mean_std_str(*self.predictive_accuracy_index)}
                    MCC:                {mean_std_str(*self.matthews_corrcoef)}
                    MAE:                {mean_std_str(*self.mean_absolute_error)}
                    RMSE:               {mean_std_str(*self.root_mean_squared_error)}         
            """
        else:
            r = rf"""
            MODEL METRICS (Over All Samples)
            Class Balance (Crime:No-Crime) - 1:{self.class_count_0 / self.class_count_1:.3f}
                Model Name: {self.model_name}
                    NDCG:               {self.ndcg:.5f}                
                    ROC AUC:            {self.roc_auc_score:.5f}                
                    Average Precision:  {self.average_precision_score:.5f}
                    Precision:          {self.precision_score:.5f}
                    Recall:             {self.recall_score:.5f}
                    Accuracy:           {self.accuracy_score:.5f}
                    MCC:                {self.matthews_corrcoef:.5f}
                    PAI:                {self.predictive_accuracy_index:.5f}
                    MAE:                {self.mean_absolute_error:.5f}
                    RMSE:               {self.root_mean_squared_error:.5f}      
            """

        return r

    def __str__(self):
        return self.__repr__()


class ModelResult:
    def __init__(self, model_name: str, y_count: ndarray, y_pred: ndarray, y_score: ndarray, t_range: DatetimeIndex,
                 shaper: Optional[Shaper], averaged_over_time: bool = False):
        """
        ModelResult: save the data the model predicted in format (N,C,L)
        Data is saved in a sparse representation - no extra zero values data saved.
        Shaper allows us to re-shape the data to create a map view of the city

        :param model_name: text used to refer to the model on plots
        :param y_count (N,1,L): ground truth the amount of crime occurring on this spot [0, inf), gets converted into y_class {0,1}
        :param y_pred (N,1,L): the model's hard prediction of the model {0,1}
        :param y_score (N,1,L): model floating point values, can be likelihoods [0,1) or count estimates [0,n)
        :param t_range (N,1): range of the times of the test set - also used in plots
        :param shaper: used to convert saved data into a sparse/grid format
        :param averaged_over_time: if the metrics should be averaged over time or calculated as a global bag of data
        """
        if len(y_count.shape) != 3 or y_count.shape[1] != 1:
            raise Exception(f"y_count must be in (N,1,L) not {y_count.shape}.")

        if len(y_pred.shape) != 3 or y_pred.shape[1] != 1:
            raise Exception(f"y_pred must be in (N,1,L) not {y_pred.shape}.")

        if len(y_score.shape) != 3 or y_score.shape[1] != 1:
            raise Exception(f"y_score must be in (N,1,L) not {y_score.shape}.")

        if not is_all_integer(y_count):
            raise Exception(f"y_count must be all integers: representing the true counts")

        self.averaged_over_time = averaged_over_time

        y_class = np.copy(y_count)
        y_class[y_class > 0] = 1

        self.model_name = model_name
        self.y_true = y_count
        self.y_class = y_class
        self.y_pred = y_pred  # remove we're getting y_true from y_pred -> get-best_threshold
        self.y_score = y_score
        self.t_range = t_range
        self.shaper = shaper

        self.metrics = ModelMetrics(model_name=self.model_name,
                                    y_count=self.y_true,
                                    y_pred=self.y_pred,
                                    y_score=self.y_score,
                                    averaged_over_time=averaged_over_time)

    def accuracy(self):
        return self.metrics.accuracy_score

    def recall_score(self):
        return self.metrics.recall_score

    def precision_score(self):
        return self.metrics.precision_score

    def roc_auc(self):
        return self.metrics.roc_auc_score

    def average_precision(self):
        return self.metrics.average_precision_score

    def matthews_corrcoef(self):
        """
        A coefficient of +1 represents a perfect prediction, 0 an average random prediction and -1 an inverse
         prediction. - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html

        """
        return self.metrics.matthews_corrcoef

    def predictive_accuracy_index(self):
        return self.metrics.predictive_accuracy_index

    def __repr__(self):
        return repr(self.metrics)

    def __str__(self):
        return self.__repr__()


def save_ranking_metrics(y_count, y_score, t_range, conf):
    """
    Only save the data in a metric object which is lighter than results that save all results to disk.
    Metrics being saved are related to how well the model ranks important cells for a given time slot

    :param y_count: true crime counts scaled to original values
    :param y_score: floating point value predictions (probas_pred or y_score)
    :param t_range: date time range of metrics
    :param conf: configuration object
    :return:
    """
    # save result
    # only saves the result of the metrics not the predicted values
    ranking_metrics = RankingMetrics(
        model_name=conf.model_name,
        y_count=y_count,
        y_score=y_score,
        t_range=t_range,
    )
    log.info(ranking_metrics)

    with open(f"{conf.model_path}ranking_metrics.pkl", "wb") as file:
        pickle.dump(ranking_metrics, file)


def save_metrics(y_count, y_pred, y_score, t_range, shaper, conf):
    """
    Only save the data in a metric object which is lighter than results that save all results to disk

    :param y_count: true crime counts scaled to original values
    :param y_pred: hot or not hard predictions
    :param y_score: floating point value predictions (probas_pred or y_score)
    :param t_range: date time range of metrics
    :param shaper: shaper used to unsqueeze results
    :param conf: configuration object
    :return:
    """
    # save result
    # only saves the result of the metrics not the predicted values
    model_metrics = ModelMetrics(model_name=conf.model_name,
                                 y_count=y_count,
                                 y_pred=y_pred,
                                 y_score=y_score)
    log.info(model_metrics)

    with open(f"{conf.model_path}model_metric.pkl", "wb") as file:
        pickle.dump(model_metrics, file)


def save_results(y_count, y_pred, y_score, t_range, shaper, conf):
    """
    Save all predicted results to disk with metrics embedded in result type

    :param y_count: true crime counts scaled to original values
    :param y_pred: hot or not hard predictions
    :param y_score: floating point value predictions (probas_pred or y_score)
    :param t_range: date time range of metrics
    :param shaper: shaper used to unsqueeze results
    :param conf: configuration object
    :return:
    """
    # saves the actual target and predicted values to be visualised later on - the one we're actually going to be using
    model_result = ModelResult(model_name=conf.model_name,
                               y_count=y_count,
                               y_pred=y_pred,
                               y_score=y_score,
                               t_range=t_range,
                               shaper=shaper)
    log.info(model_result)

    with open(f"{conf.model_path}model_result.pkl", "wb") as file:
        pickle.dump(model_result, file)


# remember pandas df.to_latex for the columns
def parse_data_path(s):
    dt, s = s.split("T")[-1].split('H')
    dx, dy, s = s.split("X")[-1].split('M')
    _, dy = dy.split("Y")
    _, start_date, end_date = s.split("_")
    r = {"dt": int(dt),
         "dx": int(dx),
         "dy": int(dy),
         "start_date": start_date,
         "stop_date": end_date}
    return r


def get_all_metrics():
    """
    """
    data = []
    paths = get_data_sub_paths()
    for data_sub_path in paths:
        data_path = f"./data/processed/{data_sub_path}/"
        models_metrics = get_models_metrics(data_path)
        for m in models_metrics:
            row = {
                **parse_data_path(data_sub_path),
                "Model": m.model_name,
                "RMSE": m.root_mean_squared_error,
                "MAE": m.mean_absolute_error,
                "ROC AUC": m.roc_auc_score,
                "AP": m.average_precision_score,
                "Precision": m.precision_score,
                "Recall": m.recall_score,
                "F1 Score": safe_f1_score((m.precision_score, m.recall_score)),
                "Accuracy": m.accuracy_score,
                "MCC": m.matthews_corrcoef,
                "PAI": m.predictive_accuracy_index,
            }
            data.append(row)

    df = pd.DataFrame(data)
    #     df.index.name = "Model Name"
    # df.sort_values("F1 Score", inplace=True, ascending=False)
    df.sort_values(["dt", "dx", "dy", "start_date", "stop_date", "AP"], inplace=True, ascending=False)
    col = ["Model", "dt", "dx", "dy", "start_date", "stop_date", "MAE", "RMSE",
           "ROC AUC", "AP", "Precision", "Recall", "F1 Score", "Accuracy", "Matthews Corrcoef"]
    return df[col]


def get_models_results(data_path):
    """
    Reads all model results give the path to a certain data-source/discretisation
    :param data_path: path to a certain data-source/discretisation
    :return: list of model metrics for the data discretisation
    """
    model_results = []
    model_names = os.listdir(f"{data_path}models")

    if '.DS_Store' in model_names:
        model_names.remove('.DS_Store')

    for model_name in model_names:
        if not os.path.exists(data_path):
            raise Exception(f"Directory ({data_path}) needs to exist.")

        model_path = f"{data_path}models/{model_name}/"

        file_name = f"{model_path}model_result.pkl"

        if not os.path.exists(file_name):
            continue

        with open(file_name, 'rb') as file_pointer:
            model_results.append(pickle.load(file_pointer))

    if len(model_results) == 0:
        raise EnvironmentError("No model results in this directory")

    return model_results


def get_model_result(model_path):
    """
    Reads all model results give the path to a certain data-source/discretisation
    :param model_path: path to a certain data-source/discretisation
    :return: model metric for the data discretisation
    """
    file_path = f"{model_path}/model_result.pkl"

    if not os.path.exists(file_path):
        raise Exception(f"File '{file_path}' does not exist.")

    model_result = None
    with open(file_path, 'rb') as file_pointer:
        model_result = pickle.load(file_pointer)

    return model_result


def get_models_metrics(data_path: str):
    """
    Reads all model metrics give the path to a certain data-source/discretisation
    :param data_path: path to a certain data-source/discretisation
    :return: list of model metrics for the data discretisation
    """
    model_metrics = []
    model_names = os.listdir(f"{data_path}models")

    if '.DS_Store' in model_names:
        model_names.remove('.DS_Store')

    for model_name in model_names:
        if not os.path.exists(data_path):
            raise Exception(f"Directory ({data_path}) needs to exist.")

        model_path = f"{data_path}models/{model_name}/"

        file_name = f"{model_path}model_metric.pkl"

        if not os.path.exists(file_name):
            continue

        with open(file_name, 'rb') as file_pointer:
            model_metrics.append(pickle.load(file_pointer))

    # if len(model_metrics) == 0:
    #     raise EnvironmentError("No model metrics in this directory")

    return model_metrics


# remember pandas df.to_latex for the columns
def get_metrics_table(models_metrics: List[ModelMetrics]):
    col = [
        'NDCG',
        'MAP',
        'ROC AUC',
        'AP',
        'Precision',
        'Recall',
        'F1 Score',
        'Accuracy',
        'MCC',
        'PAI',
        'RMSE',
        'MAE',
    ]
    data = []
    names = []
    for m in models_metrics:
        names.append(m.model_name)
        f1 = safe_f1_score((m.precision_score, m.recall_score))
        row = [
            m.ndcg,
            m.mean_ap,
            m.roc_auc_score,
            m.average_precision_score,
            m.precision_score,
            m.recall_score,
            f1,
            m.accuracy_score,
            m.matthews_corrcoef,
            m.predictive_accuracy_index,
            m.root_mean_squared_error,
            m.mean_absolute_error,
        ]
        data.append(row)

    df = pd.DataFrame(columns=col, data=data, index=names)
    df.index.name = "Model Name"
    # df.sort_values('F1 Score', inplace=True, ascending=False)
    df.sort_values('NDCG', inplace=True, ascending=False)

    return df


def compare_models(models_metrics):
    """
    Allows us to filter certain model metrics and only compare their values instead of comparing all the metrics

    :param models_metrics: list of model metrics
    :return: pandas data-frame table with all model metrics for all the models found under data_path
    """

    metrics_table = get_metrics_table(models_metrics)
    log.info(f"\n{metrics_table}")

    # pr-curve
    pr_plot = PRCurvePlotter()
    for metric in models_metrics:
        pr_plot.add_curve_(precision=metric.pr_curve.precision,
                           recall=metric.pr_curve.recall,
                           ap=metric.pr_curve.ap,
                           label_name=metric.model_name)
    pr_plot.show()  # todo - save somewhere?

    # roc-curve
    roc_plot = ROCCurvePlotter()
    for metric in models_metrics:
        roc_plot.add_curve_(fpr=metric.roc_curve.fpr,
                            tpr=metric.roc_curve.tpr,
                            auc=metric.roc_curve.auc,  # note this is the mean roc_score not this curves score
                            label_name=metric.model_name)
    roc_plot.show()  # todo - save somewhere?
    # todd add per cel and time plots per metric - should be saved on the metric class

    # det-curve
    det_plot = DETCurvePlotter()
    for metric in models_metrics:
        det_plot.add_curve_(fpr=metric.det_curve.fpr,
                            fnr=metric.det_curve.fnr,
                            eer=metric.det_curve.eer,  # note this is the mean roc_score not this curves score
                            label_name=metric.model_name)
    det_plot.show()  # todo - save somewhere?

    return metrics_table  # by returning the table - makes visualisation in notebooks easier


def compare_all_models(data_path: str):
    """
    :param data_path: string path to the data directory
    :return: pandas data-frame table with all model metrics for all the models found under data_path
    """
    models_metrics = get_models_metrics(data_path)

    metrics_table = get_metrics_table(models_metrics)
    # log.info(f"\n{metrics_table}")
    display(metrics_table)

    # pr-curve
    pr_plot = PRCurvePlotter()
    for metric in models_metrics:
        pr_plot.add_curve_(precision=metric.pr_curve.precision,
                           recall=metric.pr_curve.recall,
                           ap=metric.pr_curve.ap,
                           label_name=metric.model_name)
    pr_plot.show()  # save somewhere?

    # roc-curve
    roc_plot = ROCCurvePlotter()
    for metric in models_metrics:
        roc_plot.add_curve_(fpr=metric.roc_curve.fpr,
                            tpr=metric.roc_curve.tpr,
                            auc=metric.roc_curve.auc,  # note this is the mean roc_score not this curves score
                            label_name=metric.model_name)
    roc_plot.show()  # save somewhere?
    # todo add per cel and time plots per metric - should be saved on the metric class

    # det-curve
    det_plot = DETCurvePlotter()
    for metric in models_metrics:
        det_plot.add_curve_(fpr=metric.det_curve.fpr,
                            fnr=metric.det_curve.fnr,
                            eer=metric.det_curve.eer,  # note this is the mean roc_score not this curves score
                            label_name=metric.model_name)
    det_plot.show()  # somewhere?

    return metrics_table  # by returning the table - makes visualisation in notebooks easier


class SequenceResult:
    def __init__(
            self,
            model_name: str,
            y_count: ndarray,
            y_pred: ndarray,
            y_score: ndarray,
            t_range: DatetimeIndex,
    ):
        """

        :param model_name: Model name used in metrics tables and plots
        :param y_count: actual crime count - used in certain metrics, easily interpretable for humans
        :param y_pred: predicted class value (integer values)
        :param y_score: (float values) regression values - can be probabilities of class = 1 too
        :param t_range: date time range used to line up results between various models
        """
        self.model_name = model_name
        self.y_count = y_count  # (actual crime )
        self.y_pred = y_pred  # class predictions (integers)
        self.y_score = y_score  # predicted floating point score
        self.t_range = t_range  # needed to ensure that models have the same start and stop times


class SequenceMetrics:
    def __init__(self, y_true: ndarray, y_score: ndarray, offset=1):
        self.metrics = {
            'MASE': mean_absolute_scaled_error(y_true, y_score, offset),
            #         'MFE': mean_forecast_error(y_true, y_score),
            'MAE': mean_absolute_error(y_true, y_score),
            'MSE': mean_squared_error(y_true, y_score),
            'RMSE': root_mean_squared_error(y_true, y_score),
            #         'MAPE': mean_average_percentage_error(y_true, y_score),
        }
