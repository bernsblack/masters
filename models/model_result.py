import pickle

from numpy import ndarray
from pandas.core.indexes.datetimes import DatetimeIndex
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, matthews_corrcoef \
    , precision_recall_curve, roc_curve, recall_score, precision_score, mean_absolute_error, mean_squared_error

from utils import get_data_sub_paths
from utils.metrics import PRCurvePlotter, ROCCurvePlotter, PerTimeStepPlotter, roc_auc_score_per_time_slot, \
    average_precision_score_per_time_slot, accuracy_score_per_time_slot, precision_score_per_time_slot, \
    recall_score_per_time_slot, safe_f1_score, mae_per_time_slot, rmse_per_time_slot, \
    predictive_accuracy_index_per_time_slot, matthews_corrcoef_per_time_slot, predictive_accuracy_index
from utils.preprocessing import Shaper
import os
import numpy as np
import logging as log
import pandas as pd


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
                "Avg. Precision": m.average_precision_score,
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
    df.sort_values(["dt", "dx", "dy", "start_date", "stop_date", "Avg. Precision"], inplace=True, ascending=False)
    col = ["Model", "dt", "dx", "dy", "start_date", "stop_date", "MAE", "RMSE",
           "ROC AUC", "Avg. Precision", "Precision", "Recall", "F1 Score", "Accuracy", "Matthews Corrcoef"]
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


def get_models_metrics(data_path):
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
def get_metrics_table(models_metrics):
    col = ['RMSE', 'MAE', 'ROC AUC', 'Avg. Precision', 'Precision', 'Recall', 'F1 Score', 'Accuracy',
           'MCC', 'PAI']
    data = []
    names = []
    for m in models_metrics:
        names.append(m.model_name)
        f1 = safe_f1_score((m.precision_score, m.recall_score))
        row = [m.root_mean_squared_error, m.mean_absolute_error, m.roc_auc_score,
               m.average_precision_score, m.precision_score, m.recall_score,
               f1, m.accuracy_score, m.matthews_corrcoef, m.predictive_accuracy_index]
        data.append(row)

    df = pd.DataFrame(columns=col, data=data, index=names)
    df.index.name = "Model Name"
    # df.sort_values('F1 Score', inplace=True, ascending=False)
    df.sort_values('Avg. Precision', inplace=True, ascending=False)

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

    return metrics_table  # by returning the table - makes visualisation in notebooks easier


def compare_all_models(data_path: str):
    """
    :param data_path: string path to the data directory
    :return: pandas data-frame table with all model metrics for all the models found under data_path
    """
    models_metrics = get_models_metrics(data_path)

    metrics_table = get_metrics_table(models_metrics)
    log.info(f"\n{metrics_table}")

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
    # todd add per cel and time plots per metric - should be saved on the metric class

    return metrics_table  # by returning the table - makes visualisation in notebooks easier


class PRCurve:
    def __init__(self, y_class, probas_pred):
        self.precision, self.recall, self.thresholds = precision_recall_curve(y_true=y_class, probas_pred=probas_pred)
        self.ap = average_precision_score(y_true=y_class, y_score=probas_pred)


class ROCCurve:
    def __init__(self, y_class, probas_pred):
        y_class, probas_pred
        self.fpr, self.tpr, self.thresholds = roc_curve(y_true=y_class, y_score=probas_pred, drop_intermediate=False)
        self.auc = roc_auc_score(y_true=y_class, y_score=probas_pred)


def mean_std(a):
    return np.mean(a), np.std(a)


def mean_std_str(m, s):
    return f"{m:.4f} ± {s:.4f}"


class ModelMetrics:  # short memory light way of comparing models - does not save the actually predictions
    def __init__(self, model_name, y_true, y_pred, probas_pred, averaged_over_time=False):
        """

        :param model_name: name of model used in plots
        :param y_true: actual crime counts (N,1,L) [0,inf)
        :param y_pred: predicted hot or not spot {0,1}
        :param probas_pred: probability of hot or not [0,...,1]
        :param averaged_over_time: if the metrics should be calculated per time step and then averaged of them or calculated globally

        Note
        ----
        y_true is converted to y_class by setting all values above zero to 1. y_class is used in calculation of other
        metrics
        """

        # y_true must be in format N,1,L to be able to correctly compare all the models
        if len(y_true.shape) != 3 or y_true.shape[1] != 1:
            raise Exception(f"y_true must be in (N,1,L) not {y_true.shape}.")

        if len(y_pred.shape) != 3 or y_pred.shape[1] != 1:
            raise Exception(f"y_pred must be in (N,1,L) not {y_pred.shape}.")

        if len(probas_pred.shape) != 3 or probas_pred.shape[1] != 1:
            raise Exception(f"probas_pred must be in (N,1,L) not {probas_pred.shape}.")

        self.averaged_over_time = averaged_over_time

        mae_per_time = mae_per_time_slot(y_true=y_true, y_pred=probas_pred)
        self.mae_per_time = np.nan_to_num(mae_per_time)

        # rmse is not intuitive and skews the scores when test samples are few
        rmse_per_time = rmse_per_time_slot(y_true=y_true, y_pred=probas_pred)
        self.rmse_per_time = np.nan_to_num(rmse_per_time)

        y_class = np.copy(y_true)
        y_class[y_class > 0] = 1

        ap_per_time = average_precision_score_per_time_slot(y_true=y_class, probas_pred=probas_pred)
        self.ap_per_time = np.nan_to_num(ap_per_time)

        roc_per_time = roc_auc_score_per_time_slot(y_true=y_class, probas_pred=probas_pred)
        self.roc_per_time = np.nan_to_num(roc_per_time)

        acc_per_time = accuracy_score_per_time_slot(y_true=y_class, y_pred=y_pred)
        self.acc_per_time = np.nan_to_num(acc_per_time)

        p_per_time = precision_score_per_time_slot(y_true=y_class, y_pred=y_pred)
        self.p_per_time = np.nan_to_num(p_per_time)

        r_per_time = recall_score_per_time_slot(y_true=y_class, y_pred=y_pred)
        self.r_per_time = np.nan_to_num(r_per_time)

        pai_per_time = predictive_accuracy_index_per_time_slot(y_true=y_true, y_pred=y_pred)
        self.pai_per_time = np.nan_to_num(pai_per_time)

        mcc_per_time = matthews_corrcoef_per_time_slot(y_true=y_class, y_pred=y_pred)
        self.mcc_per_time = np.nan_to_num(mcc_per_time)

        # flatten array for the next functions
        y_class = y_class.flatten()
        y_true, y_pred, probas_pred = y_true.flatten(), y_pred.flatten(), probas_pred.flatten()

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
            self.roc_auc_score = roc_auc_score(y_true=y_class, y_score=probas_pred)
            self.recall_score = recall_score(y_true=y_class, y_pred=y_pred)
            self.precision_score = precision_score(y_true=y_class, y_pred=y_pred)
            self.average_precision_score = average_precision_score(y_true=y_class, y_score=probas_pred)
            self.matthews_corrcoef = matthews_corrcoef(y_true=y_class, y_pred=y_pred)
            self.predictive_accuracy_index = predictive_accuracy_index(y_true=y_true, y_pred=y_pred)
            # treats all errors linearly
            self.mean_absolute_error = mean_absolute_error(y_true=y_true, y_pred=probas_pred)
            # penalises large variations in error - high weight to large errors - great for training, not intuitive, when comparing models especially when the number of test samples can become large.
            self.root_mean_squared_error = np.sqrt(mean_squared_error(y_true=y_true, y_pred=probas_pred))

        """
        Extra info on MAE vs RMSE:
        MAE <= RMSE: if all errors are equal, e.g. only guessing between one and zero, error is max 1.
        RMSE <= (MAE * sqrt(n_samples)):  usually occurs when the n_samples is small.   
        """

        self.pr_curve = PRCurve(y_class=y_class, probas_pred=probas_pred)
        self.roc_curve = ROCCurve(y_class=y_class, probas_pred=probas_pred)

    def __repr__(self):
        if self.averaged_over_time:
            r = rf"""
            MODEL METRICS (Averaged over Time Steps)
                Model Name: {self.model_name}
                    MAE:                {mean_std_str(*self.mean_absolute_error)}
                    RMSE:               {mean_std_str(*self.root_mean_squared_error)}
                    ROC AUC:            {mean_std_str(*self.roc_auc_score)}                
                    Average Precision:  {mean_std_str(*self.average_precision_score)}
                    Precision:          {mean_std_str(*self.precision_score)}
                    Recall:             {mean_std_str(*self.recall_score)}
                    Accuracy:           {mean_std_str(*self.accuracy_score)}
                    PAI:                {mean_std_str(*self.predictive_accuracy_index)}
                    MCC:                {mean_std_str(*self.matthews_corrcoef)}          
            """
        else:
            r = rf"""
            MODEL METRICS (Over All Samples)
                Model Name: {self.model_name}
                    MAE:                {self.mean_absolute_error:.5f}
                    RMSE:               {self.root_mean_squared_error:.5f}
                    ROC AUC:            {self.roc_auc_score:.5f}                
                    Average Precision:  {self.average_precision_score:.5f}
                    Precision:          {self.precision_score:.5f}
                    Recall:             {self.recall_score:.5f}
                    Accuracy:           {self.accuracy_score:.5f}
                    PAI:                {self.predictive_accuracy_index:.5f}
                    MCC:                {self.matthews_corrcoef:.5f}          
            """

        return r

    def __str__(self):
        return self.__repr__()


class ModelResult:
    def __init__(self, model_name: str, y_true: ndarray, y_pred: ndarray,
                 probas_pred: ndarray, t_range: DatetimeIndex, shaper: Shaper, averaged_over_time=False):
        """
        ModelResult: save the data the model predicted in format (N,C,L)
        Data is saved in a sparse representation - no extra zero values data saved.
        Shaper allows us to re-shape the data to create a map view of the city

        :param model_name: text used to refer to the model on plots
        :param y_true (N,1,L): ground truth the amount of crime occurring on this spot, gets converted into y_class {0,1}
        :param y_pred (N,1,L): the model's hard prediction of the model {0,1}
        :param probas_pred (N,1,L): model floating point values, can be likelihoods [0,1) or count estimates [0,n)
        :param t_range (N,1): range of the times of the test set - also used in plots
        """
        if len(y_true.shape) != 3 or y_true.shape[1] != 1:
            raise Exception(f"y_true must be in (N,1,L) not {y_true.shape}.")

        if len(y_pred.shape) != 3 or y_pred.shape[1] != 1:
            raise Exception(f"y_pred must be in (N,1,L) not {y_pred.shape}.")

        if len(probas_pred.shape) != 3 or probas_pred.shape[1] != 1:
            raise Exception(f"probas_pred must be in (N,1,L) not {probas_pred.shape}.")

        self.averaged_over_time = averaged_over_time

        y_class = np.copy(y_true)
        y_class[y_class > 0] = 1

        self.model_name = model_name
        self.y_true = y_true
        self.y_class = y_class
        self.y_pred = y_pred  # remove we're getting y_true from y_pred -> get-best_threshold
        self.probas_pred = probas_pred
        self.t_range = t_range
        self.shaper = shaper

        self.metrics = ModelMetrics(model_name=self.model_name,
                                    y_true=self.y_true,
                                    y_pred=self.y_class,
                                    probas_pred=self.y_pred,
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


def save_metrics(y_true, y_pred, probas_pred, t_range, shaper, conf):
    # save result
    # only saves the result of the metrics not the predicted values
    model_metrics = ModelMetrics(model_name=conf.model_name,
                                 y_true=y_true,
                                 y_pred=y_pred,
                                 probas_pred=probas_pred)
    log.info(model_metrics)

    with open(f"{conf.model_path}model_metric.pkl", "wb") as file:
        pickle.dump(model_metrics, file)


def save_results(y_true, y_pred, probas_pred, t_range, shaper, conf):
    # saves the actual target and predicted values to be visualised later on - the one we're actually going to be using
    model_result = ModelResult(model_name=conf.model_name,
                               y_true=y_true,
                               y_pred=y_pred,
                               probas_pred=probas_pred,
                               t_range=t_range,
                               shaper=shaper)
    log.info(model_result)

    with open(f"{conf.model_path}model_result.pkl", "wb") as file:
        pickle.dump(model_result, file)
