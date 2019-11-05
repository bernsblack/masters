import pickle

from numpy import ndarray
from pandas.core.indexes.datetimes import DatetimeIndex
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, matthews_corrcoef \
    , precision_recall_curve, roc_curve, recall_score, precision_score

from utils.metrics import PRCurvePlotter, ROCCurvePlotter, PerTimeStepPlotter, roc_auc_score_per_time_slot, \
    average_precision_score_per_time_slot, accuracy_score_per_time_slot, precision_score_per_time_slot, \
    recall_score_per_time_slot, safe_f1_score
from utils.preprocessing import Shaper
import os
import numpy as np
import logging as log
import pandas as pd


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

    if len(model_metrics) == 0:
        raise EnvironmentError("No model metrics in this directory")

    return model_metrics


# remember pandas df.to_latex for the columns
def get_metrics_table(model_metrics):
    col = ['Accuracy', 'ROC AUC', 'Avg. Precision', 'Recall', 'Precision', 'F1 Score','Matthews Corrcoef']
    data = []
    names = []
    for m in model_metrics:
        names.append(m.model_name)
        f1 = safe_f1_score((m.average_precision_score, m.recall_score))
        row = [m.accuracy_score, m.roc_auc_score, m.average_precision_score, m.recall_score, m.precision_score,
               f1, m.matthews_corrcoef]
        data.append(row)

    df = pd.DataFrame(columns=col, data=data, index=names)
    df.index.name = "Model Name"

    return df


def compare_models(data_path):
    model_metrics = get_models_metrics(data_path)

    metrics_table = get_metrics_table(model_metrics)
    log.info(f"\n{metrics_table}")

    # pr-curve
    pr_plot = PRCurvePlotter()
    for metric in model_metrics:
        pr_plot.add_curve_(precision=metric.pr_curve.precision,
                           recall=metric.pr_curve.recall,
                           ap=metric.average_precision_score,
                           label_name=metric.model_name)
    pr_plot.show()  # save somewhere?

    # roc-curve
    roc_plot = ROCCurvePlotter()
    for metric in model_metrics:
        roc_plot.add_curve_(fpr=metric.roc_curve.fpr,
                            tpr=metric.roc_curve.tpr,
                            auc=metric.roc_auc_score,
                            label_name=metric.model_name)
    roc_plot.show()  # save somewhere?
    # todd add per cel and time plots per metric - should be saved on the metric class

    return metrics_table  # by returning the table - makes visualisation in notebooks easier


class PRCurve:
    def __init__(self, precision, recall, thresholds):
        self.precision, self.recall, self.thresholds = precision, recall, thresholds


class ROCCurve:
    def __init__(self, fpr, tpr, thresholds):
        self.fpr, self.tpr, self.thresholds = fpr, tpr, thresholds


class ModelMetrics:  # short memory light way of comparing models - does not save the actually predictions
    def __init__(self, model_name, y_true, y_pred, probas_pred):
        ap_per_time = average_precision_score_per_time_slot(y_true=y_true,
                                                            probas_pred=probas_pred)
        self.ap_per_time = np.nan_to_num(ap_per_time)

        roc_per_time = roc_auc_score_per_time_slot(y_true=y_true,
                                                   probas_pred=probas_pred)
        self.roc_per_time = np.nan_to_num(roc_per_time)

        acc_per_time = accuracy_score_per_time_slot(y_true=y_true,
                                                    y_pred=y_pred)
        self.acc_per_time = np.nan_to_num(acc_per_time)

        p_per_time = precision_score_per_time_slot(y_true=y_true,
                                                   y_pred=y_pred)
        self.p_per_time = np.nan_to_num(p_per_time)

        r_per_time = recall_score_per_time_slot(y_true=y_true,
                                                y_pred=y_pred)
        self.r_per_time = np.nan_to_num(r_per_time)

        # flatten array for the next functions
        y_true, y_pred, probas_pred = y_true.flatten(), y_pred.flatten(), probas_pred.flatten()
        self.model_name = model_name
        self.accuracy_score = accuracy_score(y_true, y_pred)
        self.roc_auc_score = roc_auc_score(y_true, probas_pred)

        self.recall_score = recall_score(y_true, y_pred)
        self.precision_score = precision_score(y_true, y_pred)

        self.average_precision_score = average_precision_score(y_true, probas_pred)
        self.matthews_corrcoef = matthews_corrcoef(y_true, y_pred)

        self.pr_curve = PRCurve(*precision_recall_curve(y_true, probas_pred))
        self.roc_curve = ROCCurve(*roc_curve(y_true, probas_pred, drop_intermediate=False))

    def __repr__(self):
        r = rf"""
        MODEL METRICS
            Model Name: {self.model_name}
                ROC AUC:            {self.roc_auc_score}
                Recall:             {self.recall_score}
                Precision:          {self.precision_score}
                Average Precision:  {self.average_precision_score}
                Accuracy:           {self.accuracy_score}
                MCC:                {self.matthews_corrcoef}          
        """

        return r

    def __str__(self):
        return self.__repr__()


class ModelResult:
    def __init__(self, model_name: str, y_true: ndarray, y_pred: ndarray,
                 probas_pred: ndarray, t_range: DatetimeIndex, shaper: Shaper):
        """
        ModelResult: save the data in the actual format we need it (N,C,L)
        Data is saved in a sparse representation - no extra zero values data saved.
        Shaper allows us to re-shape the data to create a map view of the city

        :param model_name: text used to refer to the model on plots
        :param y_true (N,1,L): ground truth value (i.e. did crime happen 1 or not 0) {0,1}
        :param y_pred (N,1,L): the model's hard prediction of the model {0,1}
        :param probas_pred (N,1,L): model floating point values, can be likelihoods [0,1) or count estimates [0,n)
        :param t_range (N,1): range of the times of the test set - also used in plots
        """
        if len(y_true.shape) != 3:
            raise Exception("y_true must be in (N,1,L) format, try reshaping the data.")

        if len(y_pred.shape) != 3:
            raise Exception("y_pred must be in (N,1,L) format, try reshaping the data.")

        if len(probas_pred.shape) != 3:
            raise Exception("probas_pred must be in (N,1,L) format, try reshaping the data.")

        self.model_name = model_name
        self.y_true = y_true
        self.y_pred = y_pred  # remove we're getting y_true from y_pred -> get-best_threshold
        self.probas_pred = probas_pred
        self.t_range = t_range
        self.shaper = shaper

    def accuracy(self):
        return accuracy_score(self.y_true.flatten(), self.y_pred.flatten())

    def recall_score(self):
        return recall_score(self.y_true.flatten(), self.y_pred.flatten())

    def precision_score(self):
        return precision_score(self.y_true.flatten(), self.y_pred.flatten())

    def roc_auc(self):
        return roc_auc_score(self.y_true.flatten(), self.probas_pred.flatten())

    def average_precision(self):
        return average_precision_score(self.y_true.flatten(), self.probas_pred.flatten())

    def matthews_corrcoef(self):
        """
        A coefficient of +1 represents a perfect prediction, 0 an average random prediction and -1 an inverse
         prediction. - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html
        """
        return matthews_corrcoef(self.y_true.flatten(), self.y_pred.flatten())

    def __repr__(self):
        r = rf"""
        MODEL RESULT
            Model Name: {self.model_name}
                ROC AUC:            {self.roc_auc()}
                Recall:             {self.recall_score()}
                Precision:          {self.precision_score()}
                Average Precision:  {self.average_precision()}
                Accuracy:           {self.accuracy()}
                MCC:                {self.matthews_corrcoef()}          
        """

        return r

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

    with open(f"{conf.model_path}model_metric.pkl", "wb") as file:
        pickle.dump(model_metrics, file)
