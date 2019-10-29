import pickle

from numpy import ndarray
from pandas.core.indexes.datetimes import DatetimeIndex
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, matthews_corrcoef \
    , precision_recall_curve, roc_curve, recall_score, precision_score

from utils.metrics import PRCurvePlotter, ROCCurvePlotter, PerTimeStepPlotter
from utils.preprocessing import Shaper

import logging as log


class PRCurve:
    def __init__(self, precision, recall, thresholds):
        self.precision, self.recall, self.thresholds = precision, recall, thresholds


class ROCCurve:
    def __init__(self, fpr, tpr, thresholds):
        self.fpr, self.tpr, self.thresholds = fpr, tpr, thresholds


class ModelMetrics:  # short memory light way of comparing models - does not save the actually predictions
    def __init__(self, model_name, y_true, y_pred, probas_pred):
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
    """
    Training the model for a single epoch
    """
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

    # do result plotting and saving
    pr_plotter = PRCurvePlotter()
    pr_plotter.add_curve(y_true.flatten(), probas_pred.flatten(), label_name=conf.model_name)
    pr_plotter.savefig(f"{conf.model_path}plot_pr_curve.png")

    roc_plotter = ROCCurvePlotter()
    roc_plotter.add_curve(y_true.flatten(), probas_pred.flatten(), label_name=conf.model_name)
    roc_plotter.savefig(f"{conf.model_path}plot_roc_curve.png")

    total_crime_plot = PerTimeStepPlotter(time_step=t_range.freqstr, ylabel="Total Crimes")
    total_crime_plot.plot(probas_pred.sum(-1)[:, 0], label="Predicted")
    total_crime_plot.plot(y_true.sum(-1)[:, 0], label="Actual")
    total_crime_plot.savefig(f"{conf.model_path}plot_total_crimes.png")

    # todo map of total metric over time
    # todo map of metric over entire map
