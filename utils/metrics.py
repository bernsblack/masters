#!python3
import numpy as np
import torch
from utils.data_processing import get_times

"""
Not many of these metrics can be imported straight from sklearn.metrics, e.g.:
- roc_curve
- auc 
- confusion_matrix 
- mutual_info_score
- f1_score 
- jaccard_score (Defined: size of the intersection divided by the size of the union of two label sets)
- roc_auc_score 
- accuracy_score 
- precision_recall_curve
- mean_absolute_error 
- precision_score 
- recall_score
- average_precision_score
"""

from utils.data_processing import get_times
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import logging as log
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve, \
    mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, \
    matthews_corrcoef

import unittest


def predictive_accuracy_index_averaged_over_time(y_true, y_pred):
    """
    Calculate PAI metrics averaged of each observation.

    :param y_true: ndarray (N,L)
    :param y_pred: ndarray (N,L)
    :return: mean and standard deviation of the PAI averaged over time (axis=0)
    """
    result = predictive_accuracy_index_per_time_slot(y_true, y_pred)
    return np.mean(result), np.std(result)


def predictive_accuracy_index_per_time_slot(y_count, y_pred):
    """
    Calculate the PAI for each index of time in predict values of format (N,L)

    :param y_count: true crime counts in (N,1,L) format each value [0, max_int]
    :param y_pred: predicted crime hotspots in (N,1,L) format, each value [0,1]
    :return: ndarray (N,1) with each index in N indicating the PAI for that index
    """

    pai_time_slot = []
    N, C, L = y_count.shape
    assert C == 1
    assert y_count.shape == y_pred.shape

    for n in range(N):
        pai_time_slot.append(
            predictive_accuracy_index(y_count[n, 0, :], y_pred[n, 0, :])
        )

    return np.array(pai_time_slot)


def predictive_accuracy_index(y_true, y_pred):
    """
    y_true: true crime counts of the grid (L,) with each cell value [0,max_int]
    y_pred: predicted hotspot areas in the grid -> 1 for hot 0 for not (L,) with each value [0 or 1]

    Warning: if no crime is predicted a/A will be zero and will lead to ZeroDivisionError
    """
    assert y_pred.max() == 1 or y_pred.max() == 0  # make sure if prediction is made at least the value is one

    n = np.sum(y_true[y_pred == 1])  # crimes in predicted area
    N = np.sum(y_true)  # crimes in total area

    a = np.sum(y_pred)  # area of hotspots
    A = np.product(y_pred.shape)

    if N == 0 or a == 0:  # fail safe in-case now crime occurred on the day or model predicted zero crimes
        return 0

    return (n * A) / (N * a)


Y_TRUE_PAI = np.array([
    [1, 0, 0, 1],
    [0, 0, 0, 0],
    [0, 2, 0, 0],
    [3, 0, 5, 0],
])

Y_PRED_PAI = np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 1, 1, 1],
    [0, 0, 1, 0],
])


class TestPredictiveAccuracyIndex(unittest.TestCase):

    def test_predictive_accuracy_index(self):
        y_true = Y_TRUE_PAI

        y_pred = Y_PRED_PAI

        pai = predictive_accuracy_index(y_true=y_true, y_pred=y_pred)

        self.assertAlmostEqual(pai, 2.1333333333333333)

    def test_predictive_accuracy_index_per_time(self):
        y_true = np.stack(10 * [Y_TRUE_PAI], axis=0).reshape(10, 1, -1)
        y_pred = np.stack(10 * [Y_PRED_PAI], axis=0).reshape(10, 1, -1)

        pai_per_t = predictive_accuracy_index_per_time_slot(y_count=y_true, y_pred=y_pred)
        target = np.array(10 * [2.1333333333333333])
        self.assertTrue(np.equal(pai_per_t, target).all())

    def test_predictive_accuracy_index_averaged_over_time(self):
        y_true = np.stack(10 * [Y_TRUE_PAI], axis=0).reshape(10, -1)
        y_pred = np.stack(10 * [Y_PRED_PAI], axis=0).reshape(10, -1)

        pai_mu, pai_std = predictive_accuracy_index_averaged_over_time(y_true=y_true, y_pred=y_pred)
        target_mu, target_std = 2.1333333333333333, 0
        self.assertTrue((pai_mu, pai_std) == (target_mu, target_std))


def safe_f1_score(pr):
    p, r = pr
    if p + r == 0:
        return 0
    else:
        return 2 * (p * r) / (p + r)


def best_threshold(y_class, y_score, verbose=True):
    """

    :param y_class: ndarray the true values for hot or not spots {0,1}
    :param y_score: (probas_pred or y_score) probability of hot or not [0,1] for classification models and estimated count for regression models
    :param verbose: prints out logs if true
    :return: float threshold value where the best f1 score is for the given y_class and y_score
    """
    precision, recall, thresholds = precision_recall_curve(y_class.flatten(), y_score.flatten())
    f1_scores = np.array(list(map(safe_f1_score, zip(precision, recall))))
    index_array = np.argmax(f1_scores)  # almost always a singular int, and not an array
    if verbose:
        log.info(f"f1_score: {f1_scores[index_array]} at index {index_array}, new threshold {thresholds[index_array]}")
    return thresholds[index_array]


def get_y_pred(thresh, y_score):
    """
    Note: thresh should be determined using the training data only
    :param thresh: Used best_threshold to get the optimal threshold for the maximum F1 score
    :param y_score: (probas_pred or y_score) [0,1] or [0,inf) depending if a regression or classification model
    :return y_pred: thresholds float values of probas_pred to get hard classifications
    """
    if thresh < 0:
        raise Exception("threshold must be greater than 0")
    y_pred = np.copy(y_score)

    y_pred[y_pred < thresh] = 0
    y_pred[y_pred >= thresh] = 1

    return y_pred


class TestPredictiveThresholds(unittest.TestCase):

    def test_predictive_thresholds(self):
        """
        unit test to ensure the thresholding is applied correctly for arrays with [0,1] and [0, inf) values
        """
        y_class = np.random.binomial(1, 0.4, (20, 1, 4)).flatten()
        y_score = np.random.rand(20, 1, 4).flatten()
        y_score_gt_1 = 100 * y_score / np.max(y_score)

        thresh = best_threshold(y_class=y_class, y_score=y_score)
        thresh_gt_1 = best_threshold(y_class=y_class, y_score=y_score_gt_1)

        y_pred = get_y_pred(thresh, y_score)
        y_pred_gt_1 = get_y_pred(thresh_gt_1, y_score_gt_1)

        self.assertTrue((y_pred == y_pred_gt_1).all())


def best_thresholds(y_class, y_score):
    """
    :param y_class: ndarray (N,C,L) the true values for hot or not spots {0,1}
    :param y_score: ndarray (N,C,L) (probas_pred or y_score) probability of hot or not [0,1] for classification models and estimated count for regression models
    :return list of best thresholds per cell
    """
    N, C, L = y_class.shape
    thresholds = np.empty(L)
    for l in range(L):
        # IMPORTANT NOTE: THIS SHOULD BE CALCULATED ON THE TRAINING SET
        thresholds[l] = best_threshold(y_class=y_class[:, :, l], y_score=y_score[:, :, l])

    return thresholds


def get_y_pred_by_thresholds(thresholds, y_score):
    """
    :param thresholds: ndarray (N,C,L) threshold per cell (every l in L)
    :param y_score: ndarray (N,C,L) (probas_pred or y_score) probability of hot or not [0,1] for classification models and estimated count for regression models
    :return list of best thresholds per cell
    """
    N, C, L = y_score.shape
    y_pred = np.empty(y_score.shape)  # (N,C,L)
    for l in range(L):
        y_pred[:, :, l] = get_y_pred(thresh=thresholds[l],
                                     y_score=y_score[:, :, l])

    return y_pred


def mean_absolute_scaled_error(y_true, y_pred):
    """
    Calculated the ratio between MAE of predicted value and the y_true lag by one time step.
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true_lag = y_true[:-1].copy()
    y_true = y_true[1:]
    y_pred = y_pred[1:]

    mae_lag = mean_absolute_error(y_true, y_true_lag)
    mae_pred = mean_absolute_error(y_true, y_pred)

    mase = mae_pred / mae_lag  # anything less than one is good

    return mase


def get_metrics(y_true, y_pred):  # for a single time cell (N,1)
    """
    grids_true: true crime grid (N,d,d)
    grids_pred: predicted crime grid (N,d,d)
    returns accuracy, precision, recall f1score and mase
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred)

    r = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1score': f1score, 'mase': mase}

    return r
    # mase as well mse our prediction / mse lagged by one prediction
    # from sklearn.metrics import mean_squared_error


# ============= PLOTS =============
def plot_cell_results(y_true, y_pred):
    """
    function usually called in a loop to showcase a group of cells results
    """
    indices = get_times(y_true)

    plt.figure(figsize=(20, 3))
    plt.scatter(indices, y_pred[indices], s=20, marker='1')
    plt.scatter(indices, y_true[indices], s=20, c='r', marker='2')
    plt.plot(y_pred, alpha=0.7)

    plt.show()


def plot_conf(y_true, y_pred):
    conf = confusion_matrix(y_true, y_pred)  # can't do confusion because the model never predicts higher

    plt.figure(figsize=(5, 5))
    plt.title("Confusion Matrix")
    plt.ylabel("True Value")
    plt.xlabel("Predicted Value")
    plt.imshow(conf, cmap='viridis')
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))
    plt.grid(False)
    plt.colorbar()
    plt.show()


def plot_roc_and_pr_curve(y_true, probas_pred_dict):
    # ROC and PR Are binary so the class should be specified, one-vs-many
    # selecting and setting the class
    fig, (ax0, ax1) = plt.subplots(1, 2, )
    fig.set_figwidth(15)
    fig.set_figheight(10)

    ax0.set_title("Receiver Operating Characteristic (ROC) Curve")  # (AUC = %.4f)"%auc)
    ax0.set_xlabel("False Positive Rate")
    ax0.set_ylabel("True Positive Rate")
    ax0.set_aspect(1)
    ax0.set_xlim(0, 1.)
    ax0.set_ylim(0, 1.)
    ax1.set_title("Precision-Recall Curve")  # (AUC = %.4f)"%auc)
    ax1.set_ylabel("Precision")
    ax1.set_xlabel("Recall")
    ax1.set_aspect(1.)
    ax1.set_xlim(0, 1.)
    ax1.set_ylim(0, 1.)

    for name, probas_pred in probas_pred_dict.items():
        probas_pred = probas_pred.ravel()

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_true, probas_pred, drop_intermediate=False)
        thresholds = thresholds / np.max(thresholds)
        auc = roc_auc_score(y_true, probas_pred)
        ax0.plot(fpr, tpr, label=name + " (AUC: %.3f)" % auc, alpha=0.5)
        ax0.scatter(fpr, tpr, alpha=0.5, s=5 * thresholds, marker='D')

        # Precision Recall Curve
        precision, recall, thresholds = precision_recall_curve(y_true, probas_pred)
        thresholds = thresholds / np.max(thresholds)
        ap = average_precision_score(y_true, probas_pred)
        ax1.plot(recall, precision, label=name + " (AP: %.3f)" % ap, alpha=0.5)
        ax1.scatter(recall, precision, alpha=0.5, s=5 * thresholds, marker='D')

    f_scores = [.2, .3, .4, 0.5, .6, .7, .8, .9]
    for f_score in f_scores:
        x = np.linspace(0.0001, 1.1, 200)
        y = f_score * x / (2 * x - f_score)

        x = x[y >= 0]
        y = y[y >= 0]

        l, = ax1.plot(x, y, color='gray', alpha=0.2)
    #         plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    ax0.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
               fancybox=True, shadow=True, ncol=2)

    ax0.plot([0, 1], [0, 1], c='k', alpha=0.2)

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
               fancybox=True, shadow=True, ncol=2)

    plt.tight_layout()
    plt.show()


# Metric Plots
class BaseMetricPlotter:  # todo: replace with fig axis instead
    """
    Class is used to setup and add plots to a figure and then save or show this figure
    """

    def __init__(self, title):
        rcParams["mathtext.fontset"] = "stix"
        rcParams["font.family"] = "STIXGeneral"
        rcParams["font.size"] = "18"
        self.title = title
        self.grid_alpha = 0.5

        self.fig = None
        self.ax = None

        self.plot_kwargs = {
            # "marker": 's',
            # "markersize": .5,
            # "lw": 1,
            "alpha": .5,
        }

    def finalise(self):
        plt.title(self.title)
        plt.legend(bbox_to_anchor=(1.01, 0), loc="lower left", borderaxespad=0)
        plt.grid(alpha=self.grid_alpha)

    def show(self):
        self.finalise()
        plt.show()

    def savefig(self, file_location):
        self.finalise()
        plt.savefig(file_location, bbox_inches='tight')


class LossPlotter(BaseMetricPlotter):
    """
    Class is used to plot the validation and training loss of a model
    """

    def __init__(self, title):  # setup maybe add the size of the figure
        super(LossPlotter, self).__init__(title)

        plt.figure(figsize=(15, 5))
        plt.ylabel("Loss")
        plt.xlabel("Epoch")

    @staticmethod
    def plot_losses(trn_loss, all_trn_loss, val_loss, all_val_loss):
        """
        Note: the epoch plot takes the first batch loss as its first value, all subsequent values are the averages of
        that epoch.
        """
        kwargs = {
            "marker": "|",
            "markersize": 6,
            "alpha": 0.5,
        }

        # plt.xticks(list(range(len(trn_loss)))) # plot the x ticks of grid on the epochs

        # insert the first loss to illustrate the curve better
        trn_loss_x = np.linspace(start=0, stop=len(trn_loss), num=len(trn_loss) + 1)
        plt.plot(trn_loss_x, [all_trn_loss[0], *trn_loss], label="Training Loss (Epoch)", c='g', **kwargs)
        all_trn_loss_x = np.linspace(start=0, stop=len(trn_loss), num=len(all_trn_loss))
        plt.plot(all_trn_loss_x, all_trn_loss, alpha=.2, c='g', label="Training Loss (Batch)")

        val_loss_x = np.linspace(start=0, stop=len(val_loss), num=len(val_loss) + 1)
        plt.plot(val_loss_x, [all_val_loss[0], *val_loss], label="Validation Loss (Epoch)", c='r', **kwargs)
        all_val_loss_x = np.linspace(start=0, stop=len(val_loss), num=len(all_val_loss))
        plt.plot(all_val_loss_x, all_val_loss, alpha=.2, c='r', label="Validation Loss (Batch)")
        plt.grid()


class PRCurvePlotter(BaseMetricPlotter):
    """
    Class is used to setup and add plots to a figure and then save or show this figure
    """

    def __init__(self, title="Precision-Recall Curve"):
        super(PRCurvePlotter, self).__init__(title)

        plt.figure(figsize=(10, 10))

        plt.xlabel("Recall")
        plt.ylabel("Precision")

        plt.xlim(-0.01, 1.1)
        plt.ylim(0, 1.1)

        beta = 1
        # plot f score contours`
        f_scores = [.4, 0.5, .6, 0.7, .8, .9]
        for i, f_score in enumerate(f_scores):
            x = np.linspace(0.01, 1.1)
            y = f_score * x / ((1 + (beta ** 2)) * x - f_score * (beta ** 2))
            # y = f_score * x / (2 * x - f_score)
            if i == 0:  # used to add the f_score label
                l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2, label='F1 score')
            else:
                l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('F{0}={1:0.1f}'.format(beta, f_score), xy=(0.9, y[45] + 0.02))

    def add_curve(self, y_true, probas_pred, label_name):
        precision, recall, thresholds = precision_recall_curve(y_true, probas_pred)
        ap = average_precision_score(y_true, probas_pred)

        self.plot_kwargs["label"] = label_name + f" (AP={ap:.3f})"
        plt.plot(recall, precision, **self.plot_kwargs)

    def add_curve_(self, precision, recall, ap, label_name):

        self.plot_kwargs["label"] = label_name + f" (AP={ap:.3f})"
        plt.plot(recall, precision, **self.plot_kwargs)


class ROCCurvePlotter(BaseMetricPlotter):
    """
    Class is used to setup and add plots to a figure and then save or show this figure
    """

    # setup maybe add the size of the figure
    def __init__(self, title="Receiver Operating Characteristic (ROC) Curve"):
        super(ROCCurvePlotter, self).__init__(title)

        plt.figure(figsize=(10, 10))

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.xlim(-0.01, 1.1)
        plt.ylim(0, 1.1)

    def add_curve(self, y_true, probas_pred, label_name):
        """
        :param y_true: array, shape = [n_samples] or [n_samples, n_classes]
        True binary labels or binary label indicators.
        :param probas_pred: array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers). For binary
        y_true, y_score is supposed to be the score of the class with greater
        label.
        :param label_name: name of the line shown on the legend
        :return: none
        """

        fpr, tpr, thresholds = roc_curve(y_true, probas_pred, drop_intermediate=False)
        auc = roc_auc_score(y_true, probas_pred)

        self.plot_kwargs["label"] = label_name + f" (AUC={auc:.3f})"
        plt.plot(fpr, tpr, **self.plot_kwargs)

    def add_curve_(self, fpr, tpr, auc, label_name):
        self.plot_kwargs["label"] = label_name + f" (AUC={auc:.3f})"
        plt.plot(fpr, tpr, **self.plot_kwargs)


class CellPlotter(BaseMetricPlotter):
    """
    Class is used to plot predictions vs ground truth per cell
    """

    def __init__(self, title=""):  # setup maybe add the size of the figure
        super(CellPlotter, self).__init__(title)

    @staticmethod
    def plot_predictions(y_true=None, y_pred=None, probas_pred=None):  # todo maybe just send in model result object?
        """
        function usually called in a loop to showcase a group of cells results
        """
        fig = plt.figure(figsize=(20, 3))
        ax = fig.add_subplot(1, 1, 1)
        plt.ylabel("Likelihood")  # todo overwrite with function
        plt.xlabel("Time Step")  # todo over write with function

        # indices = get_times(y_true)

        # Major ticks every 20, minor ticks every 5
        x_minor_ticks = np.arange(0, len(y_true) + 1, 1)
        x_major_ticks = np.arange(0, len(y_true) + 1, 24)

        y_minor_ticks = np.arange(0, max(y_true) + .1, .1)
        y_major_ticks = np.arange(0, int(max(y_true)) + 1, 1)

        ax.set_xticks(x_major_ticks)
        ax.set_xticks(x_minor_ticks, minor=True)
        ax.set_yticks(y_major_ticks)
        ax.set_yticks(y_minor_ticks, minor=True)

        plt.xlim(-1, len(y_true))
        plt.ylim(np.min(y_true) - .1, np.max(y_true) + .1)

        # And a corresponding grid
        ax.grid(which='both')

        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        if y_true:
            ax.plot(y_true, label="y_true")
            mean = np.ones(y_true.shape) * np.mean(y_true)
            ax.plot(mean, label="mean")
        if probas_pred:
            ax.plot(probas_pred, label="probas_pred")
        if y_pred:
            ax.plot(y_pred, label="y_pred")


class PerTimeStepPlotter(BaseMetricPlotter):
    """
    Plot time related things
    """

    # setup maybe add the size of the figure
    def __init__(self, xlabel="Time", ylabel="Score", title="Total Crime of Test Set Over Time"):
        super(PerTimeStepPlotter, self).__init__(title)

        plt.figure(figsize=(15, 5))

        plt.ylabel(ylabel)
        plt.xlabel(xlabel)

    @staticmethod
    def plot(data, label):
        plt.plot(data, label=label)


def accuracy_score_per_cell(y_true, y_pred):
    """
    y_true: shape (N,1,L)
    probas_pred: (N,1,L)

    return: (1,1,L)
    """
    N, _, L = y_true.shape
    result = np.zeros(L)

    for i in range(L):
        result[i] = accuracy_score(y_true=y_true[:, :, i].flatten(),
                                   y_pred=y_pred[:, :, i].flatten())

    result = np.expand_dims(result, axis=0)
    result = np.expand_dims(result, axis=0)

    return result


def precision_score_per_cell(y_true, y_pred):
    """
    y_true: shape (N,1,L)
    probas_pred: (N,1,L)

    return: (1,1,L)
    """
    N, _, L = y_true.shape
    result = np.zeros(L)

    for i in range(L):
        result[i] = precision_score(y_true=y_true[:, :, i].flatten(),
                                    y_pred=y_pred[:, :, i].flatten())

    result = np.expand_dims(result, axis=0)
    result = np.expand_dims(result, axis=0)

    return result


def recall_score_per_cell(y_true, y_pred):
    """
    y_true: shape (N,1,L)
    probas_pred: (N,1,L)

    return: (1,1,L)
    """
    N, _, L = y_true.shape
    result = np.zeros(L)

    for i in range(L):
        result[i] = recall_score(y_true=y_true[:, :, i].flatten(),
                                 y_pred=y_pred[:, :, i].flatten())

    result = np.expand_dims(result, axis=0)
    result = np.expand_dims(result, axis=0)

    return result


def average_precision_score_per_cell(y_true, probas_pred):
    """
    y_true: shape (N,1,L)
    probas_pred: (N,1,L)

    return: (1,1,L)
    """
    N, _, L = y_true.shape
    result = np.zeros(L)

    for i in range(L):
        result[i] = average_precision_score(y_true=y_true[:, :, i].flatten(),
                                            y_score=probas_pred[:, :, i].flatten())

    result = np.expand_dims(result, axis=0)
    result = np.expand_dims(result, axis=0)

    return result


def roc_auc_score_per_cell(y_true, probas_pred):
    """
    y_true: shape (N,1,L)
    probas_pred: (N,1,L)

    return: (1,1,L)
    """
    N, _, L = y_true.shape
    result = np.zeros(L)

    for i in range(L):
        result[i] = average_precision_score(y_true=y_true[:, :, i].flatten(),
                                            y_score=probas_pred[:, :, i].flatten())

    result = np.expand_dims(result, axis=0)
    result = np.expand_dims(result, axis=0)

    return result


def matthews_corrcoef_per_time_slot(y_class, y_pred):
    """
    :param y_class: ndarray (N,1,1)
    :param y_pred: ndarray (N,1,1)
    :return: ndarray (N,1,1)
    """
    N, _, L = y_class.shape
    result = np.zeros(N)

    for i in range(N):
        result[i] = matthews_corrcoef(
            y_true=y_class[i, :, :].flatten(),
            y_pred=y_pred[i, :, :].flatten(),
        )

    result = np.expand_dims(result, axis=1)
    result = np.expand_dims(result, axis=1)
    return result


def average_precision_score_per_time_slot(y_class, y_score):
    """
    y_true: shape (N,1,L)
    probas_pred: (N,1,L)

    return: (N,1,1)
    """
    N, _, L = y_class.shape
    result = np.zeros(N)

    for i in range(N):
        result[i] = average_precision_score(y_true=y_class[i, :, :].flatten(),
                                            y_score=y_score[i, :, :].flatten())

    result = np.expand_dims(result, axis=1)
    result = np.expand_dims(result, axis=1)
    return result


def roc_auc_score_per_time_slot(y_class, y_score):
    """
    y_true: shape (N,1,L)
    probas_pred: (N,1,L)

    return: (N,1,1)
    """
    N, _, L = y_class.shape
    result = np.zeros(N)

    for i in range(N):
        result[i] = roc_auc_score(y_true=y_class[i, :, :].flatten(),
                                  y_score=y_score[i, :, :].flatten())

    result = np.expand_dims(result, axis=1)
    result = np.expand_dims(result, axis=1)
    return result


def accuracy_score_per_time_slot(y_class, y_pred):
    """
    y_true: shape (N,1,L)
    probas_pred: (N,1,L)

    return: (N,1,1)
    """
    N, _, L = y_class.shape
    result = np.zeros(N)

    for i in range(N):
        result[i] = accuracy_score(y_true=y_class[i, :, :].flatten(),
                                   y_pred=y_pred[i, :, :].flatten())

    result = np.expand_dims(result, axis=1)
    result = np.expand_dims(result, axis=1)

    return result


def precision_score_per_time_slot(y_class, y_pred):
    """
    y_true: shape (N,1,L)
    probas_pred: (N,1,L)

    return: (N,1,1)
    """
    N, _, L = y_class.shape
    result = np.zeros(N)

    for i in range(N):
        result[i] = precision_score(y_true=y_class[i, :, :].flatten(),
                                    y_pred=y_pred[i, :, :].flatten())

    result = np.expand_dims(result, axis=1)
    result = np.expand_dims(result, axis=1)

    return result


def recall_score_per_time_slot(y_class, y_pred):
    """
    y_true: shape (N,1,L)
    probas_pred: (N,1,L)

    return: (N,1,1)
    """
    N, _, L = y_class.shape
    result = np.zeros(N)

    for i in range(N):
        result[i] = recall_score(y_true=y_class[i, :, :].flatten(),
                                 y_pred=y_pred[i, :, :].flatten())

    result = np.expand_dims(result, axis=1)
    result = np.expand_dims(result, axis=1)

    return result


def rmse_per_time_slot(y_true, y_pred):
    """
    y_true: shape (N,1,L)  float values varying between (0,1)
    probas_pred: (N,1,L) float values varying between (0,1)

    return: (N,1,1)
    """
    N, _, L = y_true.shape
    result = np.zeros(N)

    for i in range(N):
        result[i] = np.sqrt(mean_squared_error(y_true=y_true[i, :, :].flatten(),
                                               y_pred=y_pred[i, :, :].flatten()))

    result = np.expand_dims(result, axis=1)
    result = np.expand_dims(result, axis=1)

    return result


def mae_per_time_slot(y_true, y_pred):
    """
    y_true: shape (N,1,L)  float values varying between (0,1)
    probas_pred: (N,1,L) float values varying between (0,1)

    return: (N,1,1)
    """
    N, _, L = y_true.shape
    result = np.zeros(N)

    for i in range(N):
        result[i] = mean_absolute_error(y_true=y_true[i, :, :].flatten(),
                                        y_pred=y_pred[i, :, :].flatten())

    result = np.expand_dims(result, axis=1)
    result = np.expand_dims(result, axis=1)

    return result
