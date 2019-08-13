#!python3
import numpy as np
import torch

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
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve, \
    mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def mean_absolute_scaled_error(y_true, y_pred):
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


def plot_roc_and_pr_curve(ground_truth, predictions_list):
    # ROC and PR Are binary so the class should be specified, one-vs-many
    # selecting and setting the class
    fig, axs = plt.subplots(1, 2)
    fig.set_figwidth(15)

    axs[0].set_title("Receiver Operating Characteristic (ROC) Curve")  # (AUC = %.4f)"%auc)
    axs[0].set_xlabel("False Positive Rate")
    axs[0].set_ylabel("True Positive Rate")
    axs[0].set_aspect(1)
    axs[1].set_title("Precision-Recall Curve")  # (AUC = %.4f)"%auc)
    axs[1].set_ylabel("Precision")
    axs[1].set_xlabel("Recall")
    axs[1].set_aspect(1.)

    y_true = ground_truth.ravel()  # ground truth
    y_true[y_true > 1] = 1

    predictions_list_names = ['HA', 'MA', 'HAMA', 'KNN', 'GT']  # change the input to a dictionary instead

    for i, y_score in enumerate(predictions_list):
        y_score = y_score.ravel()

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)
        thresholds = thresholds / np.max(thresholds)
        auc = roc_auc_score(y_true, y_score)
        axs[0].plot(fpr, tpr, label=predictions_list_names[i] + " (AUC: %.3f)" % auc, alpha=0.5)
        axs[0].scatter(fpr, tpr, alpha=0.5, s=50 * thresholds, marker='D')

        # Precision Recall Curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        thresholds = thresholds / np.max(thresholds)
        ap = average_precision_score(y_true, y_score)
        axs[1].plot(recall, precision, label=predictions_list_names[i] + " (AP: %.3f)" % ap, alpha=0.5)
        axs[1].scatter(recall, precision, alpha=0.5, s=50 * thresholds, marker='D')

    # f_scores = [.4,0.5,.6,.8,.9]
    # for f_score in f_scores:
    #     x = np.linspace(0.01, 1.1)
    #     y = f_score * x / (2 * x - f_score)
    #     l, = axs[1].plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    #     plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    axs[0].legend()
    axs[0].plot([0, 1], [0, 1], c='k', alpha=0.2)

    axs[1].legend()

    plt.tight_layout()
    plt.show()


# Metric Plots
class BaseMetricPlotter:
    """
    Class is used to setup and add plots to a figure and then save or show this figure
    """

    def __init__(self, title, legend_loc="best"):  # setup maybe add the size of the figure
        rcParams["mathtext.fontset"] = "stix"
        rcParams["font.family"] = "STIXGeneral"
        rcParams["font.size"] = "18"
        self.title = title
        self.legend_loc = legend_loc
        self.grid_alpha = 0.2
        self.setup()

    @staticmethod
    def setup():
        raise NotImplemented

    def finalise(self):
        plt.title(self.title)
        plt.legend(loc=self.legend_loc)
        plt.grid(alpha=self.grid_alpha)

    def show(self):
        self.finalise()
        plt.show()

    def savefig(self, file_location):
        self.finalise()
        plt.savefig(file_location)


class LossPlotter(BaseMetricPlotter):
    """
    Class is used to plot the validation and training loss of a model
    """

    def __init__(self, title, legend_loc="best"):  # setup maybe add the size of the figure
        super(LossPlotter, self).__init__(title, legend_loc)

    @staticmethod
    def setup():
        plt.figure(figsize=(20, 10))
        plt.ylabel("Loss")
        plt.xlabel("Epoch")

    @staticmethod
    def plot_losses(trn_loss, all_trn_loss, val_loss, all_val_loss):
        kwargs = {
            "marker": "s",
            "markersize": 6,
            "alpha": 0.7,
        }

        # plt.xticks(list(range(len(trn_loss)))) # plot the x ticks of grid on the epochs

        plt.plot(trn_loss, label="Training Loss (Epoch)", c='g', **kwargs)
        all_trn_loss_x = np.linspace(start=0, stop=len(trn_loss) - 1, num=len(all_trn_loss))
        plt.plot(all_trn_loss_x, all_trn_loss, alpha=.2, c='g', label="Training Loss (Batch)")

        plt.plot(val_loss, label="Validation Loss (Epoch)", c='r', **kwargs)
        all_val_loss_x = np.linspace(start=0, stop=len(trn_loss) - 1, num=len(all_val_loss))
        plt.plot(all_val_loss_x, all_val_loss, alpha=.2, c='r', label="Validation Loss (Batch)")
        plt.grid()


class PRCurvePlotter(BaseMetricPlotter):
    """
    Class is used to setup and add plots to a figure and then save or show this figure
    """

    def __init__(self, title="Precision-Recall Curve", legend_loc="best"):
        super(PRCurvePlotter, self).__init__(title, legend_loc)

    @staticmethod
    def setup():
        plt.figure(figsize=(10, 10))

        plt.xlabel("Recall")
        plt.ylabel("Precision")

        plt.xlim(-0.01, 1.1)
        plt.ylim(0, 1.1)

        # plot f score contours`
        f_scores = [.4, 0.5, .6, 0.7, .8, .9]
        for i, f_score in enumerate(f_scores):
            x = np.linspace(0.01, 1.1)
            y = f_score * x / (2 * x - f_score)
            if i == 0:  # used to add the f_score label
                l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2, label='F1 score')
            else:
                l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('F1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    @staticmethod
    def add_curve(y_true, probas_pred, label_name):
        precision, recall, thresholds = precision_recall_curve(y_true, probas_pred)
        ap = average_precision_score(y_true, probas_pred)

        kwargs = {
            "label": label_name + f" (AP={ap:.3f})",
            "marker": 's',
            "markersize": 2,
            "alpha": 0.7,
        }

        plt.plot(recall, precision, **kwargs)


class ROCCurvePlotter(BaseMetricPlotter):
    """
    Class is used to setup and add plots to a figure and then save or show this figure
    """

    # setup maybe add the size of the figure
    def __init__(self, title="Receiver Operating Characteristic (ROC) Curve", legend_loc="best"):
        super(ROCCurvePlotter, self).__init__(title, legend_loc)

    @staticmethod
    def setup():
        plt.figure(figsize=(10, 10))

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.xlim(-0.01, 1.1)
        plt.ylim(0, 1.1)

    @staticmethod
    def add_curve(y_true, probas_pred, label_name):
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

        kwargs = {
            "label": label_name + f" (AUC={auc:.3f})",
            "marker": 's',
            "markersize": 2,
            "alpha": 0.7,
        }

        plt.plot(fpr, tpr, **kwargs)