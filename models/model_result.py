from pprint import pformat
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


class ModelResult:
    def __init__(self, model_name, y_true, y_pred, probas_pred, t_range, indices, shape):
        """

        :param model_name: text used to refer to the model on plots
        :param y_true: ground truth value (i.e. did crime happen 1 or not 0) {0,1}
        :param y_pred: the model's hard prediction of the model {0,1}
        :param probas_pred: model floating point values, can be likelihoods [0,1) or count estimates [0,n)
        :param t_range: range of the times of the test set - also used in plots
        :param indices: list of indices (of each prediction -> (N,C,H,W)) to reconstruct the grid maps for display
        :param shape: shape of the y_true data - (N,L) flattened so that we can link the predictions with certain times
        """

        self.model_name = model_name
        self.y_true = y_true
        self.y_pred = y_pred
        self.probas_pred = probas_pred
        self.t_range = t_range
        self.indices = indices
        self.shape = shape

        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        self.roc_auc = roc_auc_score(self.y_true, self.probas_pred)
        self.average_precision = average_precision_score(self.y_true, self.probas_pred)

    def __repr__(self):
        r = rf"""
            Model Name: {self.model_name}
                ROC AUC:            {self.roc_auc}
                Average Precision:  {self.average_precision}
                Accuracy:           {self.accuracy}
        """

        return r

    def __str__(self):  # todo change to only have metrics
        return self.__repr__()
