from pprint import pformat
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


class ModelResult:
    def __init__(self, model_name, y_true, y_pred, probas_pred):
        self.model_name = model_name
        self.y_true = y_true
        self.y_pred = y_pred
        self.probas_pred = probas_pred

        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        self.roc_auc = roc_auc_score(self.y_true, self.probas_pred)
        self.average_precision = average_precision_score(self.y_true, self.probas_pred)

    def __repr__(self):
        out = {
            "Model Name": self.model_name,
            "ROC AUC": self.roc_auc,
            "Average Precision": self.average_precision,
            "Accuracy": self.accuracy,
        }

        return pformat(out)

    def __str__(self):  # todo change to only have metrics
        return self.__repr__()


