import numpy as np
import torch


def accuracy_top_k(p, q, k):
    """
    Inputs
    ======
    p:    Predicted grid
    q:    Exact grid
    k:    Top k values
    """
    p_vals, p_args = torch.topk(p, k)
    q_vals, q_args = torch.topk(q, k)

    r = len(set(p_args) & set(q_args)) / len(set(q_args))

    return r


# top k accuracies for series of k and whole batch/set of data
def acc(pred, targ, k=1):
    """
    targ: target maps (N,d,d)
    pred: predicted maps (N,d,d)
    k: top k spots to be similar
    """
    accuracies = []
    for i in range(len(targ)):
        p = pred[i].view(-1)
        q = targ[i].view(-1)
        accuracies.append(accuracy_top_k(p, q, k))

    return accuracies


def get_recall(y_true, y_pred):
    """
    args: y_true, y_pred
    """
    all_real_pos = np.sum(y_true)
    correct_pos = np.sum(y_true & y_pred)
    return correct_pos / all_real_pos


def get_precision(y_true, y_pred):
    """
    args: y_true, y_pred
    """
    all_class_pos = np.sum(y_pred)
    correct_pos = np.sum(y_true & y_pred)
    return correct_pos / all_class_pos


def f1_score(y_true, y_pred):
    """
    args: y_true, y_pred
    returns number of true positive predictions
    """
    p = get_precision(y_true, y_pred)
    r = get_recall(y_true, y_pred)
    return 2 * p * r / (p + r)


def get_true_positive(y_true, y_pred):
    """
    returns number of true positive predictions
    """
    return np.sum(y_true & y_pred)


def get_false_positive(y_true, y_pred):
    """
    returns number of false positive predictions
    """
    xor = y_true != y_pred
    return np.sum(xor & y_pred)


def get_true_negative(y_true, y_pred):
    """
    args: y_true, y_pred
    returns number of true negative predictions
    """
    return np.sum((1 != y_true) & (1 != y_pred))


def get_false_negative(y_true, y_pred):
    """args: y_true, y_pred
    returns number of false negative predictions
    """
    xor = y_true != y_pred
    return np.sum(xor & (1 != y_pred))


def get_accuracy(y_true, y_pred):
    """
    args: y_true, y_pred
    returns accuracy between 0. and 1.

    """
    return np.sum(y_true == y_pred) / len(y_true)


def confusion(y_true, y_pred):
    """
    return confusion with true values on y axis and predicted values on x axis

           pred
          _ _ _ _
     t   |_|_|_|_|
     r   |_|_|_|_|
     u   |_|_|_|_|
     e   |_|_|_|_|


    """
    size = len(set(list(y_true) + list(y_true)))
    r = np.zeros((size, size))

    for i in range(len(y_true)):
        j = int(y_true[i])
        k = int(y_pred[i])
        r[j, k] += 1

    return r
