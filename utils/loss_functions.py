import torch
from torch import nn


class F1_Loss(nn.Module):
    """
    Calculate F1 score. Can work with gpu tensors

    The original implementation is written by Michal Haltuf on Kaggle.
    This updated implementation is taken from SuperShinyEyes on GitHub.
    (https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354)

    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1

    Example
    -------
    f1_loss = F1_Loss().cuda()


    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    - https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
    """
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true ,):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()
