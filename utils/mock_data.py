import numpy as np


def mock_data(n_samples=100, n_feats=5, class_split=0.5):
    """
    Produces mock binary classification data.
    Generates input data X (n_samples, n_feats) where each feats value is an i.i.d. sample from bernoulli(p=class_split)
    y (n_samples) is the target data where the sum of the n_feats is larger than the mean, i.e. class_split*n_feats

    :param n_samples: number of samples
    :param n_feats: number of features
    :param class_split:
    :return: X (n_samples, n_feats), y (n_samples)
    """
    thresh = n_feats * class_split
    X = np.random.binomial(n=1, p=class_split, size=(n_samples, n_feats))
    y = np.sum(X, axis=1)
    y[y <= thresh] = 0
    y[y > thresh] = 1

    return X, y
