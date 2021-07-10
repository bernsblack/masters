import numpy as np

from utils.data_processing import pad4d


def mock_cnn_data_classification(N=100, C=1, H=16, W=16, filter_size=3, p=.15, thresh=3):
    """
    C should always be one

    returns gen_data as input data to the model
    returns result as the binary classification if the sum of neighbouring cells are above the threshold
    """
    C = 1
    shape = N, C, H, W
    pad_size = (filter_size - 1) // 2
    gen_data = np.random.binomial(1, p, shape)
    pad_data = pad4d(gen_data, value=0, size=pad_size)

    result = np.zeros(shape)
    for i in range(H):
        for j in range(W):
            for k in range(N):
                result[k, :, i, j] = np.sum(pad_data[k, :, i:i + filter_size, j:j + filter_size])

    result[result < thresh] = 0
    result[result >= thresh] = 1

    return gen_data, result


def mock_rnn_data_classification(seq_len=10, batch_size=2):
    """
    Generate mock data that can be used to develop/test RNNs. The output generates 1 if the cumsum of all previous
    input is greater than the threshold, else it generates 0.
    :param seq_len: number of time steps
    :param batch_size: number of independent columns of the data
    :return: x (seq_len, batch_size, 1) input data, y (seq_len, batch_size, 1) target data
    """
    x = np.random.rand(seq_len, batch_size, 1)

    y = np.cumsum(x, axis=0)
    thresh = np.mean(y)
    y[y < thresh] = 0
    y[y >= thresh] = 1

    return x, y


# adding problem data generation - regression probem though
def mock_rnn_data_regression(seq_len=200, batch_size=50):
    """
    Adding problem example
    Generates regression mock data for a RNN model
    Data has two sequences
        - sequence one is random numbers between 0 and 1
        - sequence two is all zeros except for two random indices which are one
    Output data is the cumulative some of the product of these to sequences.

    returns:
    X: (seq_len, batch_size, n_features: 2)
    y: (seq_len, batch_size, class_target: 1)

    Can be trained on:
        - only the last sample y[-1]
        - every sample in the sequence y
    """
    t = np.zeros((seq_len, batch_size, 1))
    c = np.zeros((seq_len, batch_size, 2))
    for i in range(batch_size):
        a, b = list(np.random.choice(seq_len, 2, replace=False))
        z = np.zeros(seq_len)
        r = np.random.rand(seq_len)

        z[a] = 1
        z[b] = 1

        cum_sum = z * r
        cum_sum = np.cumsum(cum_sum).reshape(-1, 1)

        t[:, i, :] = cum_sum
        c[:, i, 0] = z
        c[:, i, 1] = r
    return c, t


def mock_fnn_data_classification(n_samples=100, n_feats=5, class_split=0.5):
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
