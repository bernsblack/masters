import numpy as np

from utils.data_processing import pad4d


def mock_cnn_data(N=100, C=1, H=16, W=16, filter_size=3, p=.15, thresh=3):
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


# todo look into incorporating a
def mock_rnn_data(seq_len=10, batch_size=2):
    """
    Generate mock data that can be used to develop/test RNNs. The output generates 1 if the cumsum of all previous
    input is greater than the threshold, else it generates 0.
    :param seq_len: number of time steps
    :param batch_size: number of independent columns of the data
    :return: x (seq_len, 1, batch_size) input data, y (seq_len, 1, batch_size) target data
    return data is in shape (seq_len, 1, batch_size) to mimic the (N,C,L) shape in the crime data, where C = 1
    """
    x = np.random.rand(seq_len, 1, batch_size)

    y = np.cumsum(x, axis=0)
    thresh = np.mean(y)
    y[y < thresh] = 0
    y[y >= thresh] = 1

    return x, y


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
