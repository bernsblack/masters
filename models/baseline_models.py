import numpy as np

def get_historic_average(a, step=24):
    """
    a: array (N,d) or (N,d,d)
    step: the historic step the average is taken over, e.g. 24 for every 24 hours
    """
    ha = []  # historic averages

    for i in range(step):
        ha.append(a[i::step].mean(0))

    b = -1 * np.ones_like(a)  # minus one there to check if anything went wrong
    for i in range(len(b)):
        b[i] = ha[i % step]

    return b


def rolling_apply(fun, a, w):  # this rolling average includes the current time step
    r = np.empty(a.shape)
    r.fill(np.nan)
    for i in range(w - 1, a.shape[0]):
        r[i] = fun(a[(i - w + 1):i + 1])
    return r


a = np.random.randn(100)

