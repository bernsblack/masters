import logging
import numpy as np


def get_max_steps(data, step):
    """
    data shape: (N,C,L)
    """
    if len(data.shape) != 3:
        raise ValueError("Data should be in format (N,C,L)")

    logging.info("finding best max_steps parameter...")
    inputs = np.copy(data[:, 0])
    targets = np.copy(inputs)
    targets = targets[1:]
    inputs = inputs[:-1]

    tst_len = int(0.3 * len(inputs))

    results = {}

    for max_steps in range(1, 10):
        out = historic_average(data=inputs,
                               step=step,
                               max_steps=max_steps)

        mae = np.mean(np.abs(targets[tst_len:] - out[tst_len:]))
        # print(f"mae:{mae} max_steps:{max_steps}")
        results[mae] = max_steps
    best = results[min(list(results.keys()))]

    logging.info(f"best max_steps -> {best}")
    return best


def historic_average(data, step, max_steps):
    """
    historic_average gets the historic average of the following time step without including that time step.
    it gets the the historic average of each cell and then moves it one time step forward.
    We do not want to use the input cell's historic average because it's less related to the target cell

    :param data: (N, ...) shape
    :param step: interval used to calculate the average
    :param max_steps: number of intervals to sample
    :return: historic average of the next time steps
    """
    r = np.zeros(data.shape)  # so that calculation can still be done on the first few cells
    # r = np.empty(data.shape)
    # r.fill(np.nan)
    for i in range(step + 1, len(r)):
        a_subset = data[i - step + 1:0:-step]  # +1 to take the historic average of the next time step
        if max_steps > 0:
            a_subset = a_subset[:max_steps]

        x = np.mean(a_subset, axis=0)
        r[i] = x

    return r
