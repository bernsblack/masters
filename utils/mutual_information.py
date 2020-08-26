from typing import List

import numpy as np
import unittest
from utils import describe_array
from utils.preprocessing import Shaper

SMALLEST_TOLERANCE = 1e-14


def construct_mi_grid(mi_arr: np.ndarray, shaper: Shaper, normalize=True):
    """
    mi_arr: shape (L,K+1) where L is the number of cells and K is the maximum time offset
             each value represents the mutual information between a cells crime at time t and t - k.
             the first value in axis 1 is mi(crime at t, crime at t)
    normalize: if true normalize the cells relative to their self-information

    mi_grid: return ndarray in the shape (1,K,H,W)
             where K is the max time offset
             each value is normalised according to its own
    """
    self_mi_arr, mi_arr = mi_arr[:, :1], mi_arr[:, 1:]
    if normalize:
        # mi_arr = mi_arr / self_mi_arr # old faulty code
        # np.divide will leave numerator as zero if denominator is zero - preventing div by zero error
        mi_arr = np.divide(mi_arr, self_mi_arr, out=np.zeros_like(mi_arr), where=self_mi_arr!=0)

    mi_arr = np.swapaxes(mi_arr, 0, 1)
    mi_grid = shaper.unsqueeze(np.expand_dims(mi_arr, (0,)))
    return mi_grid


def not_(a):
    return np.invert(a.astype(np.bool)).astype(a.dtype)


def and_(a, b):
    return (a.astype(np.bool) & b.astype(np.bool)).astype(a.dtype)


def or_(a, b):
    return (a.astype(np.bool) | b.astype(np.bool)).astype(a.dtype)


def f_00(x, y):
    """
    return 1 if x,y = 0,0 else return 0
    """
    return not_(or_(x, y))


def f_01(x, y):
    """
    return 1 if x,y = 0,1 else return 0
    """
    return and_(not_(x), y)


def f_10(x, y):
    """
    return 1 if x,y = 1,0 else return 0
    """
    return and_(x, not_(y))


def f_11(x, y):
    """
    return 1 if x,y = 1,1 else return 0
    """
    return and_(x, y)


def xlog2x_single(x):
    if x == 0:
        return 0
    elif x < 0:
        raise Exception(f"x should be greater or equal than zero, x = {x}")
    else:
        return x * np.log2(x)


xlog2x = np.vectorize(xlog2x_single)


def entropy(p_dist, axis=None):
    """
    p_dist: array of all probabilities

    entropy should always be nonnegative, .i.e. > 0

    entropy = -sum(p_i*log(p_i))
    """
    h = -1 * np.sum(xlog2x(p_dist), axis=axis)
    if h < 0:  # todo add tolerence for computation rounding errors
        raise Exception(f"Entropy should always be non negative => result was {h}")

    return h


def get_probabilities(a, b, axis=0):
    """returns probability table of a, b and joint entropy of a"""
    assert (np.unique(a) == np.array([0, 1])).all(), f"a values must be [0,1], not {np.unique(a)}"
    assert (np.unique(b) == np.array([0, 1])).all(), f"a values must be [0,1], not {np.unique(b)}"

    size = np.shape(a)[axis]

    p_a_1 = a.sum(axis=axis) / size
    p_a_0 = 1 - p_a_1

    p_b_1 = b.sum(axis=axis) / size
    p_b_0 = 1 - p_b_1

    p_ab_00 = f_00(a, b).sum(axis=axis) / size
    p_ab_01 = f_01(a, b).sum(axis=axis) / size
    p_ab_10 = f_10(a, b).sum(axis=axis) / size
    p_ab_11 = f_11(a, b).sum(axis=axis) / size

    p_a = np.stack([
        p_a_0,
        p_a_1,
    ], axis=axis)

    p_b = np.stack([
        p_b_0,
        p_b_1,
    ], axis=axis)

    p_ab = np.stack([
        p_ab_00,
        p_ab_01,
        p_ab_10,
        p_ab_11,
    ], axis=axis)

    return p_a, p_b, p_ab


def get_entropies(a, b, axis=0):
    """returns entropy of a, b and joint entropy of a"""

    # todo re-evaluate does not necessarily need to be 2-d -> what about summation
    # assert len(np.shape(a)) == 2, "a must be 2-d matrix"
    # assert len(np.shape(b)) == 2, "b must be 2-d matrix"

    p_a, p_b, p_ab = get_probabilities(a, b, axis)

    h_a = entropy(p_a, axis=axis)
    h_b = entropy(p_b, axis=axis)
    h_ab = entropy(p_ab, axis=axis)

    return h_a, h_b, h_ab


def mutual_info(a, b, axis=0):
    """
    Mutual information only for binary distributions - for our purposes input shape should be (N,L)
    """
    h_a, h_b, h_ab = get_entropies(a, b, axis)

    mi_ab = h_a + h_b - h_ab

    # should always be greater than SMALLEST_TOLERANCE

    min_ab = mi_ab.min()
    if -SMALLEST_TOLERANCE < mi_ab.min() < 0:
        print(f"WARNING: mutual information is less than zero, but within SMALLEST_TOLERANCE ({SMALLEST_TOLERANCE})")
        mi_ab[mi_ab < 0] = 0
    elif mi_ab.min() <= -SMALLEST_TOLERANCE:
        raise Exception(f"Mutual information cannot be less than " + \
                        "SMALLEST_TOLERANCE ({-SMALLEST_TOLERANCE}) => mi_ab = h_a + h_b - h_ab")

    # print(f"\n\n<= mi_ab => {describe_array(mi_ab)}")
    # print(f"\n\n<= h_a => {describe_array(h_a)}")
    # print(f"\n\n<= h_b => {describe_array(h_b)}")
    # print(f"\n\n<= h_ab => {describe_array(h_ab)}")

    return mi_ab


# TODO calculate for more than two variables
def conditional_entropy(a, b, axis=0):
    """
    returns H(A|B) = H(A,B) - H(B)
    for H(B|A) = H(A,B) - H(A) just swap a and b around
    """
    _, h_b, h_ab = get_entropies(a, b, axis)

    # TODO LOOK INTO TOLERENCE HANDLING
    return h_ab - h_b


# ===========================================    UNIT TESTS    =========================================================
# TODO ADD UNIT TEST FOR MUTUAL INFORMATION AND ENTROPY
# class TestEntropy(unittest.TestCase):
#     a = np.array([0, 0, 1, 1], dtype=np.uint)


class TestBoolFunctions(unittest.TestCase):
    # todo: make test more akin to the grids (N,C,H,W) or (N,1,H,W) -> get squeezed to (N,1,L)
    # OUTPUT SHOULD BE SHAPE (1,1,L) in squeezed format
    # axis is the axis over which we sum, the axis that will disappear

    a = np.array([0, 0, 1, 1], dtype=np.uint)
    b = np.array([0, 1, 0, 1], dtype=np.uint)

    def test_f_00(self):
        self.assertTrue((f_00(self.a, self.b) == np.array([1, 0, 0, 0])).all())

    def test_f_01(self):
        self.assertTrue((f_01(self.a, self.b) == np.array([0, 1, 0, 0])).all())

    def test_f_10(self):
        self.assertTrue((f_10(self.a, self.b) == np.array([0, 0, 1, 0])).all())

    def test_f_11(self):
        self.assertTrue((f_11(self.a, self.b) == np.array([0, 0, 0, 1])).all())


if __name__ == '__main__':
    unittest.main()
