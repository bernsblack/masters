import unittest

from sparse_discrete_table import SparseDiscreteTable


def get_mock_p_xy():
    mock_rv_names = ['x', 'y']
    mock_table = {
        (0, 0): .1,
        (0, 1): .2,
        (1, 0): .3,
        (1, 1): .4,
    }
    mock_rv = SparseDiscreteTable(rv_names=mock_rv_names, table=mock_table)
    return mock_rv


def get_mock_p_x():
    mock_rv_names = ['x']
    mock_table = {
        (0,): .3,
        (1,): .4,
    }
    mock_rv = SparseDiscreteTable(rv_names=mock_rv_names, table=mock_table)
    return mock_rv


def get_mock_p_y():
    mock_rv_names = ['y']
    mock_table = {
        (0,): .1,
        (1,): .2,
    }
    mock_rv = SparseDiscreteTable(rv_names=mock_rv_names, table=mock_table)
    return mock_rv


class TestSparseDiscreteTable(unittest.TestCase):

    def test_case_0(self):
        p_xy = get_mock_p_xy()
        p_x = p_xy.marginal(rv_names=['x'])
        p_y = p_xy.marginal(rv_names=['y'])
        print(p_xy)
        print(p_x)
        print(p_y)

        self.assertEqual(0, 0)
