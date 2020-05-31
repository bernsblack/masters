import unittest

from sparse_discrete_table import SparseDiscreteTable


def get_mock_p_xy():
    mock_rv_names = ['x', 'y']
    mock_table = {
        (0, 0): .1,
        (0, 1): .1,
        (1, 0): .4,
        (1, 1): .4,
    }
    mock_rv = SparseDiscreteTable(rv_names=mock_rv_names, table=mock_table)
    return mock_rv


def get_mock_p_x():
    mock_rv_names = ['x']
    mock_table = {
        (0,): 0.2,
        (1,): 0.8,
    }
    mock_rv = SparseDiscreteTable(rv_names=mock_rv_names, table=mock_table)
    return mock_rv


def get_mock_p_y():
    mock_rv_names = ['y']
    mock_table = {
        (0,): 0.5,
        (1,): 0.5,
    }
    mock_rv = SparseDiscreteTable(rv_names=mock_rv_names, table=mock_table)
    return mock_rv


class TestSparseDiscreteTable(unittest.TestCase):

    def test_marginal(self):
        p_xy = get_mock_p_xy()
        p_x = get_mock_p_x()
        p_y = get_mock_p_y()
        p_x_marg = p_xy.marginal(rv_names=['x'])
        p_y_marg = p_xy.marginal(rv_names=['y'])

        print(p_x, p_x_marg)
        print(p_y, p_y_marg)

        self.assertEqual(p_x, p_x_marg)
        self.assertEqual(p_y, p_y_marg)

    def test_conditional(self):
        print("\n=== test_conditional ===")
        p_xy = get_mock_p_xy()
        p_x = p_xy.marginal(rv_names=['x'])
        p_y = p_xy.marginal(rv_names=['y'])

        p_y_given_x = p_xy / p_x
        p_x_given_y = p_xy / p_y

        print("p_xy", p_xy)
        print("p_x", p_x)
        print("p_y", p_y)

        print("p_y_given_x", p_y_given_x)
        print("p_x_given_y", p_x_given_y)

        print("p_y_given_x*p_x", p_y_given_x * p_x)
        print("p_x_given_y*p_y", p_x_given_y * p_y)

        self.assertEqual(p_xy, p_y_given_x * p_x)
        self.assertEqual(p_xy, p_x_given_y * p_y)

    def test_entropy(self):
        p_xy = get_mock_p_xy()
        h = p_xy.entropy()
        print("p_xy", p_xy)
        print("p_xy.entropy()", h)
        self.assertAlmostEqual(h, 1.721928094887362)

    def test_mutual_information(self):
        p_xy = get_mock_p_xy()
        print("p_xy", p_xy)
        p_x = get_mock_p_x()
        print("p_x", p_x)
        p_y = get_mock_p_y()
        print("p_y", p_y)
        h_xy = p_xy.entropy()
        print("h_xy", h_xy)
        h_x = p_x.entropy()
        print("h_x", h_x)
        h_y = p_y.entropy()
        print("h_y", h_y)
        mi_xy_explicit = h_x + h_y - h_xy
        print("mi_xy_explicit", mi_xy_explicit)

        mi_xy_implicit = p_xy.mutual_information(rv_names_0=['x'], rv_names_1=['y'])
        print("mi_xy_implicit", mi_xy_implicit)

        self.assertAlmostEquals(mi_xy_explicit, mi_xy_implicit)

    def test_get_item(self):
        p_xy = get_mock_p_xy()
        print("p_xy['x']", p_xy['x'])
        self.assertEqual(p_xy['x'], p_xy.marginal(['x']))

        print("p_xy['x','y']", p_xy['x', 'y'])
        self.assertEqual(p_xy['x', 'y'], p_xy.marginal(['x', 'y']))
