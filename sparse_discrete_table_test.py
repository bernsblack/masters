import unittest

from sparse_discrete_table import SparseDiscreteTable
import numpy as np


def get_mock_p_xyz():
    mock_rv_names = ['x', 'y', 'z']
    mock_table = {
        (0, 0, 0): .2,
        (0, 0, 1): .2,
        (0, 1, 0): .04,
        (0, 1, 1): .06,
        (1, 0, 0): .125,
        (1, 0, 1): .125,
        (1, 1, 0): .125,
        (1, 1, 1): .125,
    }
    mock_rv = SparseDiscreteTable(rv_names=mock_rv_names, table=mock_table)
    return mock_rv


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

        # test the operational functions
        p_y_given_x = p_xy / p_x
        p_x_given_y = p_xy / p_y

        self.assertEqual(p_xy, p_y_given_x * p_x)
        self.assertEqual(p_xy, p_x_given_y * p_y)

        # test the built in methods
        p_y_given_x = p_xy.conditional(['y'], ['x'])
        p_x_given_y = p_xy.conditional(['x'], ['y'])

        self.assertEqual(p_xy, p_y_given_x * p_x)
        self.assertEqual(p_xy, p_x_given_y * p_y)

    def test_entropy(self):
        p_xy = get_mock_p_xy()
        h = p_xy.entropy()
        print("p_xy", p_xy)
        print("p_xy.entropy()", h)
        self.assertAlmostEqual(h, 1.721928094887362)

    def test_conditional_entropy(self):
        p_xy = get_mock_p_xy()

        hc_xy = p_xy.conditional_entropy(rv_names_0=['x'],
                                         rv_names_condition=['y'])

        print(hc_xy)
        self.assertAlmostEqual(hc_xy, 0.7219280948873621)

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

    def test_conditional_mutual_information(self):
        """                        +-                                   -+
                                   |                 +-               -+ |
                                   |                 | p(z) * p(x,y,z) | |
        I(X;Y|Z) = SUM_xSUM_ySUM_z | p(x,y,z) * log2 |-----------------| |
                                   |                 | p(x,z) * p(y,z) | |
                                   |                 +-               -+ |
                                   +-                                   -+
        """
        p_xyz = get_mock_p_xyz()
        p_yz = p_xyz.marginal(['y', 'z'])
        p_xz = p_xyz.marginal(['x', 'z'])
        p_z = p_xyz.marginal(['z'])

        implicit_conditional_mi = p_xyz.conditional_mutual_information(rv_names_0=['x'],
                                                                       rv_names_1=['y'],
                                                                       rv_names_condition=['z'])

        implicit_conditional_mi_alt = p_xyz.conditional_mutual_information_alt(rv_names_0=['x'],
                                                                               rv_names_1=['y'],
                                                                               rv_names_condition=['z'])
        # todo profle explicit vs implicit vs alt
        explicit_conditional_mi = 0
        for k, v in p_xyz.table.items():
            x, y, z = k
            p_xyz_val = p_xyz.table[x, y, z]
            p_yz_val = p_yz.table[y, z]
            p_xz_val = p_xz.table[x, z]
            p_z_val = p_z.table[z,]

            explicit_conditional_mi += p_xyz_val * np.log2(
                (p_z_val * p_xyz_val) / (p_xz_val * p_yz_val))

        print("explicit_conditional_mi", explicit_conditional_mi)  # 0.07489542832637161
        print("implicit_conditional_mi", implicit_conditional_mi)  # 0.07489542832637164
        print("implicit_conditional_mi_alt", implicit_conditional_mi_alt)  # 0.07489542832637142

        self.assertAlmostEqual(explicit_conditional_mi, implicit_conditional_mi)
        self.assertAlmostEqual(explicit_conditional_mi, implicit_conditional_mi_alt)
        self.assertAlmostEqual(implicit_conditional_mi_alt, implicit_conditional_mi_alt)

    def test_get_item(self):
        p_xy = get_mock_p_xy()
        print("p_xy['x']", p_xy['x'])
        self.assertEqual(p_xy['x'], p_xy.marginal(['x']))

        print("p_xy['x','y']", p_xy['x', 'y'])
        self.assertEqual(p_xy['x', 'y'], p_xy.marginal(['x', 'y']))
