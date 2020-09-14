import unittest

from sparse_discrete_table import SparseDiscreteTable, build_discrete_table
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

        self.assertAlmostEqual(mi_xy_explicit, mi_xy_implicit)

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
        # todo profile explicit vs implicit vs alt
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

    def test_mutual_information_normalized(self):
        decimal_place = 5

        obs_0 = np.array([[0, 0, 0, 0], [0, 1, 2, 3]]).T
        obs_1 = np.array([[0, 0, 1, 1], [0, 0, 0, 1]]).T
        obs_2 = np.array([[0, 0, 1, 1, 2], [1, 1, 0, 0, 2]]).T

        dt_0 = build_discrete_table(obs_0, ['x', 'y'])
        dt_1 = build_discrete_table(obs_1, ['x', 'y'])
        dt_2 = build_discrete_table(obs_2, ['x', 'y'])

        mi_0 = dt_0.mutual_information(['x'], ['y'])
        mi_1 = dt_1.mutual_information(['x'], ['y'])
        mi_2 = dt_2.mutual_information(['x'], ['y'])

        nmi_0 = dt_0.mutual_information(['x'], ['y'], normalize=True)
        nmi_1 = dt_1.mutual_information(['x'], ['y'], normalize=True)
        nmi_2 = dt_2.mutual_information(['x'], ['y'], normalize=True)

        self.assertAlmostEqual(0.0,mi_0,decimal_place)
        self.assertAlmostEqual(0.31127812445913294,mi_1,decimal_place)
        self.assertAlmostEqual(1.5219280948873621,mi_2,decimal_place)
        self.assertAlmostEqual(0.0,nmi_0,decimal_place)
        self.assertAlmostEqual(0.3437110184854509,nmi_1,decimal_place)
        self.assertAlmostEqual(1.0,nmi_2,decimal_place)

    def test_normalized_mutual_information(self):
        decimal_place = 5

        obs_0 = np.array([[0, 0, 0, 0], [0, 1, 2, 3]]).T
        obs_1 = np.array([[0, 0, 1, 1], [0, 0, 0, 1]]).T
        obs_2 = np.array([[0, 0, 1, 1, 2], [1, 1, 0, 0, 2]]).T

        dt_0 = build_discrete_table(obs_0, ['x', 'y'])
        dt_1 = build_discrete_table(obs_1, ['x', 'y'])
        dt_2 = build_discrete_table(obs_2, ['x', 'y'])

        mi_0 = dt_0.mutual_information(['x'], ['y'])
        mi_1 = dt_1.mutual_information(['x'], ['y'])
        mi_2 = dt_2.mutual_information(['x'], ['y'])

        nmi_0 = dt_0.normalized_mutual_information(['x'], ['y'])
        nmi_1 = dt_1.normalized_mutual_information(['x'], ['y'])
        nmi_2 = dt_2.normalized_mutual_information(['x'], ['y'])

        self.assertAlmostEqual(0.0,mi_0,decimal_place)
        self.assertAlmostEqual(0.31127812445913294,mi_1,decimal_place)
        self.assertAlmostEqual(1.5219280948873621,mi_2,decimal_place)
        self.assertAlmostEqual(0.0,nmi_0,decimal_place)
        self.assertAlmostEqual(0.3437110184854509,nmi_1,decimal_place)
        self.assertAlmostEqual(1.0,nmi_2,decimal_place)



class TestConditionalMutualInformation(unittest.TestCase):

    def test_conditional_references(self):
        """
        MI(X,Y) = 0.133086
        MI(X,Y|Z=0) = 0.000852316

        H(Y) = 1.48242
        H(Y|Z=0) = 1.15378
        """
        decimal_place = 5

        rv_names = ['x', 'y', 'z']
        arr_zxy = np.loadtxt('zxy.txt')
        arr_x, arr_y, arr_z = arr_zxy[:, 1], arr_zxy[:, 2], arr_zxy[:, 0]

        arr_xyz = np.stack([arr_x, arr_y, arr_z], axis=1)  # xyz
        arr_xyz0 = arr_xyz[arr_z == 0]
        arr_xyz1 = arr_xyz[arr_z == 1]

        dt_xyz = build_discrete_table(obs_arr=arr_xyz, rv_names=rv_names)
        dt_xyz0 = build_discrete_table(obs_arr=arr_xyz0, rv_names=rv_names)
        # dt_xyz1 = build_discrete_table(obs_arr=arr_xyz1, rv_names=rv_names)

        mi = dt_xyz.mutual_information(rv_names_0=['x'], rv_names_1=['y'])
        hy = dt_xyz.marginal(['y']).entropy()

        mi_xyz0 = dt_xyz0.mutual_information(rv_names_0=['x'], rv_names_1=['y'])
        hy_xyz0 = dt_xyz0.marginal(['y']).entropy()

        # mi_xyz1 = dt_xyz1.mutual_information(rv_names_0=['x'], rv_names_1=['y'])
        # hy_xyz1 = dt_xyz1.marginal(['y']).entropy()
        mic_z = dt_xyz.conditional_mutual_information(rv_names_0=['x'], rv_names_1=['y'], rv_names_condition=['z'])

        self.assertAlmostEqual(0.00074935913348, mic_z, places=decimal_place)

        self.assertAlmostEqual(0.133086, mi, places=decimal_place)
        self.assertAlmostEqual(0.000852316, mi_xyz0, places=decimal_place)
        self.assertAlmostEqual(1.48242, hy, places=decimal_place)
        self.assertAlmostEqual(1.15378, hy_xyz0, places=decimal_place)

class TestQuickMutualInformation(unittest.TestCase):
    def test_quick_mutual_info(self):
        """
        MI(X,Y) = 0.133086
        MI(X,Y|Z=0) = 0.000852316

        H(Y) = 1.48242
        H(Y|Z=0) = 1.15378
        """
        decimal_place = 5

        arr_zxy = np.loadtxt('zxy.txt')
        arr_x, arr_y, arr_z = arr_zxy[:, 1], arr_zxy[:, 2], arr_zxy[:, 0]

        from sparse_discrete_table import quick_mutual_info

        mi = quick_mutual_info(arr_x, arr_y)
        nmi = quick_mutual_info(arr_x, arr_x, True)

        self.assertAlmostEqual(0.133086, mi, places=decimal_place)
        self.assertAlmostEqual(1.0, nmi, places=decimal_place)

    def test_quick_cond_mutual_info(self):
        """
        MI(X,Y) = 0.133086
        MI(X,Y|Z=0) = 0.000852316

        H(Y) = 1.48242
        H(Y|Z=0) = 1.15378
        """
        decimal_place = 5

        arr_zxy = np.loadtxt('zxy.txt')
        arr_x, arr_y, arr_z = arr_zxy[:, 1], arr_zxy[:, 2], arr_zxy[:, 0]

        from sparse_discrete_table import quick_cond_mutual_info

        cmi = quick_cond_mutual_info(arr_x, arr_y, arr_z)
        cmi_normed = quick_cond_mutual_info(arr_x, arr_y, arr_z, norm=True)
        cmi_normed_self = quick_cond_mutual_info(arr_x, arr_x, arr_z, norm=True)

        self.assertAlmostEqual(0.00074935913348, cmi, places=decimal_place)
        self.assertAlmostEqual(0.00065854846636, cmi_normed, places=decimal_place)
        self.assertAlmostEqual(1.0, cmi_normed_self, places=decimal_place)


