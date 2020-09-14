# operations multiply, divide, sum
from __future__ import annotations

from typing import Callable, List, Dict, Union, Tuple
import numpy as np

from utils import deprecated

FLOAT_TOLERANCE_VALUE = 1e-8


def union_lists(list_0: List, list_1: List):
    return list(set(list_0 + list_1))


def almost_equal(x: float, y: float) -> bool:
    return abs(x - y) < FLOAT_TOLERANCE_VALUE


def add(x, y):
    return x + y


def sub(x, y):
    return x - y


def mul(x, y):
    return x * y


def div(x, y):
    return x / y


def normalize_table(table: Dict):
    """
    function is done in place
    """
    total = sum(table.values())
    if total == 0:
        raise ValueError("Sum of table values cannot be zero")

    for k, v in table.items():
        table[k] = v / total


def is_normalized_table(table: Dict):
    return 1 == sum(table.values())


def format_table(table: Dict) -> str:
    rows = []
    for k, v in table.items():
        rows.append(f"{k}: {v}")
    return "\n".join(rows)


def get_rv_names_intersection(rv_names_0: List[str], rv_names_1: List[str]):
    """
    returns a list of values that apear in rv_names_0,rv_names_1
    """
    return sorted(list(set(rv_names_0).intersection(set(rv_names_1))))


def get_rv_names_indices(rv_names_0: List[str], rv_names_1: List[str]):
    """
    returns the indices of the values of rv_names_1 in rv_names_0

    preferred that the order(rv_names_0) > order(rv_names_1)
    """
    intersection = get_rv_names_intersection(rv_names_0, rv_names_1)
    return sorted([rv_names_0.index(name) for name in intersection])


def apply_function(func: Callable, rv0: SparseDiscreteTable,
                   rv1: SparseDiscreteTable) -> SparseDiscreteTable:  # inplace=False
    """
    func: function in format of func(a: float,b: float) -> float
    rv0, rv1: discrete random variable
    returns a newly created object

    resulting rv_names will be of rv with biggest order
    """
    if rv0.get_order() < rv1.get_order():  # pointer swap only happens in scope of the function
        raise ValueError("rv0.get_order() < rv1.get_order()")
        # rv0, rv1 = rv1, rv0

    indices = get_rv_names_indices(rv0.rv_names, rv1.rv_names)
    new_table = {}
    for rv0_k, rv0_v in rv0.table.items():
        rv1_k = tuple([rv0_k[i] for i in indices])
        rv1_v = rv1.table.get(rv1_k)
        if rv1_v:  # is not None
            new_table[rv0_k] = func(rv0_v, rv1_v)
    return SparseDiscreteTable(rv_names=rv0.rv_names, table=new_table)


def entropy(table: Dict):
    arr = np.array(list(table.values()))
    arr_filtered = arr[arr > 0]
    return -1 * np.sum(arr_filtered * np.log2(arr_filtered))


class SparseDiscreteTable:
    """
    rv_names: list of random variable names, e.g.
            ['X','Y','Z']
    table: dictionary where key is tupple of rv names and value is the probability, e.g.
            {
                (0,0):.15,
                (0,1):.13,
                (1,0):.2,
                (1,1):.7,
            }

    is only a discrete table not a probability mass function - values do not need to sum to 1
    """

    def __init__(self, rv_names: List[str], table: Dict):
        """
        :type rv_names: List[str]
        :arg rv_names:  list of random variable names, e.g. ['X','Y','Z']
        :arg table: sparse probability mass function in a dictionary format with tuple of values as keys
        :type table: Dict
        """

        if isinstance(rv_names, list) and sorted(rv_names) != rv_names:
            raise ValueError(f"rv_names must be sorted => {sorted(rv_names)} != {rv_names}")

        # cannot be guaranteed in calculations because of floating point issues
        # if not is_normalized_table(table):
        #     raise ValueError("table must be normalized")

        self.rv_names = rv_names
        self.table = table

    def marginal(self, rv_names: List[str]) -> SparseDiscreteTable:
        if self.rv_names == rv_names:
            return self
            # return self.copy()
            # return SparseDiscreteTable(rv_names=self.rv_names.copy(), table=self.table.copy())

        indices = get_rv_names_indices(self.rv_names, rv_names)
        new_table = {}
        for old_k, old_v in self.table.items():
            new_k = tuple([old_k[i] for i in indices])
            new_table[new_k] = old_v + new_table.get(new_k, 0)
        return SparseDiscreteTable(rv_names=rv_names, table=new_table)

    def normalize(self) -> None:
        normalize_table(self.table)

    def apply_inplace(self, func: Callable, ext_rv: SparseDiscreteTable) -> SparseDiscreteTable:  # inplace=False
        """
        f: function in format of f(a: float,b: float) -> float
        ext_rv: external discrete random variable, where order is at most equal to self

        modifies the current object - returns nothing
        """
        if self.get_order() < ext_rv.get_order():
            raise ValueError("ext_rv order should at most equal to self")

        indices = get_rv_names_indices(self.rv_names, ext_rv.rv_names)
        for old_k, old_v in self.table.items():
            ext_k = tuple([old_k[i] for i in indices])
            ext_v = ext_rv.table.get(ext_k)
            if ext_v:  # is not None
                self.table[old_k] = func(old_v, ext_v)
            else:
                # remove the key and value because it is not present in ext_rv - same as multiplying with zero
                self.table.pop(old_k)
        return self

    def apply(self, func: Callable, ext_rv: SparseDiscreteTable) -> SparseDiscreteTable:  # inplace=False
        """
        func: function in format of f(a: float,b: float) -> float
        ext_rv: external discrete random variable, where order is at most equal to self

        returns a newly created object
        """
        return apply_function(func=func, rv0=self, rv1=ext_rv)

    def __getitem__(self, keys: Union[str, Tuple]):
        if isinstance(keys, str):
            rv_names = [keys]
        else:
            rv_names = list(keys)

        return self.marginal(rv_names)

    @deprecated
    def _condition(self, rv_names: List[str]):
        return self / self.marginal(rv_names=rv_names)

    def conditional(self,
                    rv_names_0: List[str],
                    rv_names_condition: List[str]):
        union_names = sorted(union_lists(rv_names_0, rv_names_condition))
        return self.marginal(rv_names=union_names) / self.marginal(rv_names=rv_names_condition)

    def entropy(self) -> float:
        """
        uses numpy
        :return: shannon entropy
        """
        return entropy(self.table)

    def conditional_entropy(self,
                            rv_names_0: List[str],
                            rv_names_condition: List[str]):
        """
        entropy is symmetric H(X,Y) = H(Y,X)
        conditional entropy H(X|Y) = H(X,Y) - H(Y)
        """
        union_names = sorted(union_lists(rv_names_0, rv_names_condition))
        p_0c = self.marginal(union_names)
        p_c = p_0c.marginal(rv_names_condition)
        return p_0c.entropy() - p_c.entropy()

    def conditional_mutual_information(self,
                                       rv_names_0: List[str],
                                       rv_names_1: List[str],
                                       rv_names_condition: List[str]):
        """
        uses entropy to calculate value
        conditional mutual information chain rule: I(X;Y|Z) = I(X;Y,Z) - I(X;Z)
        this will measure the mutual information between x and y given z

        :param rv_names_0: random variable name(s) we want to measure MI with
        :param rv_names_1: random variable name(s) we want to measure MI with
        :param rv_names_condition: random variable name(s) we want to condition the MI measurement with
        :return:
        """
        union_names = sorted(union_lists(rv_names_1, rv_names_condition))
        return self.mutual_information(rv_names_0, union_names) - self.mutual_information(rv_names_0,
                                                                                          rv_names_condition)

    def conditional_mutual_information_alt(self,
                                           rv_names_0: List[str],
                                           rv_names_1: List[str],
                                           rv_names_condition: List[str]):
        """
        uses mutual information to calculate value
        conditional mutual information identity alt.: I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
        :param rv_names_0: random variable name(s) we want to measure MI with
        :param rv_names_1: random variable name(s) we want to measure MI with
        :param rv_names_condition: random variable name(s) we want to condition the MI measurement with
        :return:
        """
        # alternative way of calculating cmi
        # todo check that none of the rv_names_condition are in rv_names_0 or rv_names_1
        union_names_0c = sorted(union_lists(rv_names_0, rv_names_condition))
        union_names_1c = sorted(union_lists(rv_names_1, rv_names_condition))
        union_names_01c = sorted(union_lists(union_names_0c, rv_names_1))

        p_01c = self.marginal(union_names_01c)
        p_0c = p_01c.marginal(union_names_0c)
        p_1c = p_01c.marginal(union_names_1c)
        p_c = p_1c.marginal(rv_names_condition)
        return p_0c.entropy() + p_1c.entropy() - p_01c.entropy() - p_c.entropy()  # more summing might lead to greater floating point errors

    def mutual_information(self, rv_names_0: List[str], rv_names_1: List[str], normalize=False):
        """
        Mutual information: the reduction in uncertainty about one random variable given knowledge of another
        Measures how much rv_0 tells us about rv_1 or vice-versa
        :param normalize: if the mutual information should be scaled to [0,1]
        :param rv_names_0:
        :param rv_names_1:
        :return: mutual information in bits (log2 is used in calculating the entropy)
        """
        union_names = sorted(union_lists(rv_names_0, rv_names_1))
        p_01 = self.marginal(union_names)
        h_01 = p_01.entropy()
        h_0 = p_01.marginal(rv_names=rv_names_0).entropy()
        h_1 = p_01.marginal(rv_names=rv_names_1).entropy()

        mi_01 = h_0 + h_1 - h_01

        if normalize:
            return 2 * mi_01 / (h_0 + h_1)
        else:
            return mi_01

    def normalized_mutual_information(self, rv_names_0: List[str], rv_names_1: List[str]):
        """
        Mutual information: the reduction in uncertainty about one random variable given knowledge of another
        Measures how much rv_0 tells us about rv_1 or vice-versa. By normalizing it we scale the min and max mutual
        information to be 1 and 0 respectively using: NI(X,I) = 2*I(X,Y)/[H(X) + H(Y)]
        :param rv_names_0:
        :param rv_names_1:
        :return: normalized mutual information value between 0 and 1
        """
        union_names = sorted(union_lists(rv_names_0, rv_names_1))
        p_01 = self.marginal(union_names)
        h_01 = p_01.entropy()
        h_0 = p_01.marginal(rv_names=rv_names_0).entropy()
        h_1 = p_01.marginal(rv_names=rv_names_1).entropy()

        return 2 * (h_0 + h_1 - h_01) / (h_0 + h_1)

    def self_information(self, rv_names_0: List[str]):
        """
        H(Y|Y) = 0
        I(Y,Y) = H(Y) - H(Y|Y) = H(Y)

        :param rv_names_0: variable names we want the measure the information shared with it self
        :return: entropy of rv_names_0
        """

        p_0 = self.marginal(rv_names=rv_names_0)
        return p_0.entropy()

    def get_order(self) -> int:
        return len(self.rv_names)

    def __len__(self) -> int:
        return len(self.table)

    def __repr__(self) -> str:
        return f'''        
{self.rv_names}        
{format_table(self.table)}
        '''

    def __eq__(self, other: SparseDiscreteTable) -> bool:
        if self.get_order() != other.get_order():
            return False

        if len(self) != len(other):
            return False

        union_keys = set(self.table.keys()).union(other.table.keys())
        for k in union_keys:
            if not almost_equal(self.table.get(k, None), other.table.get(k, None)):
                return False
        return True

    def __add__(self, other: SparseDiscreteTable) -> SparseDiscreteTable:
        return self.apply(func=add, ext_rv=other)

    def __sub__(self, other: SparseDiscreteTable) -> SparseDiscreteTable:
        return self.apply(func=sub, ext_rv=other)

    def __mul__(self, other: SparseDiscreteTable) -> SparseDiscreteTable:
        return self.apply(func=mul, ext_rv=other)

    def __truediv__(self, other: SparseDiscreteTable) -> SparseDiscreteTable:
        return self.apply(func=div, ext_rv=other)


def build_discrete_table(obs_arr: np.ndarray, rv_names: List[str]) -> SparseDiscreteTable:
    """
    obs_arr: array of observations N,D where N is number of observations and D the number of RVs
    rv_names: list of D rv names in same order as the obs_arr axis 1
    """
    val, cnt = np.unique(obs_arr, return_counts=True, axis=0)
    prb = cnt / np.sum(cnt)
    table = {}
    for k, v in zip(list(map(tuple, val)), list(prb)):
        table[k] = v

    return SparseDiscreteTable(rv_names=rv_names, table=table)


def new_discrete_table(**kwargs):
    """
    new_discrete_table will sort the kwargs by key

    kwargs names arrays where
    - keys are array name
    - value is the np.ndarray observations of the variabels
    """

    rv_names = sorted(kwargs.keys())
    stack = []
    for k in rv_names:
        stack.append(kwargs[k])

    obs_arr = np.stack(stack, axis=1)
    val, cnt = np.unique(obs_arr, return_counts=True, axis=0)
    prb = cnt / np.sum(cnt)
    table = {}
    for k, v in zip(list(map(tuple, val)), list(prb)):
        table[k] = v

    return SparseDiscreteTable(rv_names=rv_names, table=table)


# quick functions for mutual info and conditional mutual info
def quick_mutual_info(x, y, norm=False):
    """
    Determine the mutual information between x and y conditioned on z

    x: np.ndarray (N,d_x) with N observations of d_x dimensional vector
    y: np.ndarray (N,d_y) with N observations of d_y dimensional vector
    z: np.ndarray (N,d_z) with N observations of d_z dimensional vector
    norm: bool, if symmetric normalisation should be done using 0.5*(h(x)+h(y)) as normalising constant
    """
    dt = new_discrete_table(x=x, y=y)
    mi = dt.mutual_information(['x'], ['y'], norm)
    return mi


def quick_cond_mutual_info(x, y, z, norm=False):
    """
    Determine the mutual information between x and y conditioned on z

    x: np.ndarray (N,d_x) with N observations of d_x dimensional vector
    y: np.ndarray (N,d_y) with N observations of d_y dimensional vector
    z: np.ndarray (N,d_z) with N observations of d_z dimensional vector
    norm: bool, if asymmetric normalisation should be done using cmi(x,x,z) as normalising constant
    """
    p_xyz = new_discrete_table(x=x, y=y, z=z)
    p_xz = p_xyz['x', 'z']
    p_yz = p_xyz['y', 'z']
    p_z = p_yz['z']

    h_xyz = p_xyz.entropy()
    h_xz = p_xz.entropy()
    h_yz = p_yz.entropy()
    h_z = p_z.entropy()

    if norm:
        return (h_xz + h_yz - h_xyz - h_z) / (h_xz - h_z)
    else:
        return h_xz + h_yz - h_xyz - h_z
