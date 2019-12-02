import unittest

import numpy as np

from datasets.flat_dataset import FlatDataset
from datasets.grid_dataset import GridDataGroup
from utils.configs import BaseConf
from utils.preprocessing import Shaper


class TestDataGroup(unittest.TestCase):
    def test_input_target_offset(self):
        data_path = './data/processed/T1H-X1700M-Y1760M_2013-01-01_2017-01-01/'

        conf = BaseConf()
        data_group = GridDataGroup(data_path=data_path, conf=conf)

        self.assertTrue(np.equal(data_group.targets[:-1], data_group.crimes[1:]).all())
        self.assertTrue(np.equal(data_group.training_set.targets[:-1], data_group.training_set.crimes[1:]).all())
        self.assertTrue(np.equal(data_group.validation_set.targets[:-1], data_group.validation_set.crimes[1:]).all())
        self.assertTrue(np.equal(data_group.testing_set.targets[:-1], data_group.testing_set.crimes[1:]).all())

if __name__ == "__main__":
    unittest.main()
