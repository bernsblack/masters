import unittest

import numpy as np

from dataloaders.cell_loader import CellDataLoaders, reconstruct_from_cell_loader
from datasets.cell_dataset import CellDataGroup
from utils.configs import BaseConf
from utils.data_processing import crop4d


class TestCellDataLoaderIndexing(unittest.TestCase):

    def test_cell_loader_reconstruction(self):
        conf = BaseConf()
        data_path = './data/processed/T24H-X850M-Y880M_2012-01-01_2019-01-01_#826/'
        conf.sub_sample_test_set = 0
        conf.sub_sample_train_set = 0
        conf.sub_sample_validation_set = 0
        conf.seq_len = 1
        data_group = CellDataGroup(data_path=data_path, conf=conf)
        loaders = CellDataLoaders(data_group=data_group, conf=conf)

        y_true, reconstructed_targets, t_range = reconstruct_from_cell_loader(loaders.test_loader)

        self.assertTrue((y_true == reconstructed_targets).all())

    def test_test_loader_indices(self):
        # CRIME DATA
        conf = BaseConf()
        data_path = './data/processed/T24H-X850M-Y880M_2012-01-01_2019-01-01_#826/'
        conf.sub_sample_test_set = 0
        conf.sub_sample_train_set = 0
        conf.sub_sample_validation_set = 0
        conf.seq_len = 2
        conf.pad_width = 1
        data_group = CellDataGroup(data_path=data_path, conf=conf)

        targets_constructed = np.zeros(data_group.testing_set.target_shape)
        targets_len = len(targets_constructed)

        targets_original = crop4d(data_group.testing_set.targets[-targets_len:],
                                  data_group.testing_set.pad_width)
        crimes_original = crop4d(data_group.testing_set.crimes[-targets_len:],
                                 data_group.testing_set.pad_width)
        crimes_channels = crimes_original.shape[1]

        crimes_constructed = np.zeros(crimes_original.shape)

        loaders = CellDataLoaders(data_group=data_group, conf=conf)

        # shape for tmp_feats (seq_len, batch_len, n_feats)
        # shape for targets (seq_len, batch_len, 1)
        # shape for env_feats (1, batch_len, n_feats)
        # shape for spc_feats (1, batch_len, n_feats)

        for indices, spc_feats, tmp_feats, env_feats, targets, labels in loaders.test_loader:
            for i in range(len(indices)):
                n, c, h, w = indices[i]  # in this case: c == 0 always - targets have only one channel
                targets_constructed[n, c, h, w] = targets[-1, i]

                # extract original crimes from the tmp_vec
                crime_i = (2 * conf.pad_width * (conf.pad_width + 1))
                step = (2 * conf.pad_width + 1) ** 2
                crime_j = crimes_channels * step
                crimes_constructed[n, :, h, w] = tmp_feats[-1, i, crime_i:crime_j:step]  # n,c,h,w

        self.assertEqual(np.equal(targets_original, targets_constructed).all(), True)
        self.assertEqual(np.equal(crimes_original, crimes_constructed).all(), True)


if __name__ == "__main__":
    unittest.main()
