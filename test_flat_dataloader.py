import unittest
from dataloaders.flat_loader import FlatDataLoaders, reconstruct_from_flat_loader
import numpy as np
from dataloaders.flat_loader import FlatBatchLoader
from datasets.flat_dataset import FlatDataGroup
from utils.configs import BaseConf


class TestFlatDataLoaderIndexing(unittest.TestCase):

    def test_flat_loader_reconstruction(self):
        conf = BaseConf()
        data_path = './data/processed/T24H-X850M-Y880M_2013-01-01_2017-01-01/'
        conf.sub_sample_test_set = 0
        conf.sub_sample_train_set = 0
        conf.sub_sample_validation_set = 0
        conf.seq_len = 1
        data_group = FlatDataGroup(data_path=data_path, conf=conf)
        loaders = FlatDataLoaders(data_group=data_group, conf=conf)

        y_true, reconstructed_targets, t_range = reconstruct_from_flat_loader(loaders.test_loader)

        self.assertTrue((y_true == reconstructed_targets).all())

    def test_test_loader_indices(self):
        # CRIME DATA
        conf = BaseConf()
        data_path = './data/processed/T24H-X850M-Y880M_2013-01-01_2017-01-01/'
        conf.sub_sample_test_set = 0
        conf.sub_sample_train_set = 0
        conf.sub_sample_validation_set = 0
        conf.seq_len = 1
        data_group = FlatDataGroup(data_path=data_path, conf=conf)

        new_targets = np.ones(data_group.testing_set.targets.shape)
        c = np.expand_dims(np.expand_dims(np.arange(len(new_targets)), -1), -1)
        new_targets = c * new_targets
        data_group.testing_set.targets = new_targets

        probas_pred = np.zeros(data_group.testing_set.target_shape)
        y_true = data_group.testing_set.targets[-len(probas_pred):]
        ones = np.zeros(y_true.shape)

        loaders = FlatDataLoaders(data_group=data_group, conf=conf)

        count = 0
        for indices, spc_feats, tmp_feats, env_feats, targets in loaders.test_loader:
            for i in range(len(indices)):
                n, c, l = indices[i]
                probas_pred[n, c, l] = targets[-1, i]
                ones[n, c, l] = 1
                count += 1

        self.assertEqual(np.equal(y_true, probas_pred).all(), True)


if __name__ == "__main__":
    unittest.main()
