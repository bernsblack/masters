

from dataloaders.grid_loader import GridDataLoaders, reconstruct_from_grid_loader
from datasets.grid_dataset import GridDataGroup

import unittest

from utils.configs import BaseConf

import os
from logger.logger import setup_logging
import logging as log

class TestGridDataLoaderIndexing(unittest.TestCase):

    def test_grid_loaders_reconstruction(self):
        data_sub_path = 'T24H-X850M-Y880M_2012-01-01_2019-01-01_#826'
        conf = BaseConf()
        conf.model_name = "test"
        conf.data_path = f"./data/processed/{data_sub_path}/"

        conf.model_path = f"{conf.data_path}models/{conf.model_name}/"
        os.makedirs(conf.data_path, exist_ok=True)
        os.makedirs(conf.model_path, exist_ok=True)

        # logging config is set globally thus we only need to call this in this file
        # imported function logs will follow the configuration
        setup_logging(save_dir=conf.model_path, log_config='./logger/standard_logger_config.json',
                      default_level=log.INFO)
        log.info("=====================================BEGIN=====================================")

        grid_data_group = GridDataGroup(data_path=conf.data_path, conf=conf)
        grid_loaders = GridDataLoaders(data_group=grid_data_group, conf=conf)

        y_counts, reconstructed_targets, t_range = reconstruct_from_grid_loader(grid_loaders.test_loader)

        self.assertTrue((y_counts == reconstructed_targets).all())
        self.assertEqual(len(y_counts), len(reconstructed_targets))
        self.assertEqual(len(t_range), len(reconstructed_targets))

        y_counts, reconstructed_targets, t_range = reconstruct_from_grid_loader(grid_loaders.validation_loader)

        self.assertTrue((y_counts == reconstructed_targets).all())
        self.assertEqual(len(y_counts), len(reconstructed_targets))
        self.assertEqual(len(t_range), len(reconstructed_targets))

        y_counts, reconstructed_targets, t_range = reconstruct_from_grid_loader(grid_loaders.train_loader)

        self.assertTrue((y_counts == reconstructed_targets).all())
        self.assertEqual(len(y_counts), len(reconstructed_targets))
        self.assertEqual(len(t_range), len(reconstructed_targets))
