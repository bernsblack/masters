import unittest

import numpy as np

from datasets.flat_dataset import FlatDataset
from utils.configs import BaseConf
from utils.preprocessing import Shaper


class TestDatasetIndexing(unittest.TestCase):
    """
    crime_feature_indices_shape = (C,)
    crime_types_grids_shape = (N, C, H, W)
    crime_grids_shape = (N, 1, H, W)
    demog_grid_shape = (1, 37, H, W)
    street_grid_shape = (1, 512, H, W)
    time_vectors_shape = (N + 1, 44)
    weather_vectors_shape = (N, C)
    x_range_shape = (W,)
    y_range_shape = (H,)
    t_range_shape = (N + 1,)
    """

    def setUp(self) -> None:
        N, C, H, W = 100, 3, 17, 11
        seq_len = 4
        offset_year = 13

        conf = BaseConf()

        # crime_feature_indices_shape = (C,)
        crime_types_grids_shape = (N, C, H, W)
        total_crimes_shape = (N, 1)
        # crime_grids_shape = (N, 1, H, W)
        demog_grid_shape = (1, 37, H, W)
        street_grid_shape = (1, 512, H, W)
        time_vectors_shape = (N + 1, 44)
        weather_vectors_shape = (N, C)
        # x_range_shape = (W,)
        # y_range_shape = (H,)
        t_range_shape = (N + 1,)

        original_crime_data = np.ones(crime_types_grids_shape)
        shaper = Shaper(original_crime_data, conf)

        crimes = shaper.squeeze(original_crime_data)
        targets = np.copy(crimes)[1:, 0:1]
        labels = np.copy(targets)
        labels[labels > 0] = 1

        total_crimes = np.ones(total_crimes_shape)
        t_range = np.ones(t_range_shape)  # t_range is matched to the target index
        time_vectors = np.ones(time_vectors_shape)
        # weather_vectors = np.ones(weather_vectors_shape)
        demog_grid = shaper.squeeze(np.ones(demog_grid_shape))
        street_grid = shaper.squeeze(np.ones(street_grid_shape))

        self.mock_dataset = FlatDataset(
            crimes=crimes,
            targets=targets,
            labels=labels,
            total_crimes=total_crimes,
            t_range=t_range,
            time_vectors=time_vectors,
            # weather_vectors=weather_vectors,
            demog_grid=demog_grid,
            street_grid=street_grid,
            seq_len=seq_len,
            offset_year=offset_year,
            shaper=shaper,
        )

    def test_min_index_lt_error(self):
        with self.assertRaises(IndexError):
            _ = self.mock_dataset[self.mock_dataset.min_index - 1]

    def test_min_index_eq(self):
        out = self.mock_dataset[self.mock_dataset.min_index]
        self.assertEqual(True, isinstance(out, tuple))

    def test_max_index_lt(self):
        out = self.mock_dataset[self.mock_dataset.max_index - 1]
        self.assertEqual(True, isinstance(out, tuple))

    def test_max_index_eq_error(self):
        with self.assertRaises(IndexError):
            _ = self.mock_dataset[self.mock_dataset.max_index]


if __name__ == "__main__":
    unittest.main()
