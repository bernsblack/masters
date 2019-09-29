import unittest
from datasets.flat_dataset import FlatDataset
from utils.preprocessing import Shaper
from utils.mock_data import generate_mock_data
import numpy as np


# todo test for gird and flat data
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
    N, C, H, W = 31, 3, 17, 11
    L = H * W

    crime_feature_indices_shape = (C,)
    crime_types_grids_shape = (N, C, H, W)
    total_crimes_shape = (N, 1)
    crime_grids_shape = (N, 1, H, W)
    demog_grid_shape = (1, 37, H, W)
    street_grid_shape = (1, 512, H, W)
    time_vectors_shape = (N + 1, 44)
    weather_vectors_shape = (N, C)
    x_range_shape = (W,)
    y_range_shape = (H,)
    t_range_shape = (N + 1,)

    original_crime_data = generate_mock_data(crime_types_grids_shape)
    shaper = Shaper(original_crime_data)

    crimes = shaper.squeeze(original_crime_data)
    targets = np.copy(crimes)[1:, 0]
    total_crimes = generate_mock_data(total_crimes_shape)
    t_range = generate_mock_data(t_range_shape)  # t_range is matched to the target index
    time_vectors = generate_mock_data(time_vectors_shape)
    weather_vectors = generate_mock_data(weather_vectors_shape)
    demog_grid = generate_mock_data(demog_grid_shape)
    street_grid = generate_mock_data(street_grid_shape)
    seq_len = 4

    mock_dataset = FlatDataset(
        crimes=crimes,
        targets=targets,
        total_crimes=total_crimes,
        t_range=t_range,
        time_vectors=time_vectors,
        weather_vectors=weather_vectors,
        demog_grid=demog_grid,
        street_grid=street_grid,
        seq_len=seq_len,
        shaper=shaper,
    )

    def test_single_index(self):
        self.assertEqual(1,1)


if __name__ == "__main__":
    unittest.main()
