import unittest
from datasets.flat_dataset import FlatDataset
from utils.preprocessing import Shaper
from utils.mock_data import generate_mock_data
import numpy as np

# todo (bernard) choose flat mock or grid mock
N, C, H, W = 31, 3, 17, 11
L = H * W
IS_FLAT = True

crime_feature_indices_shape = (C,)
crime_types_grids_shape = (N, C, H, W)
total_crimes_shape = (N, 1)
crime_grids_shape = (N, 1, H, W)  # only the totals when none of the crime types are being used
demog_grid_shape = (1, 37, H, W)
street_grid_shape = (1, 512, H, W)
time_vectors_shape = (N + 1, 44)
weather_vectors_shape = (N, C)
x_range_shape = (W,)
y_range_shape = (H,)
t_range_shape = (N + 1,)

shaper = None  # Shaper(original_crime_data)

crimes = generate_mock_data(crime_types_grids_shape)
targets = np.copy(crimes)[1:, 0]
total_crimes = generate_mock_data(total_crimes_shape)
t_range = generate_mock_data(t_range_shape)  # t_range is matched to the target index
time_vectors = generate_mock_data(time_vectors_shape)
weather_vectors = generate_mock_data(weather_vectors_shape)
demog_grid = generate_mock_data(demog_grid_shape)
street_grid = generate_mock_data(street_grid_shape)
seq_len = 1

if IS_FLAT:  # reshape
    crimes = np.reshape(crimes, (N, C, L))
    demog_grid = np.reshape(demog_grid, (1, 37, L))
    street_grid = np.reshape(street_grid, (1, 512, L))

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



# spc_feats: [demog_vec]
# tmp_feats: [time_vec, weather_vec, crime_vec]
# env_feats: [street_vec]
# targets: [targets]

spc_feats, tmp_feats, env_feats, targets = mock_dataset[mock_dataset.min_index]

print(f"spc_feats\n{spc_feats}\n")
print(f"tmp_feats\n{tmp_feats}\n")
print(f"env_feats\n{env_feats}\n")
print(f"targets\n{targets}\n")
