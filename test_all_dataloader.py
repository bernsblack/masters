import unittest
from utils.utils import by_ref
from dataloaders.cell_loader import CellDataLoaders
from dataloaders.flat_loader import FlatDataLoaders
from datasets.cell_dataset import CellDataGroup
from datasets.flat_dataset import FlatDataGroup
from datasets.grid_dataset import GridDataGroup
from dataloaders.grid_loader import GridDataLoaders, reconstruct_from_grid_loader
from dataloaders.flat_loader import reconstruct_from_flat_loader
from dataloaders.cell_loader import reconstruct_from_cell_loader
from utils.setup import setup
import numpy as np


class TestAllDataLoaderIndexing(unittest.TestCase):

    def test_all_loaders_reconstruction_without_crime_types(self):
        data_sub_path = by_ref("c97")[0]
        print(f"using: {data_sub_path}")

        conf, shaper, sparse_crimes, crime_feature_indices = setup(data_sub_path, 'test')

        conf.use_classification = False

        conf.use_crime_types = False

        grid_data_group = GridDataGroup(data_path=conf.data_path, conf=conf)
        grid_loaders = GridDataLoaders(data_group=grid_data_group, conf=conf)

        flat_data_group = FlatDataGroup(data_path=conf.data_path, conf=conf)
        flat_loaders = FlatDataLoaders(data_group=flat_data_group, conf=conf)

        cell_data_group = CellDataGroup(data_path=conf.data_path, conf=conf)
        cell_loaders = CellDataLoaders(data_group=cell_data_group, conf=conf)

        shaper = flat_data_group.shaper

        flat_trg, flat_trg_rcn, flat_t_range = reconstruct_from_flat_loader(flat_loaders.test_loader)
        grid_trg, grid_trg_rcn, grid_t_range = reconstruct_from_grid_loader(grid_loaders.test_loader)
        cell_trg, cell_trg_rcn, cell_t_range = reconstruct_from_cell_loader(cell_loaders.test_loader)

        og_trg = sparse_crimes[-len(flat_t_range):, 0:1]
        og_trg = shaper.squeeze(og_trg)[:, 0]

        self.assertTrue((flat_trg == flat_trg_rcn).all())
        self.assertTrue((cell_trg == cell_trg_rcn).all())
        self.assertTrue((grid_trg == grid_trg_rcn).all())
        self.assertTrue((flat_t_range == grid_t_range).all())
        self.assertTrue((flat_t_range == cell_t_range).all())

        flat_trg_sparse = shaper.unsqueeze(flat_trg)[:, 0]
        grid_trg_dense = shaper.squeeze(grid_trg)
        cell_trg_dense = shaper.squeeze(cell_trg)

        f_spr = shaper.unsqueeze(flat_trg)
        g_spr = grid_trg
        c_spr = cell_trg

        f_dns = flat_trg
        g_dns = shaper.squeeze(grid_trg)
        c_dns = shaper.squeeze(cell_trg)

        # test if the crime hotspots are the same - not the counts
        grid_ones = np.copy(grid_trg_dense)
        grid_ones[grid_ones > 0] = 1

        flat_ones = np.copy(flat_trg)
        flat_ones[flat_ones > 0] = 1

        cell_ones = np.copy(cell_trg_dense)
        cell_ones[cell_ones > 0] = 1

        self.assertTrue((grid_ones == flat_ones).all())
        self.assertTrue((cell_ones == flat_ones).all())

        print(f"{'' if conf.use_crime_types else 'not'} using crime types")

        g_inv = grid_data_group.target_scaler.inverse_transform(g_spr)
        g_inv = shaper.squeeze(g_inv)[:, 0]
        f_inv = flat_data_group.target_scaler.inverse_transform(f_dns)[:, 0]
        c_inv = cell_data_group.target_scaler.inverse_transform(c_dns)[:, 0]

        self.assertEqual(hash(str(flat_data_group.crime_scaler.__dict__)),
                         hash(str(cell_data_group.crime_scaler.__dict__)))

        g_og = np.round(2 ** g_inv - 1)
        f_og = np.round(2 ** f_inv - 1)
        c_og = np.round(2 ** c_inv - 1)

        f_series = f_og.sum(-1)
        g_series = g_og.sum(-1)
        c_series = c_og.sum(-1)
        og_series = og_trg.sum(-1)

        self.assertTrue((f_series == g_series).all())
        self.assertTrue((f_series == c_series).all())
        self.assertTrue((f_series == og_series).all())
        self.assertTrue((og_trg == g_og).all())
        self.assertTrue((og_trg == c_og).all())
        self.assertTrue((og_trg == f_og).all())

    def test_all_loaders_reconstruction_with_crime_types(self):
        data_sub_path = by_ref("c97")[0]
        print(f"using: {data_sub_path}")

        conf, shaper, sparse_crimes, crime_feature_indices = setup(data_sub_path, 'test')

        conf.use_classification = False
        conf.use_crime_types = True

        grid_data_group = GridDataGroup(data_path=conf.data_path, conf=conf)
        grid_loaders = GridDataLoaders(data_group=grid_data_group, conf=conf)

        flat_data_group = FlatDataGroup(data_path=conf.data_path, conf=conf)
        flat_loaders = FlatDataLoaders(data_group=flat_data_group, conf=conf)

        cell_data_group = CellDataGroup(data_path=conf.data_path, conf=conf)
        cell_loaders = CellDataLoaders(data_group=cell_data_group, conf=conf)

        shaper = flat_data_group.shaper

        self.assertTrue((grid_loaders.data_group.targets == shaper.unsqueeze(cell_loaders.data_group.targets)).all())
        self.assertTrue(
            (grid_loaders.data_group.crimes[:, 0] == shaper.unsqueeze(cell_loaders.data_group.crimes)[:, 0]).all())
        self.assertTrue((grid_loaders.data_group.labels == shaper.unsqueeze(cell_loaders.data_group.labels)).all())

        self.assertTrue((grid_loaders.data_group.targets == shaper.unsqueeze(flat_loaders.data_group.targets)).all())
        self.assertTrue(
            (grid_loaders.data_group.crimes[:, 0] == shaper.unsqueeze(flat_loaders.data_group.crimes)[:, 0]).all())
        self.assertTrue((grid_loaders.data_group.labels == shaper.unsqueeze(flat_loaders.data_group.labels)).all())

        flat_trg, flat_trg_rcn, flat_t_range = reconstruct_from_flat_loader(flat_loaders.test_loader)
        grid_trg, grid_trg_rcn, grid_t_range = reconstruct_from_grid_loader(grid_loaders.test_loader)
        cell_trg, cell_trg_rcn, cell_t_range = reconstruct_from_cell_loader(cell_loaders.test_loader)

        g_count = grid_data_group.to_counts(grid_trg_rcn)
        f_count = flat_data_group.to_counts(flat_trg_rcn)
        c_count = cell_data_group.to_counts(cell_trg_rcn)

        g_cd = shaper.squeeze(g_count)
        c_cd = shaper.squeeze(c_count)
        f_cd = f_count
        o_cd = shaper.squeeze(sparse_crimes[-len(f_count):, 0:1])

        self.assertTrue((o_cd == g_cd).all())
        self.assertTrue((o_cd == f_cd).all())
        self.assertTrue((o_cd == c_cd).all())

        og_trg = sparse_crimes[-len(flat_t_range):, 0:1]
        og_trg = shaper.squeeze(og_trg)[:, 0]

        self.assertTrue((flat_trg == flat_trg_rcn).all())
        self.assertTrue((cell_trg == cell_trg_rcn).all())
        self.assertTrue((grid_trg == grid_trg_rcn).all())
        self.assertTrue((flat_t_range == grid_t_range).all())
        self.assertTrue((flat_t_range == cell_t_range).all())

        flat_trg_sparse = shaper.unsqueeze(flat_trg)[:, 0]
        grid_trg_dense = shaper.squeeze(grid_trg)
        cell_trg_dense = shaper.squeeze(cell_trg)

        f_spr = shaper.unsqueeze(flat_trg)
        g_spr = grid_trg
        c_spr = cell_trg

        f_dns = flat_trg
        g_dns = shaper.squeeze(grid_trg)
        c_dns = shaper.squeeze(cell_trg)

        # test if the crime hotspots are the same - not the counts
        grid_ones = np.copy(grid_trg_dense)
        grid_ones[grid_ones > 0] = 1

        flat_ones = np.copy(flat_trg)
        flat_ones[flat_ones > 0] = 1

        cell_ones = np.copy(cell_trg_dense)
        cell_ones[cell_ones > 0] = 1

        self.assertTrue((grid_ones == flat_ones).all())
        self.assertTrue((cell_ones == flat_ones).all())

        print(f"{'' if conf.use_crime_types else 'not'} using crime types")

        g_inv = grid_data_group.target_scaler.inverse_transform(g_spr)
        g_inv = shaper.squeeze(g_inv)[:, 0]
        f_inv = flat_data_group.target_scaler.inverse_transform(f_dns)[:, 0]
        c_inv = cell_data_group.target_scaler.inverse_transform(c_dns)[:, 0]

        self.assertEqual(hash(str(flat_data_group.crime_scaler.__dict__)),
                         hash(str(cell_data_group.crime_scaler.__dict__)))

        g_og = np.round(2 ** g_inv - 1)
        f_og = np.round(2 ** f_inv - 1)
        c_og = np.round(2 ** c_inv - 1)

        f_series = f_og.sum(-1)
        g_series = g_og.sum(-1)
        c_series = c_og.sum(-1)
        og_series = og_trg.sum(-1)

        self.assertTrue((f_series == g_series).all())
        self.assertTrue((f_series == c_series).all())
        self.assertTrue((f_series == og_series).all())
        self.assertTrue((og_trg == g_og).all())
        self.assertTrue((og_trg == c_og).all())
        self.assertTrue((og_trg == f_og).all())


if __name__ == "__main__":
    unittest.main()
