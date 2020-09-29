from utils.utils import pshape, get_data_sub_paths, by_ref
from pprint import pprint
from utils.interactive import InteractiveHeatmaps
from dataloaders.cell_loader import CellDataLoaders
from dataloaders.flat_loader import FlatDataLoaders
from dataloaders.grid_loader import GridDataLoaders
from datasets.cell_dataset import CellDataGroup
from datasets.flat_dataset import FlatDataGroup
from datasets.grid_dataset import GridDataGroup
from dataloaders.grid_loader import GridDataLoaders, reconstruct_from_grid_loader
from dataloaders.flat_loader import reconstruct_from_flat_loader
from dataloaders.cell_loader import reconstruct_from_cell_loader
from utils.utils import describe_array
from utils.setup import setup
import pandas as pd

data_sub_path = by_ref("3a0")[0]
print(f"using: {data_sub_path}")

conf, shaper, sparse_crimes, crime_feature_indices = setup(data_sub_path, 'test')


conf.use_classification = False
conf.use_crime_types = True

t_range = pd.date_range(*conf.data_path.split('M_')[1].split('_')[:2])[:-1]

grid_data_group = GridDataGroup(data_path=conf.data_path,conf=conf)
grid_loaders = GridDataLoaders(data_group=grid_data_group,conf=conf)
print(grid_data_group.crimes.shape)

