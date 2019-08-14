import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class GenericDataset(Dataset):
    """
    Characterizes a dataset for PyTorch
    """

    def __init__(self, features, labels):
        """Initialization"""
        self.labels = labels
        self.feats = features

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.feats)

    def __getitem__(self, index):
        """Generates one sample of data"""
        X = self.feats[index]
        y = self.labels[index]

        return X, y


# todo add set of valid / living sells / to be able to sample the right ones
class GenericCrimeDataSet(Dataset):
    """
    Generic data set that contains all data we need for our model
    Provide only with the folder where the data is stored
    """
    def __init__(self, data_folder):
        """
        Args:
            data_folder (string): Path to the data folder with all spatial and temporal data.
        """
        # todo normalise the data
        # todo cap the crime grids at a certain level - instead use np.log2(1 + x) to normalise
        # todo flatten the crime grids and normalise according to the features
        # dont use function to get values each time - create join and add table
        # [√] number of incidents of crime occurrence by sampling point in 2013 (1-D)
        # [√] number of incidents of crime occurrence by census tract in 2013 (1-D)
        # [√] number of incidents of crime occurrence by census tract yesterday (1-D).
        # [√] number of incidents of crime occurrence by date in 2013 (1-D)
        zip_file = np.load(data_folder + "generated_data.npz")
        # used in determine what columns in crime_types_grids represent
        self.crime_feature_indices = zip_file["crime_feature_indices"]
        self.crime_types_grids = zip_file["crime_types_grids"]
        self.crime_grids = zip_file["crime_grids"] # todo determine if it is necessary

        # getting the total sum over the whole data set
        self.total_crimes = self.crime_grids.sum(1).sum(1)  # or self.crime_types_grids[0].sum(1).sum(1)

        self.tract_count_grids = zip_file["tract_count_grids"]
        self.demog_grid = zip_file["demog_grid"]
        self.street_grid = zip_file["street_grid"]
        self.time_vectors = zip_file["time_vectors"]
        self.x_range = zip_file["x_range"]
        self.y_range = zip_file["y_range"]
        self.t_range = pd.read_pickle(data_folder + "t_range.pkl")

    def __len__(self):
        return len(self.crime_grids)

    def __getitem__(self, idx):
        return
