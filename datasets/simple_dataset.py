# sourced form https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
"""
DataLoader allows us to:
- Batch the data
- Shuffle the data
- Load the data in parallel using multiprocessing workers
"""


class SimpleDataSet(Dataset):
    """Simple example dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sample = self.data_frame.iloc[idx, 1:]
        sample = np.array([sample])
        sample = sample.astype('float').reshape(-1, 2)

        if self.transform:
            sample = self.transform(sample)  # transformation happens when the data is being fetched

        return sample


"""
DataLoader allows us to:
- Batch the data
- Shuffle the data
- Load the data in parallel using multiprocessing workers
- Save a list of valid cells, i.e. living cells on the class then when you give the index 
    you just mod the length of that and and spit out a coordinate value to use as index 
"""
dataset = SimpleDataSet("./data/original/Crimes_Chicago_2001_to_2019.csv")
dataloader = DataLoader(dataset, batch_size=4,
                        shuffle=True, num_workers=4)