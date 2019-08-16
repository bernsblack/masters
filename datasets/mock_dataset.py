from torch.utils.data import Dataset


class MockDataset(Dataset):
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
