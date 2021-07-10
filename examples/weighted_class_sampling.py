import torch
from torch.utils.data import sampler, DataLoader, Dataset

batch_size = 20
class_sample_count = [10, 1, 20, 3, 4]  # dataset has 10 class-1 samples, 1 class-2 samples, etc.
weights = 1 / torch.Tensor(class_sample_count)
sampler = sampler.WeightedRandomSampler(weights, batch_size)
train_dataset = Dataset()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
