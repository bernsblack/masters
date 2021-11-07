from typing import Optional, Union, Dict

from torch import optim

from dataloaders.grid_loader import BaseDataLoaders
from datasets.sequence_dataset import SequenceDataLoaders

TParamValue = Optional[Union[str, bool, float, int]]
TParameterization = Dict[str, TParamValue]

LrScheduler = Union[
    optim.lr_scheduler.StepLR,
    optim.lr_scheduler.LambdaLR,
    optim.lr_scheduler.MultiStepLR,
    optim.lr_scheduler.CyclicLR,
    optim.lr_scheduler.CosineAnnealingLR,
    optim.lr_scheduler.ExponentialLR,
    optim.lr_scheduler.ReduceLROnPlateau,
]

TDataLoaders = Union[BaseDataLoaders, SequenceDataLoaders]
