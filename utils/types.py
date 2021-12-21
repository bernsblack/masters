from typing import Optional, Union, Dict

from torch import optim

from dataloaders.grid_loader import BaseDataLoaders
from datasets.sequence_dataset import SequenceDataLoaders
from numpy import ndarray

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


# todo: define Array Types types for N,C,H,W and N,C,L - there seems to be conflict between distributed vs whole city
#  array types type aliases to make understanding the code easier. For distribution models L = flattened H,W. But, 
#  for whole city models L = sequence length 
class ArrayNCHW(ndarray):
    """
    N: the number of samples in batch
    C: number of channels/features
    H: picture/slice/frame Height
    W: picture/slice/frame Width
    """
    pass


class ArrayN1HW(ndarray):
    """
    N: the number of samples in batch
    C: number of channels/features - number of channels is always 1
    H: picture/slice/frame Height
    W: picture/slice/frame Width
    """
    pass


class ArrayHWC(ndarray):
    """
    H: picture/slice/frame Height
    W: picture/slice/frame Width
    C: number of channels/features
    """
    pass


class ArrayN(ndarray):
    """
    N: the number of samples in batch
    Array with shape (N,)
    """
    pass


class ArrayNCL(ndarray):
    """
    N: the number of samples in batch
    C number of channels/features
    L: flattened 2D data from H, W
    """
    pass


class ArrayNC(ndarray):
    """
    N: the number of samples in batch
    C number of channels/features
    """
    pass


class ArrayN1L(ndarray):
    """
    N: the number of samples in batch
    C: number of channels/features - number of channels is always 1
    L: flattened 2D data from H, W
    """
    pass


class ArrayNL(ndarray):
    """
    N: the number of samples in batch
    L: flattened 2D data from H, W
    """
    pass


class ArrayLC(ndarray):
    """
    L: flattened 2D data from H, W
    C number of channels/features
    """
    pass


class ArrayNHW(ndarray):
    """
    N: the number of samples in batch
    H: picture/slice/frame Height
    W: picture/slice/frame Width
    """
    pass
